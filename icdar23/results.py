"""
results.py
──────────
Evaluation metrics and result serialisation.

Public API
──────────
calc_foreign_map(page_writers, foreign_writers, dist_matrix, excluded_per_query)
    Asymmetric AP: queries are contaminated pages; relevant gallery entries are
    pages of the foreign writer, minus the single source page (FIX 2).

build_and_save(res_dict, label, pkl_suffix, csv_suffix,
               pooling_configs, realised_pct_per_level, mean_units_replaced,
               pages_per_level, n_valid_pages, extra_meta)
    Print a summary table, write a CSV and a pkl to OUT_DIR.
"""

import os, pickle
import numpy as np

import config


# ══════════════════════════════════════════════════════════════════════════════
# Foreign-writer mAP
# ══════════════════════════════════════════════════════════════════════════════

def calc_foreign_map(page_writers_arr, foreign_writers_arr, dist_matrix,
                     excluded_per_query=None):
    """
    For each query i contaminated with content from foreign_writers_arr[i],
    compute AP over gallery pages whose real writer == foreign_writers_arr[i].

    excluded_per_query : list[set[int]] | None
        Per-query sets of gallery indices to exclude entirely (from the relevant
        set AND from the ranking). Pass the gallery index of the single source
        page to prevent descriptor-leakage from inflating the metric (FIX 2).
        Excluded pages are pushed to the end of the ranking like the query itself;
        leaving them in would rank the source page first as a non-relevant hit
        and cap AP well below 1 at high contamination.
    """
    n = len(page_writers_arr)
    relevance = (page_writers_arr[None, :] == foreign_writers_arr[:, None])
    np.fill_diagonal(relevance, False)

    dist_copy = dist_matrix.copy().astype(np.float32)
    np.fill_diagonal(dist_copy, np.inf)

    if excluded_per_query is not None:
        for qi, excl in enumerate(excluded_per_query):
            for gi in excl:
                relevance[qi, gi] = False
                dist_copy[qi, gi] = np.inf

    n_relevant = relevance.sum(axis=1)
    sorted_idx = np.argsort(dist_copy, axis=1)

    rel_sorted = relevance[np.arange(n)[:, None], sorted_idx]
    cum_hits   = np.cumsum(rel_sorted, axis=1).astype(np.float32)
    ranks      = np.arange(1, n + 1, dtype=np.float32)[None, :]
    prec_at_r  = rel_sorted * cum_hits / ranks
    ap         = prec_at_r.sum(axis=1) / np.maximum(n_relevant, 1)

    valid    = n_relevant > 0
    map_val  = float(ap[valid].mean())                                 if valid.any() else 0.0
    top1_val = float(relevance[np.arange(n), sorted_idx[:, 0]][valid].mean()) \
               if valid.any() else 0.0

    # hard-k: fraction of valid queries with at least one relevant result in top-k
    # soft-5: same criterion at k=5 (relaxed / wider window)
    def hits_at_k(k):
        top_k_rel = rel_sorted[:, :k].any(axis=1)
        return float(top_k_rel[valid].mean()) if valid.any() else 0.0

    return {
        'map':    map_val,
        'top1':   top1_val,
        'hard-1': hits_at_k(1),
        'hard-2': hits_at_k(2),
        'hard-3': hits_at_k(3),
        'hard-4': hits_at_k(4),
        'soft-5': hits_at_k(5),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers — column naming is mode-dependent
# ══════════════════════════════════════════════════════════════════════════════

_CSV_METRICS = ['map', 'top1', 'hard-1', 'hard-2', 'hard-3', 'hard-4', 'soft-5']

if config.MODE == 'lines':
    _LEVEL_COL  = 'bin_center'
    _UNITS_COL  = 'mean_lines_replaced'
    _LEVEL_HEAD = 'Bin%'
    _UNITS_HEAD = 'Lines'
    _UNITS_FMT  = '5.2f'
    _UNITS_W    = 6
else:
    _LEVEL_COL  = 'nominal_pct'
    _UNITS_COL  = 'mean_patches_replaced'
    _LEVEL_HEAD = 'Nom%'
    _UNITS_HEAD = 'Patches'
    _UNITS_FMT  = '7.1f'
    _UNITS_W    = 8


# ══════════════════════════════════════════════════════════════════════════════
# Build summary, print table, write CSV + pkl
# ══════════════════════════════════════════════════════════════════════════════

def build_and_save(res_dict, label, pkl_suffix, csv_suffix,
                   pooling_configs, realised_pct_per_level, mean_units_replaced,
                   pages_per_level, n_valid_pages, extra_meta):
    """
    Parameters
    ──────────
    res_dict               : { pooling_name -> { lvl -> [metric_dict, ...] } }
    label                  : human-readable description for the table header
    pkl_suffix / csv_suffix: appended to FILE_STEM for output filenames
    pooling_configs        : list of (name, pooling_obj) — only names are used
    realised_pct_per_level : { lvl -> float }
    mean_units_replaced    : { lvl -> float }
    pages_per_level        : { lvl -> int }   (distinct n_replace values)
    n_valid_pages          : int
    extra_meta             : dict merged verbatim into the pkl payload
    """
    levels = config.PCT_LEVELS

    # ── print table ──────────────────────────────────────────────────────────
    print(f'\n\n=== RESULTS (mAP) — {label} ===')
    print(f'{_LEVEL_HEAD:>5}  {"Real%":>6}  {_UNITS_HEAD:>{_UNITS_W}}  ', end='')
    for name, _ in pooling_configs:
        print(f'{name:>26}', end='')
    print()
    print('-' * (5 + 2 + 6 + 2 + _UNITS_W + 2 + 26 * len(pooling_configs)))

    summary = {}
    for lvl in levels:
        row = {
            _LEVEL_COL:     lvl,
            'realised_pct': realised_pct_per_level[lvl],
            _UNITS_COL:     mean_units_replaced[lvl],
        }
        line = (f'{lvl:>5d}  {realised_pct_per_level[lvl]:>5.2f}%  '
                f'{mean_units_replaced[lvl]:>{_UNITS_FMT}}  ')
        for name, _ in pooling_configs:
            trial_dicts = res_dict[name][lvl]
            for metric in trial_dicts[0].keys():
                vals = [d[metric] for d in trial_dicts]
                row[f'{name}_{metric}_mean'] = float(np.mean(vals))
                row[f'{name}_{metric}_std']  = float(np.std(vals))
            map_v = [d['map'] for d in trial_dicts]
            line += f'{np.mean(map_v):.4f}±{np.std(map_v):.4f}          '
        summary[lvl] = row
        print(line)

    # ── pkl ──────────────────────────────────────────────────────────────────
    pkl_path = os.path.join(config.OUT_DIR, f'{config.FILE_STEM}{pkl_suffix}.pkl')
    payload  = {
        'results':               res_dict,
        'summary':               summary,
        'pct_levels':            levels,
        'realised_pct':          realised_pct_per_level,
        'mean_units_replaced':   mean_units_replaced,
        'pages_per_level':       pages_per_level,
        'label':                 label,
        'dataset':               config.DATASET,
        'mode':                  config.MODE,
        'n_pages':               n_valid_pages,
        'n_trials':              config.N_TRIALS,
        'foreign_pooling':       'single_page_large_enough',
        'foreign_src_excluded':  True,
        **extra_meta,
    }
    pickle.dump(payload, open(pkl_path, 'wb'))

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(config.OUT_DIR, f'{config.FILE_STEM}{csv_suffix}.csv')
    with open(csv_path, 'w') as f:
        header = [_LEVEL_COL, 'realised_pct', _UNITS_COL]
        for name, _ in pooling_configs:
            for m in _CSV_METRICS:
                header += [f'{name}_{m}_mean', f'{name}_{m}_std']
        f.write(','.join(header) + '\n')
        for lvl in levels:
            r    = summary[lvl]
            vals = [str(r[_LEVEL_COL]),
                    f"{r['realised_pct']:.4f}",
                    f"{r[_UNITS_COL]:.4f}"]
            for name, _ in pooling_configs:
                for m in _CSV_METRICS:
                    vals += [f"{r.get(f'{name}_{m}_mean', float('nan')):.6f}",
                             f"{r.get(f'{name}_{m}_std',  float('nan')):.6f}"]
            f.write(','.join(vals) + '\n')

    print(f'\nSaved: {csv_path}')
    print(f'Saved: {pkl_path}')
    return summary