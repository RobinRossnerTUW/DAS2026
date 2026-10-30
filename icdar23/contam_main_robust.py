"""
contam_main_robust.py
──────────────────────
Extends contam_main.py with alternative "final evaluations" of the same
contaminated patch data, all aimed at reducing the influence contaminated
(foreign-writer) patches have on retrieval:

    1. SumPooling-PN0p4         (baseline, unchanged — kept for comparison)
    2. GeometricMedian-PN0p4    (robust patch aggregation)
    3. TrimmedMean20-PN0p4      (robust patch aggregation)
    4. KMeansReject-PN0p4       (robust patch aggregation)
    5. SumPooling-PN0p4+SGR     (baseline descriptors, SGR graph reranking
                                 applied to the retrieval distance matrix)

This is a separate script rather than an edit to contam_main.py, so your
existing working pipeline is untouched — run whichever one you need.

Efficiency
──────────
The expensive part — reading + L2-normalising raw patch rows from the HDF5
file and building the "mixed" (own + foreign) patch matrix for a given
contamination level — happens exactly once per (page, distinct n_replace,
trial), inside workers.py, UNCHANGED. All four pooling methods above are
computed from that same per-page patch matrix inside the same worker call
(workers._encode iterates every entry in workers.POOLING_CONFIGS), so adding
more pooling-based "final evaluations" costs only the aggregation step
itself — no extra pass over the HDF5 file, no extra selection/sampling work.

SGR reranking is the odd one out — it isn't a patch-aggregation method, it
reranks the already-pooled, PCA-transformed, L2-normalised descriptor matrix
for the whole gallery. That happens once per (level, trial), reusing the
exact `pfs_tf` the baseline SumPooling distance matrix is already built
from — so it's one extra matmul-heavy call, not a repeat of the encode loop.

Every one of the 5 methods gets both the standard mAP (original writer
labels) and the foreign-writer mAP evaluation (if EVAL_FOREIGN_MAP), and
all 5 are written to ONE CSV/pkl pair per eval type (suffix _robust /
_robust_foreign), exactly analogous to contam_main.py's own/foreign split.

Usage
─────
    python contam_main_robust.py
    python contam_main_robust.py --dataset icdar --mode random
    python contam_main_robust.py --dataset cvl   --mode lines --no-wandb
"""

import sys, os, pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
import multiprocessing as mp

# Local modules — unchanged, shared with contam_main.py
import config
import page_xml
import workers
import selection
import results

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from evaluators.retrieval import Retrieval
from page_encodings import SumPooling
from robust_pooling import GeometricMedianPooling, TrimmedMeanPooling, KMeansRejectionPooling

# ── SGR import: real project package first, standalone file as fallback ──────
# Adjust this to wherever your `reranking` package (gnn/krnn/sgr) actually
# lives — this default matches the path already used in rerank.py.
RERANK_REPO_PATH = "/data/rrossner/DAS2026/icdar23"
if RERANK_REPO_PATH not in sys.path:
    sys.path.insert(0, RERANK_REPO_PATH)
try:
    from reranking.sgr import sgr_reranking
except ImportError:
    from sgr import sgr_reranking          # standalone copy alongside this script


# ══════════════════════════════════════════════════════════════════════════════
# Robust-eval specific config
# (kept local so contam_main.py / config.py stay untouched — tune here)
# ══════════════════════════════════════════════════════════════════════════════
PN_ALPHA    = 0.4     # power-norm alpha, matched to the existing SumPooling baseline
TRIM_FRAC   = 0.20    # TrimmedMeanPooling: fraction of farthest patches dropped
KMEANS_K    = 2       # KMeansRejectionPooling: number of clusters

SGR_K       = 2
SGR_LAYERS  = 2
SGR_GAMMA   = 0.15
RERANK_BASE = 'SumPooling-PN0p4'         # which descriptor SGR is applied on top of
RERANK_NAME = 'SumPooling-PN0p4+SGR'


def apply_sgr(pfs_tf, k=SGR_K, layers=SGR_LAYERS, gamma=SGR_GAMMA):
    """
    pfs_tf : (N, D) L2-normalised page descriptors.
    Returns an (N, N) numpy distance matrix after SGR graph reranking.
    Diagonal is zeroed to match the convention already used for SGR outputs
    in rerank.py's own evaluate(), rather than left at SGR's raw self-distance.
    """
    X = torch.tensor(pfs_tf, dtype=torch.float32)
    _, dist = sgr_reranking(X, k, layer=layers, gamma=gamma)
    dist = dist.numpy()
    np.fill_diagonal(dist, 0.0)
    return dist


# ══════════════════════════════════════════════════════════════════════════════
# wandb (mirrors contam_main.py, with the extra method list recorded)
# ══════════════════════════════════════════════════════════════════════════════
if config.USE_WANDB:
    import wandb
    _wb_cfg = {
        'dataset': config.DATASET, 'mode': config.MODE,
        'experiment_dir': config.EXPERIMENT_DIR, 'pct_levels': config.PCT_LEVELS,
        'n_trials': config.N_TRIALS, 'n_workers': config.N_WORKERS, 'seed': config.SEED,
        'eval_foreign_map': config.EVAL_FOREIGN_MAP,
        'foreign_pooling': 'single_page_large_enough', 'foreign_src_excluded': True,
        'robust_methods': ['SumPooling-PN0p4', 'GeometricMedian-PN0p4',
                          'TrimmedMean20-PN0p4', 'KMeansReject-PN0p4', RERANK_NAME],
        'pn_alpha': PN_ALPHA, 'trim_frac': TRIM_FRAC, 'kmeans_k': KMEANS_K,
        'sgr_k': SGR_K, 'sgr_layers': SGR_LAYERS, 'sgr_gamma': SGR_GAMMA,
    }
    if config.MODE == 'lines':
        _wb_cfg.update(min_lines=config.MIN_LINES, bin_centers=config.BIN_CENTERS)
    else:
        _wb_cfg['replacement'] = 'random_patches'
    wandb.init(project=config.WANDB_PROJECT,
              name=(config.WANDB_RUN or f'{config.DATASET}-{config.MODE}-robust'),
              config=_wb_cfg)
else:
    wandb = None


# ══════════════════════════════════════════════════════════════════════════════
# Load artifacts (identical to contam_main.py)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\nDataset : {config.DATASET}   Mode: {config.MODE}   [robust-eval]')
print(f'Experiment dir : {config.EXPERIMENT_DIR}')
print(f'Out dir        : {config.OUT_DIR}\n')

print('Loading artifacts...')
with open(config.INDEX_PATH, 'rb') as f:
    index = pickle.load(f)
pca = pickle.load(open(config.PCA_PATH, 'rb'))

wp_to_key        = {(e['writer'], e['page']): e['hdf5_key'] for e in index}
page_num_patches = {(e['writer'], e['page']): e['patch_end'] - e['patch_start']
                    for e in index}
writer_to_pages  = defaultdict(list)
for e in index:
    writer_to_pages[e['writer']].append((e['writer'], e['page']))

all_writers = list(writer_to_pages.keys())
all_wps     = list(wp_to_key.keys())
print(f'Total pages: {len(all_wps)}   Writers: {len(all_writers)}')


# ══════════════════════════════════════════════════════════════════════════════
# Build page metadata and filter to evaluable pages (identical to contam_main.py)
# ══════════════════════════════════════════════════════════════════════════════
if config.MODE == 'lines':
    print('\nParsing PAGE XML and assigning patches to textlines...')
    page_line_map, page_num_lines, xml_stats = page_xml.build_all_line_maps(all_wps)
    print(f"\nXML found: {xml_stats['xml_found']}  "
          f"XML missing (Y-cluster fallback): {xml_stats['xml_missing']}  "
          f"coords missing: {xml_stats['coords_missing']}")

    valid_wps = [wp for wp in all_wps if page_num_lines[wp] >= config.MIN_LINES]
    dropped   = len(all_wps) - len(valid_wps)
    all_n     = [page_num_lines[wp] for wp in valid_wps]
    print(f'\nGallery filter: keep {len(valid_wps)} / {len(all_wps)} pages '
          f'(dropped {dropped} with < {config.MIN_LINES} lines)')
    if not all_n:
        print('FATAL: no valid pages.'); sys.exit(1)

    extra_meta_lines = dict(
        bin_centers=config.BIN_CENTERS, mean_lines_replaced={}, min_lines=config.MIN_LINES,
        n_valid_pages=len(valid_wps), n_dropped=dropped,
        xml_found=xml_stats['xml_found'], xml_missing=xml_stats['xml_missing'],
        coords_missing=xml_stats['coords_missing'], json_missing=xml_stats['coords_missing'],
        lines_per_page_stats=dict(
            mean=float(np.mean(all_n)), median=float(np.median(all_n)),
            std=float(np.std(all_n)), min=int(min(all_n)), max=int(max(all_n))),
    )
else:
    page_line_map  = None
    page_num_lines = None
    dropped        = 0
    valid_wps      = all_wps
    n_patches_arr  = np.array([page_num_patches[wp] for wp in all_wps])
    all_n          = list(n_patches_arr)
    print(f'Patches per page: mean={n_patches_arr.mean():.1f}  '
          f'median={np.median(n_patches_arr):.1f}  std={n_patches_arr.std():.1f}')
    extra_meta_random = dict(
        mean_patches_replaced={}, replacement='random_patches',
        patches_per_page_stats=dict(
            mean=float(n_patches_arr.mean()), median=float(np.median(n_patches_arr)),
            std=float(n_patches_arr.std()), min=int(n_patches_arr.min()),
            max=int(n_patches_arr.max())),
    )

valid_writer_to_pages = defaultdict(list)
for wp in valid_wps:
    valid_writer_to_pages[wp[0]].append(wp)
valid_writers = list(valid_writer_to_pages.keys())

realised_pct, mean_units, pages_per = selection.compute_realised_pct(
    valid_wps, page_num_lines=page_num_lines, page_num_patches=page_num_patches)

if config.MODE == 'lines':
    extra_meta_lines['mean_lines_replaced'] = mean_units
else:
    extra_meta_random['mean_patches_replaced'] = mean_units


# ══════════════════════════════════════════════════════════════════════════════
# Pooling configurations — baseline + 3 robust aggregation methods
# ══════════════════════════════════════════════════════════════════════════════
pooling_configs = [
    ('SumPooling-PN0p4',      SumPooling('l2', pn_alpha=PN_ALPHA)),
    ('GeometricMedian-PN0p4', GeometricMedianPooling(pn_alpha=PN_ALPHA)),
    ('TrimmedMean20-PN0p4',   TrimmedMeanPooling(trim_frac=TRIM_FRAC, pn_alpha=PN_ALPHA)),
    ('KMeansReject-PN0p4',    KMeansRejectionPooling(k=KMEANS_K, pn_alpha=PN_ALPHA)),
]
# SGR isn't a pooling method — it reranks the baseline's distance matrix, so it
# gets a result slot below but no entry here (workers.py never needs to see it).
pooling_configs_out = pooling_configs + [(RERANK_NAME, None)]   # names only, for build_and_save

workers.WP_TO_KEY        = wp_to_key
workers.PAGE_LINE_MAP    = page_line_map
workers.PAGE_NUM_PATCHES = page_num_patches
workers.POOLING_CONFIGS  = pooling_configs


# ══════════════════════════════════════════════════════════════════════════════
# Trial loop
# ══════════════════════════════════════════════════════════════════════════════
method_names = [name for name, _ in pooling_configs] + [RERANK_NAME]
res         = {name: {lvl: [] for lvl in config.PCT_LEVELS} for name in method_names}
res_foreign = {name: {lvl: [] for lvl in config.PCT_LEVELS} for name in method_names}

page_writers = np.array([wp[0] for wp in valid_wps])
wp_to_idx    = {wp: i for i, wp in enumerate(valid_wps)}

print(f'\nStarting Pool with {config.N_WORKERS} workers...')
print(f'Methods this run: {method_names}')
if config.EVAL_FOREIGN_MAP:
    print('EVAL_FOREIGN_MAP=True  (source page excluded from relevant set).\n')

mp_ctx = mp.get_context('fork')

with mp_ctx.Pool(processes=config.N_WORKERS,
                 initializer=workers.worker_init,
                 initargs=(config.H5_PATH,)) as pool:

    for trial in range(config.N_TRIALS):
        print(f'=== Trial {trial + 1}/{config.N_TRIALS} ===')

        foreign_map, sels = selection.precompute(
            config.SEED + trial * 1009,
            valid_wps, valid_writers, valid_writer_to_pages,
            page_num_lines=page_num_lines, page_line_map=page_line_map,
            page_num_patches=page_num_patches,
        )

        if config.EVAL_FOREIGN_MAP:
            foreign_writers = np.array([foreign_map[wp][0] for wp in valid_wps])
            excluded_per_query = [
                {wp_to_idx[foreign_map[wp]]} if foreign_map[wp] in wp_to_idx else set()
                for wp in valid_wps
            ]

        jobs      = [(wp, foreign_map[wp], sels[wp]) for wp in valid_wps]
        chunksize = max(1, len(jobs) // (4 * config.N_WORKERS))

        # Only the 4 real pooling methods need a slot here — SGR is derived below.
        page_descs = {lvl: {name: [None] * len(valid_wps) for name, _ in pooling_configs}
                      for lvl in config.PCT_LEVELS}

        for wp, encs in tqdm(
                pool.imap_unordered(workers.encode_page_job, jobs, chunksize=chunksize),
                total=len(jobs), desc=f'  Trial {trial + 1}: pages', mininterval=2):
            i = wp_to_idx[wp]
            for lvl in config.PCT_LEVELS:
                for name, _ in pooling_configs:
                    page_descs[lvl][name][i] = encs[lvl][name]

        for lvl in tqdm(config.PCT_LEVELS, desc=f'  Trial {trial + 1}: eval'):
            baseline_pfs_tf = None      # reused below for the SGR pass — no recompute

            for name, _ in pooling_configs:
                pfs    = np.concatenate(page_descs[lvl][name], axis=0)
                pfs_tf = normalize(pca.transform(pfs), axis=1)
                dist   = 1 - pfs_tf @ pfs_tf.T

                _, csv_res = Retrieval.calc_map_from_distances(page_writers, dist)
                res[name][lvl].append(csv_res)

                if config.EVAL_FOREIGN_MAP:
                    res_foreign[name][lvl].append(
                        results.calc_foreign_map(page_writers, foreign_writers, dist,
                                                 excluded_per_query=excluded_per_query))

                if name == RERANK_BASE:
                    baseline_pfs_tf = pfs_tf

            # ── 5th "method": SGR reranking on top of the baseline descriptors ──
            dist_sgr = apply_sgr(baseline_pfs_tf)
            _, csv_sgr = Retrieval.calc_map_from_distances(page_writers, dist_sgr)
            res[RERANK_NAME][lvl].append(csv_sgr)
            if config.EVAL_FOREIGN_MAP:
                res_foreign[RERANK_NAME][lvl].append(
                    results.calc_foreign_map(page_writers, foreign_writers, dist_sgr,
                                             excluded_per_query=excluded_per_query))

            if wandb is not None:
                log = ({'trial': trial, 'bin_center': lvl, 'realised_pct': realised_pct[lvl],
                        'mean_lines_replaced': mean_units[lvl]} if config.MODE == 'lines' else
                       {'trial': trial, 'nominal_pct': lvl, 'realised_pct': realised_pct[lvl],
                        'mean_patches_replaced': mean_units[lvl]})
                for name in method_names:
                    for metric, val in res[name][lvl][-1].items():
                        log[f'{name}/{metric}'] = val
                    if config.EVAL_FOREIGN_MAP:
                        for metric, val in res_foreign[name][lvl][-1].items():
                            log[f'foreign/{name}/{metric}'] = val
                wandb.log(log)


# ══════════════════════════════════════════════════════════════════════════════
# Save results — one CSV/pkl pair per (original | foreign), covering all 5 methods
# ══════════════════════════════════════════════════════════════════════════════
extra_meta = extra_meta_lines if config.MODE == 'lines' else extra_meta_random

results.build_and_save(
    res, 'Original writer labels (robust methods)', '_robust', '_robust',
    pooling_configs_out, realised_pct, mean_units, pages_per, len(valid_wps), extra_meta,
)

if config.EVAL_FOREIGN_MAP:
    results.build_and_save(
        res_foreign, 'Foreign writer labels (robust methods, source page excluded)',
        '_robust_foreign', '_robust_foreign',
        pooling_configs_out, realised_pct, mean_units, pages_per, len(valid_wps), extra_meta,
    )

if wandb is not None:
    wandb.finish()

print('\nDone. Results saved with suffixes _robust / _robust_foreign in:')
print(f'  {config.OUT_DIR}')
