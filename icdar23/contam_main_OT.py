"""
main.py
───────
Entry point. Loads artifacts, builds page metadata, runs the multiprocessing
contamination trial loop, and saves results.

Usage
─────
    python main.py
    python main.py --dataset icdar --mode random
    python main.py --dataset cvl   --mode lines --no-wandb

New in this version (vs. the trimming-only version)
─────────────────────────────────────────────────────
- Partial optimal-transport distance (ot_distance.py) is evaluated as extra
  rows alongside Trim-eps*, gated the same way trimming is (config.USE_OT,
  needs pca_patch.pkl — reuses the SAME page_patches[lvl] local descriptors
  trimming already uses, no separate patch prep).
- Results are checkpointed to disk every config.CHECKPOINT_EVERY_HOURS,
  not just once at the very end — same build_and_save call, just also
  invoked mid-run so you have a continuously-updating "current best guess"
  file on disk independent of wandb. See config.py for the new knobs.
"""

import sys, pickle, time
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import multiprocessing as mp

# Local modules
import config
import page_xml
import workers
import selection
import results
import trimming
import ot_distance
import entropic_ot_distance

# Pooling implementations (from the wider project)
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from evaluators.retrieval import Retrieval
from page_encodings import SumPooling, MaxPooling, LSEPooling


# ══════════════════════════════════════════════════════════════════════════════
# wandb
# ══════════════════════════════════════════════════════════════════════════════
if config.USE_WANDB:
    import wandb
    _wb_cfg = {
        'dataset':              config.DATASET,
        'mode':                 config.MODE,
        'experiment_dir':       config.EXPERIMENT_DIR,
        'pct_levels':           config.PCT_LEVELS,
        'n_trials':             config.N_TRIALS,
        'n_workers':            config.N_WORKERS,
        'seed':                 config.SEED,
        'eval_foreign_map':     config.EVAL_FOREIGN_MAP,
        'foreign_pooling':      'single_page_large_enough',
        'foreign_src_excluded': True,
    }
    if config.MODE == 'lines':
        _wb_cfg.update(min_lines=config.MIN_LINES, bin_centers=config.BIN_CENTERS)
    else:
        _wb_cfg['replacement'] = 'random_patches'
    wandb.init(project=config.WANDB_PROJECT, name=config.WANDB_RUN, config=_wb_cfg)
else:
    wandb = None


# ══════════════════════════════════════════════════════════════════════════════
# Load artifacts
# ══════════════════════════════════════════════════════════════════════════════
print(f'\nDataset : {config.DATASET}   Mode: {config.MODE}')
print(f'Experiment dir : {config.EXPERIMENT_DIR}')
print(f'Out dir        : {config.OUT_DIR}\n')

print('Loading artifacts...')
with open(config.INDEX_PATH, 'rb') as f:
    index = pickle.load(f)
pca = pickle.load(open(config.PCA_PATH, 'rb'))

pca_patch = None
if config.USE_TRIMMING or getattr(config, 'USE_OT', False) or getattr(config, 'USE_ENTROPIC_OT', False) \
        or getattr(config, 'USE_TOPK_OT', False):
    try:
        pca_patch = pickle.load(open(config.PCA_PATCH_PATH, 'rb'))
        print(f'Loaded patch-level PCA from {config.PCA_PATCH_PATH}')
    except FileNotFoundError:
        print(f'WARNING: trimming/OT requested but {config.PCA_PATCH_PATH} not found. '
              f'Run the patch-level PCA fitting step in main.py::test() first. '
              f'Disabling trimming and OT for this run.')

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
# Build page metadata and filter to evaluable pages
# ══════════════════════════════════════════════════════════════════════════════
if config.MODE == 'lines':
    print('\nParsing PAGE XML and assigning patches to textlines...')
    page_line_map, page_num_lines, xml_stats = page_xml.build_all_line_maps(all_wps)
    print(f"\nXML found: {xml_stats['xml_found']}  "
          f"XML missing (Y-cluster fallback): {xml_stats['xml_missing']}  "
          f"coords missing: {xml_stats['coords_missing']}")
    print(f"Patches assigned to nearest line (outside any region): {xml_stats['total_outside']}")

    valid_wps = [wp for wp in all_wps if page_num_lines[wp] >= config.MIN_LINES]
    dropped   = len(all_wps) - len(valid_wps)
    all_n     = [page_num_lines[wp] for wp in valid_wps]
    print(f'\nGallery filter: keep {len(valid_wps)} / {len(all_wps)} pages '
          f'(dropped {dropped} with < {config.MIN_LINES} lines)')
    if not all_n:
        print('FATAL: no valid pages.'); sys.exit(1)
    print(f'Lines per page: mean={np.mean(all_n):.2f}  median={np.median(all_n):.1f}  '
          f'std={np.std(all_n):.2f}  min={min(all_n)}  max={max(all_n)}')

    n_patches_arr = None
    extra_meta_lines = dict(
        bin_centers          = config.BIN_CENTERS,
        mean_lines_replaced  = {},       # filled below
        min_lines            = config.MIN_LINES,
        n_valid_pages        = len(valid_wps),
        n_dropped            = dropped,
        xml_found            = xml_stats['xml_found'],
        xml_missing          = xml_stats['xml_missing'],
        coords_missing       = xml_stats['coords_missing'],
        json_missing         = xml_stats['coords_missing'],
        lines_per_page_stats = dict(
            mean=float(np.mean(all_n)), median=float(np.median(all_n)),
            std=float(np.std(all_n)),   min=int(min(all_n)), max=int(max(all_n))),
    )

else:
    page_line_map  = None
    page_num_lines = None
    xml_stats      = dict(xml_found=0, xml_missing=0, coords_missing=0, total_outside=0)
    dropped        = 0
    valid_wps      = all_wps
    n_patches_arr  = np.array([page_num_patches[wp] for wp in all_wps])
    all_n          = list(n_patches_arr)
    print(f'Patches per page: mean={n_patches_arr.mean():.1f}  '
          f'median={np.median(n_patches_arr):.1f}  std={n_patches_arr.std():.1f}  '
          f'min={n_patches_arr.min()}  max={n_patches_arr.max()}')
    extra_meta_random = dict(
        mean_patches_replaced  = {},    # filled below
        replacement            = 'random_patches',
        patches_per_page_stats = dict(
            mean=float(n_patches_arr.mean()), median=float(np.median(n_patches_arr)),
            std=float(n_patches_arr.std()),   min=int(n_patches_arr.min()),
            max=int(n_patches_arr.max())),
    )

valid_writer_to_pages = defaultdict(list)
for wp in valid_wps:
    valid_writer_to_pages[wp[0]].append(wp)
valid_writers = list(valid_writer_to_pages.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Realised contamination summary
# ══════════════════════════════════════════════════════════════════════════════
realised_pct, mean_units, pages_per = selection.compute_realised_pct(
    valid_wps, page_num_lines=page_num_lines, page_num_patches=page_num_patches)

if config.MODE == 'lines':
    extra_meta_lines['mean_lines_replaced'] = mean_units
    print(f'\nBin centre  ->  mean realised %  (mean lines replaced):')
    for bc in config.BIN_CENTERS:
        print(f'  {bc:>3d}%  ->  {realised_pct[bc]:5.2f}%  ({mean_units[bc]:.2f} lines)'
              f'  [{pages_per[bc]} distinct n values]')
else:
    extra_meta_random['mean_patches_replaced'] = mean_units
    print(f'\nNominal pct  ->  realised %  (mean patches replaced):')
    for pct in config.PCT_LEVELS:
        print(f'  {pct:>3d}%  ->  {realised_pct[pct]:5.2f}%  ({mean_units[pct]:.1f} patches)')


# ══════════════════════════════════════════════════════════════════════════════
# Pooling configurations
# ══════════════════════════════════════════════════════════════════════════════
pooling_configs = [
    ('SumPooling-PN0p4',  SumPooling('l2', pn_alpha=0.4)),
    #('SumPooling-PNsqrt', SumPooling('powernorm')),
    #('MaxPooling',        MaxPooling('l2')),
    #('LSEPooling',        LSEPooling('l2')),
]

# Trimmed distance and partial OT are evaluated as extra rows alongside the
# poolings above. Neither is a page-pooling function, so neither has an
# entry in pooling_configs / workers.POOLING_CONFIGS — instead workers
# additionally hand back a raw, PCA-reduced patch array per page/level (see
# workers._prep_patch_repr), shared by BOTH methods, and the (N, N) distance
# matrices are computed once per level below.
use_trimming = config.USE_TRIMMING and pca_patch is not None
use_ot       = getattr(config, 'USE_OT', False) and pca_patch is not None
use_entropic_ot = getattr(config, 'USE_ENTROPIC_OT', False) and pca_patch is not None
use_topk_ot  = getattr(config, 'USE_TOPK_OT', False) and pca_patch is not None
need_patch_repr = use_trimming or use_ot or use_entropic_ot or use_topk_ot

trim_names  = [f'Trim-eps{eps:g}' for eps in config.TRIM_EPS] if use_trimming else []
OT_M_LEVELS = getattr(config, 'OT_M_LEVELS', [0.5, 0.7, 0.9])
OT_MEM_LIMIT_GB = getattr(config, 'OT_MEM_LIMIT_GB', None)
OT_GPUID    = getattr(config, 'OT_GPUID', None)
ot_names    = [f'OT-m{m:g}' for m in OT_M_LEVELS] if use_ot else []

TOPK_OT_K        = getattr(config, 'TOPK_OT_K', 5)
TOPK_OT_M_LEVELS = getattr(config, 'TOPK_OT_M_LEVELS', [0.5, 0.7, 0.9])
topk_ot_names    = [f'SumPooling+Top{TOPK_OT_K}_OT-m{m:g}' for m in TOPK_OT_M_LEVELS] if use_topk_ot else []

ENTROPIC_OT_M_LEVELS   = getattr(config, 'ENTROPIC_OT_M_LEVELS', [0.5, 0.7, 0.9])
ENTROPIC_OT_REG        = getattr(config, 'ENTROPIC_OT_REG', 0.03)
ENTROPIC_OT_DEVICE     = getattr(config, 'ENTROPIC_OT_DEVICE', 'cuda')
ENTROPIC_OT_PAIR_CHUNK = getattr(config, 'ENTROPIC_OT_PAIR_CHUNK', 128)
ENTROPIC_OT_NUMITERMAX = getattr(config, 'ENTROPIC_OT_NUMITERMAX', 1000)
ENTROPIC_OT_STOPTHR    = getattr(config, 'ENTROPIC_OT_STOPTHR', 1e-6)
entropic_names = [f'EntropicOT-m{m:g}' for m in ENTROPIC_OT_M_LEVELS] if use_entropic_ot else []

# Expose globals to worker processes (inherited via fork — must be set before Pool)
workers.WP_TO_KEY        = wp_to_key
workers.PAGE_LINE_MAP    = page_line_map
workers.PAGE_NUM_PATCHES = page_num_patches
workers.POOLING_CONFIGS  = pooling_configs
workers.TRIM_PCA         = pca_patch if need_patch_repr else None


# ══════════════════════════════════════════════════════════════════════════════
# Trial loop
# ══════════════════════════════════════════════════════════════════════════════
all_names   = [name for name, _ in pooling_configs] + trim_names + ot_names + entropic_names + topk_ot_names
res         = {name: {lvl: [] for lvl in config.PCT_LEVELS} for name in all_names}
res_foreign = {name: {lvl: [] for lvl in config.PCT_LEVELS} for name in all_names}

page_writers = np.array([wp[0] for wp in valid_wps])
wp_to_idx    = {wp: i for i, wp in enumerate(valid_wps)}

# ── report_configs / extra_meta: moved up from "Save results" (unchanged
#    otherwise) so the periodic checkpoint below can reuse them ─────────────
extra_meta = extra_meta_lines if config.MODE == 'lines' else extra_meta_random
extra_meta['trim_eps'] = config.TRIM_EPS if use_trimming else []
extra_meta['ot_m_levels'] = OT_M_LEVELS if use_ot else []
extra_meta['entropic_ot_m_levels'] = ENTROPIC_OT_M_LEVELS if use_entropic_ot else []
extra_meta['entropic_ot_reg'] = ENTROPIC_OT_REG if use_entropic_ot else None
extra_meta['topk_ot_k'] = TOPK_OT_K if use_topk_ot else None
extra_meta['topk_ot_m_levels'] = TOPK_OT_M_LEVELS if use_topk_ot else []
report_configs = pooling_configs + [(name, None) for name in trim_names + ot_names + entropic_names + topk_ot_names]

unit_word = 'bins' if config.MODE == 'lines' else 'levels'
print(f'\nStarting Pool with {config.N_WORKERS} workers...')
print(f'Running {len(config.PCT_LEVELS)} {unit_word} × {config.N_TRIALS} trials × '
      f'{len(pooling_configs)} poolings on {len(valid_wps)} pages...')
if use_trimming:
    print(f'Trimming enabled: {trim_names}  (device={config.TRIM_DEVICE})')
elif config.USE_TRIMMING:
    print('Trimming requested but disabled (missing pca_patch.pkl — see warning above).')
if use_ot:
    print(f'Partial OT enabled: {ot_names}  '
          f'(n_jobs={config.N_WORKERS}, mem_limit_gb={OT_MEM_LIMIT_GB}, gpuid={OT_GPUID})')
elif getattr(config, 'USE_OT', False):
    print('Partial OT requested but disabled (missing pca_patch.pkl — see warning above).')
if use_entropic_ot:
    print(f'Entropic (Sinkhorn) OT enabled: {entropic_names}  '
          f'(reg={ENTROPIC_OT_REG}, device={ENTROPIC_OT_DEVICE}, '
          f'pair_chunk={ENTROPIC_OT_PAIR_CHUNK})')
elif getattr(config, 'USE_ENTROPIC_OT', False):
    print('Entropic OT requested but disabled (missing pca_patch.pkl — see warning above).')
if use_topk_ot:
    print(f'SumPooling+Top{TOPK_OT_K}-OT enabled: {topk_ot_names}  '
          f'(k={TOPK_OT_K}, n_jobs={config.N_WORKERS}, mem_limit_gb={OT_MEM_LIMIT_GB}, gpuid={OT_GPUID})')
elif getattr(config, 'USE_TOPK_OT', False):
    k_disabled = getattr(config, 'TOPK_OT_K', 5)
    print(f'SumPooling+Top{k_disabled}-OT requested but disabled (missing pca_patch.pkl — see warning above).')
if config.EVAL_FOREIGN_MAP:
    print('EVAL_FOREIGN_MAP=True  (source page excluded from relevant set).\n')

# ── checkpoint clock ────────────────────────────────────────────────────────
checkpoint_every_sec = getattr(config, 'CHECKPOINT_EVERY_HOURS', 12) * 3600
last_checkpoint = time.time()

mp_ctx = mp.get_context('fork')

with mp_ctx.Pool(processes=config.N_WORKERS,
                 initializer=workers.worker_init,
                 initargs=(config.H5_PATH,)) as pool:

    for trial in range(config.N_TRIALS):
        print(f'=== Trial {trial + 1}/{config.N_TRIALS} ===')

        foreign_map, sels = selection.precompute(
            config.SEED + trial * 1009,
            valid_wps, valid_writers, valid_writer_to_pages,
            page_num_lines=page_num_lines,
            page_line_map=page_line_map,
            page_num_patches=page_num_patches,
        )

        if config.EVAL_FOREIGN_MAP:
            foreign_writers = np.array([foreign_map[wp][0] for wp in valid_wps])
            # Exactly one source page excluded per query (level-independent).
            excluded_per_query = [
                {wp_to_idx[foreign_map[wp]]} if foreign_map[wp] in wp_to_idx else set()
                for wp in valid_wps
            ]

        jobs      = [(wp, foreign_map[wp], sels[wp]) for wp in valid_wps]
        chunksize = max(1, len(jobs) // (4 * config.N_WORKERS))

        page_descs = {lvl: {name: [None] * len(valid_wps)
                            for name, _ in pooling_configs}
                      for lvl in config.PCT_LEVELS}
        page_patches = {lvl: [None] * len(valid_wps) for lvl in config.PCT_LEVELS} \
                       if need_patch_repr else None

        for wp, encs in tqdm(
                pool.imap_unordered(workers.encode_page_job, jobs, chunksize=chunksize),
                total=len(jobs), desc=f'  Trial {trial + 1}: pages', mininterval=2):
            i = wp_to_idx[wp]
            for lvl in config.PCT_LEVELS:
                pooled_lvl, patch_repr = encs[lvl]
                for name, _ in pooling_configs:
                    page_descs[lvl][name][i] = pooled_lvl[name]
                if need_patch_repr:
                    page_patches[lvl][i] = patch_repr

        for lvl in tqdm(config.PCT_LEVELS, desc=f'  Trial {trial + 1}: eval'):
            sumpool_dist = None
            for name, _ in pooling_configs:
                pfs    = np.concatenate(page_descs[lvl][name], axis=0)
                pfs_tf = normalize(pca.transform(pfs), axis=1)
                dist   = 1 - pfs_tf @ pfs_tf.T

                _, csv_res = Retrieval.calc_map_from_distances(page_writers, dist)
                res[name][lvl].append(csv_res)

                if config.EVAL_FOREIGN_MAP:
                    res_foreign[name][lvl].append(
                        results.calc_foreign_map(
                            page_writers, foreign_writers, dist,
                            excluded_per_query=excluded_per_query))

                if name == 'SumPooling-PN0p4':
                    sumpool_dist = dist   # base ranking that TopK-OT reranks

            if use_trimming:
                # One pass over all eps values: sorting is shared, only the
                # trim fraction differs (see trimming.trimmed_distance_matrices).
                trim_dists = trimming.trimmed_distance_matrices(
                    page_patches[lvl], config.TRIM_EPS,
                    device=config.TRIM_DEVICE, j_chunk=config.TRIM_J_CHUNK)

                for eps, trim_name in zip(config.TRIM_EPS, trim_names):
                    dist = trim_dists[eps]

                    _, csv_res = Retrieval.calc_map_from_distances(page_writers, dist)
                    res[trim_name][lvl].append(csv_res)

                    if config.EVAL_FOREIGN_MAP:
                        res_foreign[trim_name][lvl].append(
                            results.calc_foreign_map(
                                page_writers, foreign_writers, dist,
                                excluded_per_query=excluded_per_query))

            if use_ot:
                # Exact partial-OT solve, CPU-parallel across page pairs
                # (no GPU-native exact solver exists — see ot_distance.py).
                # mem_limit_gb / gpuid are safety-net / environment-hygiene
                # knobs only; see ot_distance.partial_ot_distance_matrices.
                ot_dists = ot_distance.partial_ot_distance_matrices(
                    page_patches[lvl], OT_M_LEVELS,
                    n_jobs=config.N_WORKERS,
                    mem_limit_gb=OT_MEM_LIMIT_GB, gpuid=OT_GPUID)

                for m, ot_name in zip(OT_M_LEVELS, ot_names):
                    dist = ot_dists[m]

                    _, csv_res = Retrieval.calc_map_from_distances(page_writers, dist)
                    res[ot_name][lvl].append(csv_res)

                    if config.EVAL_FOREIGN_MAP:
                        res_foreign[ot_name][lvl].append(
                            results.calc_foreign_map(
                                page_writers, foreign_writers, dist,
                                excluded_per_query=excluded_per_query))

            if use_topk_ot:
                # Two-stage retrieval: SumPooling picks each query's top-k
                # candidates (sumpool_dist, computed above), exact partial OT
                # only reranks those — see
                # ot_distance.partial_ot_distance_matrices_topk.
                topk_dists = ot_distance.partial_ot_distance_matrices_topk(
                    page_patches[lvl], sumpool_dist, TOPK_OT_K, TOPK_OT_M_LEVELS,
                    n_jobs=config.N_WORKERS,
                    mem_limit_gb=OT_MEM_LIMIT_GB, gpuid=OT_GPUID)

                for m, topk_name in zip(TOPK_OT_M_LEVELS, topk_ot_names):
                    dist = topk_dists[m]

                    _, csv_res = Retrieval.calc_map_from_distances(page_writers, dist)
                    res[topk_name][lvl].append(csv_res)

                    if config.EVAL_FOREIGN_MAP:
                        res_foreign[topk_name][lvl].append(
                            results.calc_foreign_map(
                                page_writers, foreign_writers, dist,
                                excluded_per_query=excluded_per_query))

            if use_entropic_ot:
                # Matrix-op-only alternative to the exact solver above — see
                # entropic_ot_distance.py for the correctness validation and
                # the reg trade-off (too small -> unstable/NaN pairs, too
                # large -> blurred/less discriminative transport plan).
                ent_mats, unstable = entropic_ot_distance.entropic_ot_distance_matrices(
                    page_patches[lvl], ENTROPIC_OT_M_LEVELS,
                    reg=ENTROPIC_OT_REG, device=ENTROPIC_OT_DEVICE,
                    numItermax=ENTROPIC_OT_NUMITERMAX, stopThr=ENTROPIC_OT_STOPTHR,
                    pair_chunk=ENTROPIC_OT_PAIR_CHUNK)

                if unstable:
                    print(f'  [entropic OT] trial={trial + 1} lvl={lvl}: '
                          f'{len(unstable)} (pair, m) went unstable at reg={ENTROPIC_OT_REG} '
                          f'-> scored as inf (not dropped, not silently 0). If this count is '
                          f'large relative to total pairs, consider raising ENTROPIC_OT_REG.')
                    for ui, uj, um in unstable:
                        ent_mats[um][ui, uj] = np.inf
                        ent_mats[um][uj, ui] = np.inf

                for m, ent_name in zip(ENTROPIC_OT_M_LEVELS, entropic_names):
                    dist = ent_mats[m]

                    _, csv_res = Retrieval.calc_map_from_distances(page_writers, dist)
                    res[ent_name][lvl].append(csv_res)

                    if config.EVAL_FOREIGN_MAP:
                        res_foreign[ent_name][lvl].append(
                            results.calc_foreign_map(
                                page_writers, foreign_writers, dist,
                                excluded_per_query=excluded_per_query))

            if wandb is not None:
                log = ({'trial': trial, 'bin_center': lvl,
                        'realised_pct': realised_pct[lvl],
                        'mean_lines_replaced': mean_units[lvl]}
                       if config.MODE == 'lines' else
                       {'trial': trial, 'nominal_pct': lvl,
                        'realised_pct': realised_pct[lvl],
                        'mean_patches_replaced': mean_units[lvl]})
                for name in all_names:
                    for metric, val in res[name][lvl][-1].items():
                        log[f'{name}/{metric}'] = val
                    if config.EVAL_FOREIGN_MAP:
                        for metric, val in res_foreign[name][lvl][-1].items():
                            log[f'foreign/{name}/{metric}'] = val
                wandb.log(log)

            # ── periodic checkpoint: write current (partial) results to disk ──
            # Guarded to trial > 0: during trial 0, levels not yet reached this
            # trial still have an EMPTY result list (res[name][lvl] == []),
            # and results.build_and_save indexes trial_dicts[0] unconditionally
            # -> IndexError, which previously killed the whole run and lost
            # everything computed so far. From trial 1 onward every level has
            # at least one entry from a completed prior trial, so this is safe.
            now = time.time()
            if trial > 0 and now - last_checkpoint >= checkpoint_every_sec:
                print(f'  [checkpoint] trial={trial + 1}/{config.N_TRIALS} lvl={lvl}: '
                      f'writing partial results to {config.OUT_DIR} ...')
                results.build_and_save(
                    res, 'Original writer labels', '', '',
                    report_configs, realised_pct, mean_units, pages_per,
                    len(valid_wps), extra_meta,
                )
                if config.EVAL_FOREIGN_MAP:
                    results.build_and_save(
                        res_foreign, 'Foreign writer labels (source page excluded)', '_foreign', '_foreign',
                        report_configs, realised_pct, mean_units, pages_per,
                        len(valid_wps), extra_meta,
                    )
                last_checkpoint = now


# ══════════════════════════════════════════════════════════════════════════════
# Save results (final, unconditional — same call as the periodic checkpoint)
# ══════════════════════════════════════════════════════════════════════════════
results.build_and_save(
    res, 'Original writer labels', '', '',
    report_configs, realised_pct, mean_units, pages_per,
    len(valid_wps), extra_meta,
)

if config.EVAL_FOREIGN_MAP:
    results.build_and_save(
        res_foreign, 'Foreign writer labels (source page excluded)', '_foreign', '_foreign',
        report_configs, realised_pct, mean_units, pages_per,
        len(valid_wps), extra_meta,
    )

if wandb is not None:
    wandb.finish()