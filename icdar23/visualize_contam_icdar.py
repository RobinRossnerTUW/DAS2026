"""
visualize_contamination_icdar.py
────────────────────────────────
Visualizes retrieval degradation under contamination for the ICDAR
HistoricalWI-2017 dataset.  Uses *exactly* the same contamination
protocol as contam_main.py:

  • page_xml.build_all_line_maps()   textline assignment
  • selection.precompute()           foreign-page selection
    (foreign page guaranteed large enough for the hardest level;
     one consistent source page per query across all levels)
  • SumPooling-PN0.4 + PCA + L2     encoding

Contamination modes
───────────────────
  lines   — replace whole textlines with strips from a foreign writer page
  random  — overwrite randomly-selected patch descriptors

Result cell border colours
──────────────────────────
  Green  (#4CAF50) — retrieved page belongs to the correct (query) writer
  Orange (#FF9800) — retrieved page belongs to the foreign (contamination) writer
  Red    (#F44336) — retrieved page belongs to some other wrong writer

The query cell always has a blue (#2196F3) border.

Cross-mode comparison  (--state_path)
──────────────────────────────────────
Pass the same --state_path to both mode runs.

  First run  (e.g. lines):  state file does not exist → computes query pages
                             and foreign assignments, then SAVES them.
  Second run (e.g. random): state file exists → LOADS the saved query pages
                             and foreign assignments so both runs use exactly
                             the same pages for a fair comparison.

Usage
─────
    # lines mode — save state for later comparison
    python visualize_contamination_icdar.py --mode lines  --state_path compare.pkl

    # random mode — reuse same queries / foreign pages
    python visualize_contamination_icdar.py --mode random --state_path compare.pkl

    # standalone (no cross-mode comparison)
    python visualize_contamination_icdar.py --mode random

    # skip wandb
    python visualize_contamination_icdar.py --no_wandb

    # extra knobs
    python visualize_contamination_icdar.py --top_k 8 --num_queries 6 --seed 99
"""

import argparse
import os
import pickle
import random
import sys
from collections import defaultdict

# ── Parse our own args BEFORE touching sys.argv for config.py ────────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--mode',        choices=['lines', 'random'], default='lines')
_pre.add_argument('--no_wandb',    action='store_true')
_pre.add_argument('--num_queries', type=int, default=40)
_pre.add_argument('--top_k',       type=int, default=5)
_pre.add_argument('--seed',        type=int, default=43)
_pre.add_argument('--out_dir',     type=str, default=None)
_pre.add_argument('--state_path',  type=str, default=None,
                  help='Path to a .pkl for saving/loading query pages + foreign '
                       'assignments so both modes use identical pages.')
_pre_args, _ = _pre.parse_known_args()

MODE = _pre_args.mode

# Inject --dataset icdar --mode <MODE> into sys.argv so config.py picks them up
for _flag, _val in [('--dataset', 'icdar'), ('--mode', MODE)]:
    if _flag not in sys.argv:
        sys.argv += [_flag, _val]

import numpy as np
import h5py
from sklearn.preprocessing import normalize
from tqdm import tqdm
from PIL import Image

# ── Local modules (config must be imported after sys.argv patching) ───────────
import config
import page_xml
import selection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_encodings import SumPooling

# ══════════════════════════════════════════════════════════════════════════════
# Hardcoded ICDAR paths
# ══════════════════════════════════════════════════════════════════════════════
IMAGES_DIR = (
    '/data/rrossner/datasets/ICDAR17/binarized_test/'
    'ScriptNet-HistoricalWI-2017-binarized'
)

# ══════════════════════════════════════════════════════════════════════════════
# Visualization constants
# ══════════════════════════════════════════════════════════════════════════════
# VIZ_LEVELS  = [0, 10, 20, 25, 30, 40, 50]   # subset of config.PCT_LEVELS
VIZ_LEVELS  = [0, 25, 50]   # subset of config.PCT_LEVELS
THUMB_H     = 150                        # pixel height of each thumbnail cell
TOP_K       = _pre_args.top_k
NUM_QUERIES = _pre_args.num_queries
SEED        = _pre_args.seed
NO_WANDB    = _pre_args.no_wandb
STATE_PATH  = _pre_args.state_path
OUT_DIR     = _pre_args.out_dir or os.path.join(
    config.EXPERIMENT_DIR, f'contamination_vis_{MODE}'
)
os.makedirs(OUT_DIR, exist_ok=True)

# Result-cell border colours
COL_CORRECT = '#4CAF50'   # green  — correct (same) writer
COL_FOREIGN = '#FF9800'   # orange — foreign (contamination source) writer
COL_WRONG   = '#F44336'   # red    — any other wrong writer
COL_QUERY   = '#2196F3'   # blue   — query cell border

# ── Thumbnail title font sizes ────────────────────────────────────────────
# Doubled from the original (6.5 / 6) so the per-image labels are readable.
QUERY_TITLE_FONTSIZE  = 15.0
RESULT_TITLE_FONTSIZE = 15.0

POOLING = SumPooling('l2', pn_alpha=0.4)

# Degradation curve metric
CURVE_METRIC = 'ap'        # 'ap' | 'top1' | 'soft5'
METRIC_LABEL = {
    'ap':    'mAP',
    'top1':  'Top-1 accuracy',
    'soft5': 'Soft-5 (hit@5)',
}

# lazy matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print(f'\n{"="*60}')
print(f' visualize_contamination_icdar.py')
print(f'  mode        : {MODE}')
print(f'  top_k       : {TOP_K}')
print(f'  num_queries : {NUM_QUERIES}')
print(f'  seed        : {SEED}')
print(f'  state_path  : {STATE_PATH}')
print(f'  out_dir     : {OUT_DIR}')
print(f'  images_dir  : {IMAGES_DIR}')
print(f'  experiment  : {config.EXPERIMENT_DIR}')
print(f'{"="*60}\n')

# ── wandb ─────────────────────────────────────────────────────────────────────
if not NO_WANDB:
    import wandb
    wandb.init(
        project=config.WANDB_PROJECT,
        name=f'icdar-vis-{MODE}',
        config=dict(
            dataset='icdar', mode=MODE, top_k=TOP_K,
            num_queries=NUM_QUERIES, viz_levels=VIZ_LEVELS, seed=SEED,
            curve_metric=CURVE_METRIC,
        ),
    )
    print(f'wandb project : {config.WANDB_PROJECT}\n')
else:
    wandb = None


# ══════════════════════════════════════════════════════════════════════════════
# Image I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_image_path(writer, page):
    """Return the path to the binarized ICDAR image, or None."""
    candidates = [
        os.path.join(IMAGES_DIR, f'{writer}-IMG_MAX_{page}.jpg'),
        os.path.join(IMAGES_DIR, f'{int(writer):04d}-IMG_MAX_{page}.jpg'),
        os.path.join(IMAGES_DIR, f'{writer}-IMG_MAX_{page}.png'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_page_image_full(writer, page):
    """Load binarized page image as a grayscale numpy array (H, W)."""
    path = _find_image_path(writer, page)
    if path is not None:
        return np.array(Image.open(path).convert('L'))
    print(f'  [WARN] image not found for W{writer} P{page} in {IMAGES_DIR}')
    return np.ones((600, 420), dtype=np.uint8) * 220


def load_page_thumbnail(writer, page, max_height=THUMB_H):
    """Load, resize to thumbnail height, and invert to white background."""
    arr = load_page_image_full(writer, page)
    img = Image.fromarray(arr)
    ratio = max_height / img.height
    new_w = max(1, int(img.width * ratio))
    thumb = np.array(img.resize((new_w, max_height), Image.LANCZOS))
    return 255 - thumb   # invert: black bg + white ink  →  white bg + dark ink


# ══════════════════════════════════════════════════════════════════════════════
# ICDAR textline y-ranges for visualization
# ══════════════════════════════════════════════════════════════════════════════

def get_textline_yranges(writer, page):
    """
    Return [(y_min, y_max), ...] sorted by y_min for one page.
    Uses the same XML lookup as page_xml._build_line_map_icdar.
    Returns [] if the XML is not found or has no lines (with a warning).
    """
    xml_path = page_xml._find_xml_icdar(writer, page)
    if xml_path is None:
        print(f'  [WARN] no XML for textline viz: W{writer} P{page}')
        return []
    textlines = page_xml._parse_icdar_xml(xml_path)
    if not textlines:
        print(f'  [WARN] XML parsed but no textlines: W{writer} P{page}  ({xml_path})')
        return []
    return [(tl['bbox'][1], tl['bbox'][3]) for tl in textlines]


# ══════════════════════════════════════════════════════════════════════════════
# Encoding helpers  (inline — no multiprocessing needed for visualization)
# ══════════════════════════════════════════════════════════════════════════════

def read_and_norm(h5, key):
    """Read a patch matrix from HDF5 and L2-normalise each row."""
    arr   = h5[key][:].astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms < 1e-8, 1.0, norms)


def make_mixed_lines(own, foreign, sel, line_map, f_line_map):
    """Apply lines-mode contamination to a patch matrix."""
    if sel['n_replace'] == 0:
        return own
    n_own = own.shape[0]
    n_for = foreign.shape[0]
    sorted_lines = sorted(line_map.keys())
    keep = [
        i
        for li in sorted_lines
        if li not in sel['lines_to_replace']
        for i in line_map[li]
        if i < n_own
    ]
    fgn = [
        i
        for fli in sel['foreign_lines']
        for i in f_line_map.get(fli, [])
        if i < n_for
    ]
    if keep and fgn:
        return np.concatenate([own[keep], foreign[fgn]], axis=0)
    if keep:
        return own[keep]
    print('  [WARN] make_mixed_lines: no kept patches — falling back to full own')
    return own


def make_mixed_random(own, foreign, sel):
    """Apply random-patch-mode contamination to a patch matrix."""
    if sel['n_replace'] == 0:
        return own
    dest  = sel['dest_idx']
    src   = sel['src_idx']
    n_own = own.shape[0]
    n_for = foreign.shape[0]
    mask  = (dest < n_own) & (src < n_for)
    if not mask.any():
        print('  [WARN] make_mixed_random: mask is all-False — returning own unchanged')
        return own
    mixed = own.copy()
    mixed[dest[mask]] = foreign[src[mask]]
    return mixed


def encode_and_project(patches, pca):
    enc = POOLING.encode([patches])
    tf  = pca.transform(enc.reshape(1, -1))
    return normalize(tf, axis=1).flatten()


# ══════════════════════════════════════════════════════════════════════════════
# Cross-mode selection helper
# ══════════════════════════════════════════════════════════════════════════════

def make_sels_random_fixed_foreign(valid_wps, foreign_map, page_num_patches, seed):
    """
    Mirrors selection._precompute_random but accepts a pre-determined
    foreign_map (loaded from a lines-mode state file) so both modes use
    the same source pages.  Only processes wps present in foreign_map.
    """
    np_rng = np.random.RandomState(seed)
    sels   = {}
    for wp in valid_wps:
        if wp not in foreign_map:
            continue
        fwp   = foreign_map[wp]
        n_own = page_num_patches.get(wp,  0)
        n_for = page_num_patches.get(fwp, 0)
        per_level, cache = {}, {}
        for pct in config.PCT_LEVELS:
            n = min(int(round(pct * n_own / 100.0)), n_own, n_for)
            if n in cache:
                per_level[pct] = cache[n]
                continue
            if n == 0:
                sel = {'n_replace': 0}
            else:
                sel = {
                    'n_replace': n,
                    'dest_idx':  np_rng.choice(n_own, n, replace=False),
                    'src_idx':   np_rng.choice(n_for, n, replace=False),
                }
            cache[n]       = sel
            per_level[pct] = sel
        sels[wp] = per_level
    return sels


# ══════════════════════════════════════════════════════════════════════════════
# Load artifacts
# ══════════════════════════════════════════════════════════════════════════════
print('Loading artifacts...')

with open(config.INDEX_PATH, 'rb') as f:
    index = pickle.load(f)
pca = pickle.load(open(config.PCA_PATH, 'rb'))

print(f'  Index entries : {len(index)}')
print(f'  Sample entry  : { {k: v for k, v in list(index[0].items())[:6]} }')

# Build lookup tables — try both patch_end/patch_start and num_patches
wp_to_key        = {}
page_num_patches = {}
_missing_patch_count = 0
for e in index:
    wp            = (e['writer'], e['page'])
    wp_to_key[wp] = e['hdf5_key']
    if 'patch_end' in e and 'patch_start' in e:
        page_num_patches[wp] = e['patch_end'] - e['patch_start']
    elif 'num_patches' in e:
        page_num_patches[wp] = e['num_patches']
    else:
        page_num_patches[wp] = 0
        _missing_patch_count += 1

if _missing_patch_count:
    print(f'  [WARN] {_missing_patch_count} index entries had no patch count field '
          f'(tried patch_end-patch_start and num_patches) — defaulting to 0')

writer_to_pages = defaultdict(list)
for e in index:
    writer_to_pages[e['writer']].append((e['writer'], e['page']))

all_writers = list(writer_to_pages.keys())
all_wps     = list(wp_to_key.keys())
print(f'  Total pages   : {len(all_wps)}')
print(f'  Total writers : {len(all_writers)}')


# ══════════════════════════════════════════════════════════════════════════════
# Build page metadata
# ══════════════════════════════════════════════════════════════════════════════
if MODE == 'lines':
    print('\nParsing PAGE XML and building line maps...')
    page_line_map, page_num_lines, xml_stats = page_xml.build_all_line_maps(all_wps)
    print(
        f"  XML found        : {xml_stats['xml_found']}\n"
        f"  XML missing      : {xml_stats['xml_missing']}  (Y-cluster fallback)\n"
        f"  coords missing   : {xml_stats['coords_missing']}\n"
        f"  patches outside  : {xml_stats['total_outside']}"
    )
    valid_wps = [wp for wp in all_wps if page_num_lines[wp] >= config.MIN_LINES]
    print(f'\n  Valid pages (>= {config.MIN_LINES} lines) : '
          f'{len(valid_wps)} / {len(all_wps)}')
    all_n = [page_num_lines[wp] for wp in valid_wps]
    if all_n:
        print(f'  Lines/page : mean={np.mean(all_n):.1f}  '
              f'median={np.median(all_n):.1f}  '
              f'min={min(all_n)}  max={max(all_n)}')
    else:
        print('  [FATAL] No valid pages after line filter — exiting.')
        sys.exit(1)
else:
    page_line_map  = None
    page_num_lines = None
    valid_wps      = all_wps
    print(f'\nRandom mode — all {len(valid_wps)} pages are valid.')
    n_arr = np.array([page_num_patches[wp] for wp in valid_wps])
    if n_arr.max() > 0:
        print(f'  Patches/page : mean={n_arr.mean():.1f}  '
              f'median={np.median(n_arr):.1f}  '
              f'min={n_arr.min()}  max={n_arr.max()}')
    else:
        print('  [WARN] All patch counts are 0 — check index structure.')

valid_writer_to_pages = defaultdict(list)
for wp in valid_wps:
    valid_writer_to_pages[wp[0]].append(wp)
valid_writers = list(valid_writer_to_pages.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Select query pages + precompute contamination selections
# ══════════════════════════════════════════════════════════════════════════════

if STATE_PATH and os.path.isfile(STATE_PATH):
    # ── load saved state ──────────────────────────────────────────────────
    print(f'\nLoading comparison state from {STATE_PATH} ...')
    _state      = pickle.load(open(STATE_PATH, 'rb'))
    query_pages = _state['query_pages']
    foreign_map = _state['foreign_map']

    _missing = [wp for wp in query_pages if wp not in wp_to_key]
    if _missing:
        print(f'  [WARN] {len(_missing)} loaded query pages not in index — '
              f'they will produce empty results.')
    print(f'  Loaded {len(query_pages)} query pages from state.')

    if MODE == 'lines':
        # Recompute lines-mode sels (need line indices); foreign_map from state
        # is used only for panel labels — the freshly computed sels determine
        # which actual lines get replaced, which is fine since sels are seeded.
        _, sels = selection.precompute(
            SEED,
            valid_wps, valid_writers, valid_writer_to_pages,
            page_num_lines=page_num_lines,
            page_line_map=page_line_map,
            page_num_patches=page_num_patches,
        )
        print('  Lines sels recomputed; foreign_map overridden from state.')
    else:
        # Random mode: use the fixed foreign pages from the state
        sels = make_sels_random_fixed_foreign(
            valid_wps, foreign_map, page_num_patches, SEED)
        print('  Random sels computed with fixed foreign pages from state.')

else:
    # ── fresh selection ───────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)

    _min_lines_for_query = 6
    if MODE == 'lines':
        eligible = [
            wp for wp in valid_wps
            if page_num_lines[wp] >= _min_lines_for_query
            and len(valid_writer_to_pages[wp[0]]) >= 2
        ]
    else:
        eligible = [
            wp for wp in valid_wps
            if len(valid_writer_to_pages[wp[0]]) >= 2
        ]

    if len(eligible) < NUM_QUERIES:
        print(f'  [WARN] Only {len(eligible)} eligible query pages '
              f'(requested {NUM_QUERIES}) — using all.')
    random.shuffle(eligible)
    query_pages = eligible[:NUM_QUERIES]

    print('\nPrecomputing contamination selections...')
    foreign_map, sels = selection.precompute(
        SEED,
        valid_wps, valid_writers, valid_writer_to_pages,
        page_num_lines=page_num_lines,
        page_line_map=page_line_map,
        page_num_patches=page_num_patches,
    )

    if STATE_PATH:
        pickle.dump(
            dict(query_pages=query_pages, foreign_map=foreign_map),
            open(STATE_PATH, 'wb'),
        )
        print(f'  Comparison state saved → {STATE_PATH}')

print(f'\nSelected {len(query_pages)} query pages:')
for qwp in query_pages:
    fwp   = foreign_map[qwp]
    extra = (f'  {page_num_lines[qwp]} lines'
             if MODE == 'lines' and page_num_lines
             else f'  {page_num_patches[qwp]} patches')
    print(f'  W{qwp[0]} P{qwp[1]}{extra}  '
          f'({len(valid_writer_to_pages[qwp[0]])-1} same-writer pages in gallery)'
          f'  → foreign W{fwp[0]} P{fwp[1]}')


# ══════════════════════════════════════════════════════════════════════════════
# Build clean gallery embeddings for all valid pages
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding clean gallery embeddings...')
clean_embs = {}
with h5py.File(config.H5_PATH, 'r') as h5:
    for wp in tqdm(valid_wps, desc='  Gallery'):
        patches        = read_and_norm(h5, wp_to_key[wp])
        clean_embs[wp] = POOLING.encode([patches]).flatten()

emb_matrix = np.stack([clean_embs[wp] for wp in valid_wps])
emb_matrix = normalize(pca.transform(emb_matrix), axis=1)
for i, wp in enumerate(valid_wps):
    clean_embs[wp] = emb_matrix[i]

wp_to_idx = {wp: i for i, wp in enumerate(valid_wps)}
print(f'  Gallery matrix shape: {emb_matrix.shape}')


# ══════════════════════════════════════════════════════════════════════════════
# Compute contaminated rankings for each query page × each VIZ_LEVEL
# ══════════════════════════════════════════════════════════════════════════════
print('\nComputing contaminated query embeddings and rankings...')
query_rankings = {}

with h5py.File(config.H5_PATH, 'r') as h5:
    for qwp in tqdm(query_pages, desc='  Query pages'):
        qw, qp = qwp
        fwp    = foreign_map[qwp]
        fw, fp = fwp

        own     = read_and_norm(h5, wp_to_key[qwp])
        foreign = read_and_norm(h5, wp_to_key[fwp])

        lm   = page_line_map[qwp] if MODE == 'lines' else None
        f_lm = page_line_map[fwp] if MODE == 'lines' else None

        # ── writer page count debug ───────────────────────────────────────
        n_qw_pages = len(valid_writer_to_pages.get(qw, []))
        n_fw_pages = len(valid_writer_to_pages.get(fw, []))
        print(f'  W{qw} P{qp}  →  foreign W{fw} P{fp}')
        print(f'    query   writer W{qw}: {n_qw_pages} pages in gallery '
              f'({n_qw_pages - 1} retrievable, self excluded)')
        print(f'    foreign writer W{fw}: {n_fw_pages} pages in gallery '
              f'({n_fw_pages - 1} retrievable, source page P{fp} excluded)')

        tl_yranges   = get_textline_yranges(qw, qp) if MODE == 'lines' else []
        f_tl_yranges = get_textline_yranges(fw, fp) if MODE == 'lines' else []

        if MODE == 'lines' and not tl_yranges:
            print(f'  [WARN] No textline y-ranges for query W{qw} P{qp} — '
                  f'pixel splice will be skipped for this query.')

        query_rankings[qwp] = {}
        for lvl in VIZ_LEVELS:
            sel = sels[qwp][lvl]

            # ── contaminate ───────────────────────────────────────────────
            if MODE == 'lines':
                mixed          = make_mixed_lines(own, foreign, sel, lm, f_lm)
                replaced_lines = sel.get('lines_to_replace', set()) if lvl > 0 else set()
            else:
                mixed          = make_mixed_random(own, foreign, sel)
                replaced_lines = set()

            # ── encode + rank ─────────────────────────────────────────────
            enc  = encode_and_project(mixed, pca)
            sims = emb_matrix @ enc
            sims[wp_to_idx[qwp]] = -999.0   # exclude self
            if fwp in wp_to_idx:
                sims[wp_to_idx[fwp]] = -999.0   # exclude foreign source page (patch leakage)

            ranked = np.argsort(-sims)
            top_k  = [
                dict(
                    writer     = int(valid_wps[ri][0]),
                    page       = int(valid_wps[ri][1]),
                    correct    = int(valid_wps[ri][0]) == int(qw),
                    is_foreign = int(valid_wps[ri][0]) == int(fw),
                    similarity = float(sims[ri]),
                )
                for ri in ranked[:TOP_K]
            ]

            # ── per-level metrics over full gallery ───────────────────────
            ranked_writers = np.array([valid_wps[ri][0] for ri in ranked])
            relevant       = (ranked_writers == int(qw))
            n_rel          = relevant.sum()

            if n_rel > 0:
                hit_positions = np.where(relevant)[0] + 1
                precisions    = np.arange(1, n_rel + 1) / hit_positions
                ap            = float(precisions.mean())
                top1          = bool(relevant[0])
                soft5         = bool(relevant[:5].any())
            else:
                ap = top1 = soft5 = 0.0

            query_rankings[qwp][lvl] = dict(
                top_k          = top_k,
                foreign_wp     = fwp,
                replaced_lines = replaced_lines,
                tl_yranges     = tl_yranges,
                f_tl_yranges   = f_tl_yranges,
                ap             = ap,
                top1           = top1,
                soft5          = soft5,
            )

        # per-query sanity print
        print(f'    W{qw} P{qp} :  '
              f'mAP   0%→{query_rankings[qwp][0]["ap"]:.3f}  '
              f'50%→{query_rankings[qwp][50]["ap"]:.3f}   |  '
              f'top1  0%→{int(query_rankings[qwp][0]["top1"])}  '
              f'50%→{int(query_rankings[qwp][50]["top1"])}')


# ══════════════════════════════════════════════════════════════════════════════
# Pixel-splice helpers  (lines mode only)
# ══════════════════════════════════════════════════════════════════════════════

def splice_foreign_lines(qimg, fimg, replaced_lines, tl_yranges, f_tl_yranges):
    """
    Return (spliced_img, n_spliced).
    Copies horizontal strips from fimg into qimg at the replaced line positions.
    No crop offset for ICDAR — coordinates are direct image pixels.
    """
    out             = qimg.copy()
    q_h, q_w        = qimg.shape[:2]
    f_h             = fimg.shape[0]
    replaced_sorted = sorted(replaced_lines)

    n_spliced = 0
    for i, li in enumerate(replaced_sorted):
        if li >= len(tl_yranges):
            print(f'    [splice WARN] line idx {li} >= len(tl_yranges)={len(tl_yranges)}')
            continue
        y_min, y_max = tl_yranges[li]
        dst_y1 = max(0,   int(y_min))
        dst_y2 = min(q_h, int(y_max))
        dst_h  = dst_y2 - dst_y1
        if dst_h <= 0:
            continue

        if i < len(f_tl_yranges):
            fy_min, fy_max = f_tl_yranges[i]
            src_y1 = max(0,   int(fy_min))
            src_y2 = min(f_h, int(fy_max))
            if src_y2 <= src_y1:
                out[dst_y1:dst_y2, :] = 180   # mid-grey sentinel
                continue
            strip     = fimg[src_y1:src_y2, :]
            strip_pil = Image.fromarray(strip).resize((q_w, dst_h), Image.LANCZOS)
            out[dst_y1:dst_y2, :] = np.array(strip_pil)[:dst_h, :q_w]
            n_spliced += 1
        else:
            print(f'    [splice WARN] no foreign line for replaced idx {i}')

    return out, n_spliced


def draw_line_borders(img_rgb, replaced_lines, tl_yranges, orig_h, border_px=1):
    """Draw orange rectangles around replaced textlines on a thumbnail."""
    thumb_h    = img_rgb.shape[0]
    scale      = thumb_h / max(orig_h, 1)
    hx         = COL_FOREIGN.lstrip('#')
    border_col = [int(hx[i:i+2], 16) for i in (0, 2, 4)]
    for li in replaced_lines:
        if li >= len(tl_yranges):
            continue
        y_min, y_max = tl_yranges[li]
        y1 = max(0,           int(y_min * scale))
        y2 = min(thumb_h - 1, int(y_max * scale))
        if y2 <= y1:
            continue
        b = border_px
        img_rgb[max(0, y1 - b): y1,       :]  = border_col
        img_rgb[y2: min(thumb_h, y2 + b), :]  = border_col
        img_rgb[y1: y2,              :b]       = border_col
        img_rgb[y1: y2,              -b:]      = border_col
    return img_rgb


# ══════════════════════════════════════════════════════════════════════════════
# Generate per-query panels
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating panels...')

for qi, qwp in enumerate(query_pages):
    qw, qp     = qwp
    n_levels   = len(VIZ_LEVELS)
    n_cols     = 1 + TOP_K
    fw_global  = foreign_map[qwp][0]

    # NOTE: row height bumped from 2.6 → 3.3 so the larger title fonts have
    # room to breathe without tight_layout shrinking the actual page images.
    fig, axes = plt.subplots(n_levels, n_cols,
                             figsize=(2.0 * n_cols, 3.3 * n_levels))
    if n_levels == 1:
        axes = axes[np.newaxis, :]

    n_same = len(valid_writer_to_pages[qw]) - 1
    fig.suptitle(
        f'Historical-WI | mode={MODE} | Query W{qw} P{qp}  '
        f'[{n_same} same-writer page{"s" if n_same != 1 else ""} in gallery  '
        f'| foreign source: W{fw_global}]',
        fontsize=11, fontweight='bold', y=1.01,
    )

    qimg_full = load_page_image_full(qw, qp)
    q_h_orig  = qimg_full.shape[0]

    for row, lvl in enumerate(VIZ_LEVELS):
        ranking        = query_rankings[qwp][lvl]
        top_k          = ranking['top_k']
        replaced_lines = ranking['replaced_lines']
        tl_yranges     = ranking['tl_yranges']
        f_tl_yranges   = ranking['f_tl_yranges']
        fw, fp         = ranking['foreign_wp']
        ap_val         = ranking['ap']

        # ── query thumbnail cell ──────────────────────────────────────────
        ax = axes[row, 0]

        if MODE == 'lines' and replaced_lines and tl_yranges:
            fimg_full          = load_page_image_full(fw, fp)
            display_full, n_sp = splice_foreign_lines(
                qimg_full, fimg_full, replaced_lines, tl_yranges, f_tl_yranges)
            if n_sp == 0:
                print(f'  [WARN] level {lvl}%: 0 lines actually spliced for W{qw} P{qp}')
        else:
            display_full = qimg_full

        thumb_pil = Image.fromarray(display_full)
        ratio     = THUMB_H / thumb_pil.height
        thumb_w   = max(1, int(thumb_pil.width * ratio))
        thumb     = np.array(thumb_pil.resize((thumb_w, THUMB_H), Image.LANCZOS))
        thumb     = 255 - thumb   # invert: white background, dark ink
        img_rgb   = np.stack([thumb] * 3, axis=-1) if thumb.ndim == 2 else thumb.copy()

        if MODE == 'lines' and replaced_lines and tl_yranges:
            img_rgb = draw_line_borders(img_rgb, replaced_lines, tl_yranges, q_h_orig)

        ax.imshow(img_rgb, aspect='auto')
        n_replaced = (len(replaced_lines) if MODE == 'lines'
                      else sels[qwp][lvl]['n_replace'])
        ax.set_title(
            f'{lvl}% → W{fw}\n'
            f'({n_replaced} {"lines" if MODE == "lines" else "patches"} replaced)\n',
            #f'mAP={ap_val:.3f}',
            fontsize=QUERY_TITLE_FONTSIZE, fontweight='bold',
        )
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COL_QUERY)
            spine.set_linewidth(2)

        # ── retrieval result cells ────────────────────────────────────────
        for col_idx, result in enumerate(top_k):
            ax = axes[row, 1 + col_idx]
            rimg = load_page_thumbnail(result['writer'], result['page'], THUMB_H)
            ax.imshow(rimg, cmap='gray', aspect='auto')

            if result['correct']:
                color = COL_CORRECT
                tag   = ''
            elif result['is_foreign']:
                color = COL_FOREIGN
                tag   = ''
            else:
                color = COL_WRONG
                tag   = ''

            # Title in neutral dark colour — the border carries the colour signal
            ax.set_title(
                f'#{col_idx+1} W{result["writer"]}{tag}\n'
                f'{result["similarity"]:.3f}',
                fontsize=RESULT_TITLE_FONTSIZE, color='#222222', fontweight='bold',
            )
            ax.axis('off')
            # draw border as a patch — spines are suppressed by axis('off')
            ax.add_patch(mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                linewidth=2, edgecolor=color, facecolor='none', clip_on=False,
            ))

    # colour legend
    legend_patches = [
        mpatches.Patch(color=COL_CORRECT, label='Correct writer'),
        mpatches.Patch(color=COL_FOREIGN, label=f'Foreign source (W{fw_global})'),
        mpatches.Patch(color=COL_WRONG,   label='Other wrong writer'),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=4, fontsize=14, framealpha=0.9,
               handlelength=2.2, handleheight=1.4, borderpad=0.8,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    panel_path = os.path.join(OUT_DIR, f'query_{qw}_{qp}_{MODE}.png')
    fig.savefig(panel_path, dpi=450, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {panel_path}')

    if wandb is not None:
        wandb.log({
            f'query_{qw}_{qp}': wandb.Image(
                panel_path,
                caption=(f'W{qw} P{qp} | mode={MODE} | top-{TOP_K} | '
                         f'{n_same} same-writer in gallery | foreign W{fw_global}'),
            )
        })
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Summary degradation curve
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating degradation curve...')

curve_vals = [
    float(np.mean([query_rankings[qwp][lvl][CURVE_METRIC] for qwp in query_pages]))
    for lvl in VIZ_LEVELS
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(VIZ_LEVELS, curve_vals, 'o-', color=COL_QUERY, lw=2, ms=8)
ax.fill_between(VIZ_LEVELS, curve_vals, alpha=0.15, color=COL_QUERY)
for pct, val in zip(VIZ_LEVELS, curve_vals):
    ax.annotate(f'{val:.3f}', (pct, val),
                textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Nominal contamination (%)', fontsize=12)
ax.set_ylabel(METRIC_LABEL[CURVE_METRIC], fontsize=12)
title_str = (
    f'Historical-WI — Retrieval Degradation Under '
    f'{"Line" if MODE == "lines" else "Random-Patch"} Contamination  '
    f'(n={len(query_pages)} queries)'
)
if STATE_PATH and os.path.isfile(STATE_PATH):
    title_str += '\n[comparison run — same queries & foreign pages as partner mode]'
ax.set_title(title_str, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_xticks(VIZ_LEVELS)
ax.grid(True, alpha=0.3)

curve_path = os.path.join(OUT_DIR, f'degradation_curve_{MODE}.png')
fig.savefig(curve_path, dpi=450, bbox_inches='tight', facecolor='white')
print(f'Saved: {curve_path}')

if wandb is not None:
    wandb.log({f'degradation_curve_{MODE}': wandb.Image(curve_path)})
    wandb.finish()
plt.close(fig)

# ── persist rankings ──────────────────────────────────────────────────────────
ranks_path = os.path.join(OUT_DIR, f'rankings_{MODE}.pkl')
pickle.dump(
    dict(
        query_rankings = {f'{w}_{p}': v for (w, p), v in query_rankings.items()},
        query_pages    = [(int(w), int(p)) for w, p in query_pages],
        viz_levels     = VIZ_LEVELS,
        top_k          = TOP_K,
        mode           = MODE,
        seed           = SEED,
        curve_metric   = CURVE_METRIC,
    ),
    open(ranks_path, 'wb'),
)
print(f'Saved: {ranks_path}')
print(f'\nDone.  All outputs in {OUT_DIR}')