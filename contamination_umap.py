#!/usr/bin/env python3
"""
contamination_umap.py
─────────────────────
Visualises how contaminating a page moves it through UMAP embedding space,
with full support for both datasets and both contamination modes.

TWO CONFIGURATION AXES
    DATASET : 'cvl' | 'icdar'
    MODE    : 'lines' | 'random'

  lines  — replace whole textlines (uses PAGE XML annotations).
           Contamination levels: 0, 1, 2, … up to n_lines//2 replaced lines.
           The script reports actual % of content replaced per focal page.

  random — replace random patches (no XML required).
           Contamination levels: 0%, 5%, 10%, … up to MAX_PCT_RANDOM.
           n_replace = round(pct * N_patches / 100)

QUICK START
  1. Edit the CONFIG block below (paths, DATASET, MODE, focal pages).
  2. Run:  python contamination_umap.py
           python contamination_umap.py --dataset icdar --mode random
  3. Download the whole OUT_DIR folder.
  4. Open contamination_trajectory.html in any browser (file:// works fine).

OUTPUT (all in OUT_DIR)
    contamination_trajectory.html  — self-contained interactive UMAP
    trajectory_data.json           — raw data used by the HTML
    images/                        — page thumbnails
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit this block, then just run the script
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = dict(

    # ── Two main axes ─────────────────────────────────────────────────────────
    dataset = 'cvl',   # 'cvl' | 'icdar'
    mode    = 'random',   # 'lines' | 'random'

    # ── Per-dataset paths ─────────────────────────────────────────────────────
    # CVL paths
    cvl_experiment_dir = '/data/rrossner/datasets/CVL/7663856',
    cvl_patches_dir    = '/data/rrossner/DAS2026/cvl/test_patches_coords',
    cvl_xml_dir        = '/data/rrossner/datasets/CVL/xml',
    cvl_pages_dir      = '/data/rrossner/datasets/CVL/cvl-database-cropped-1-1.zip',
    cvl_embed_path     = '/data/rrossner/datasets/CVL/7663856/embeddings_SumPooling-PN0p4_512.npy',
    cvl_writer_path    = '/data/rrossner/datasets/CVL/7663856/writers_SumPooling-PN0p4_512.npy',

    # ICDAR paths — xml_dirs is a LIST covering all three colour sub-datasets
    icdar_experiment_dir = '/data/rrossner/DAS2026/icdar23/experiments/test_repository-2026-05-08-12-23',
    icdar_patches_dir    = '/data/rrossner/DAS2026/icdar/test_patches_coords',
    icdar_xml_dirs       = [
        '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-1/page',
        '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-2/page',
        '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-3/page',
    ],
    icdar_pages_dir      = '/data/rrossner/datasets/ICDAR17/binarized_test/ScriptNet-HistoricalWI-2017-binarized/',
    icdar_embed_path     = '/data/rrossner/DAS2026/icdar23/experiments/test_repository-2026-05-08-12-23/embeddings_SumPooling-PN0p4_512.npy',
    icdar_writer_path    = '/data/rrossner/DAS2026/icdar23/experiments/test_repository-2026-05-08-12-23/writers_SumPooling-PN0p4_512.npy',

    # ── Output ────────────────────────────────────────────────────────────────
    out_dir = '/data/rrossner/DAS2026/umap_contamination',

    # ── Focal page selection ──────────────────────────────────────────────────
    # Option A — pin specific (writer, page) pairs. Set to None to use n_focal.
    # CVL example:   [(205, 1), (213, 2), (324, 2), (168, 4)]
    # ICDAR example: [('270', '3'), ('271', '4')]   (string IDs for ICDAR)
    focal_pages = None,
    # focal_pages = [('2326', '626911')],
    # focal_pages = [('290', '4')],

    # Option B — auto-pick N pages (one per randomly chosen writer)
    n_focal = 30,

    # ── Contamination settings ────────────────────────────────────────────────
    # lines mode: levels are 0, 1, … up to n_lines // 2  (per-page)
    min_lines = 2,   # skip pages with fewer lines (lines mode only)

    # random mode: which % levels to show as trajectory points
    random_pct_levels = [0, 10, 20, 30, 40, 50],

    # Fix a single foreign writer for all focal pages (int/str ID, or None = random)
    foreign_writer = None,
    # foreign_writer = '1061',
    # foreign_writer = '384',

    # ── Pooling ───────────────────────────────────────────────────────────────
    # 'SumPooling-PN0p4' | 'SumPooling-PNsqrt' | 'MaxPooling' | 'LSEPooling'
    pooling = 'SumPooling-PN0p4',

    # ── Misc ──────────────────────────────────────────────────────────────────
    thumb_size = 100,
    seed       = 42,
)
# ══════════════════════════════════════════════════════════════════════════════
#  END CONFIG
# ══════════════════════════════════════════════════════════════════════════════

import os, sys, re, json, pickle, random, glob, io, argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import normalize
import umap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_encodings import SumPooling, MaxPooling, LSEPooling


# ══════════════════════════════════════════════════════════════════════════════
#  CLI overrides
# ══════════════════════════════════════════════════════════════════════════════
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--dataset',        choices=['cvl', 'icdar'], default=None)
_parser.add_argument('--mode',           choices=['lines', 'random'], default=None)
_parser.add_argument('--out_dir',        default=None)
_parser.add_argument('--n_focal',        type=int, default=None)
_parser.add_argument('--focal_pages',    default=None,
                     help="e.g. '205:1,213:2' (writer:page pairs, colon-separated)")
_parser.add_argument('--foreign_writer', default=None)
_parser.add_argument('--pooling',        default=None)
_parser.add_argument('--seed',           type=int, default=None)
_parser.add_argument('--min_lines',      type=int, default=None)
_cli, _ = _parser.parse_known_args()

def _C(key):
    val = getattr(_cli, key, None)
    return val if val is not None else CONFIG[key]

DATASET        = _C('dataset')
MODE           = _C('mode')
OUT_DIR        = _C('out_dir')
N_FOCAL        = _C('n_focal')
POOLING_NAME   = _C('pooling')
SEED           = _C('seed')
MIN_LINES      = _C('min_lines')
FOREIGN_WRITER = _C('foreign_writer')

if _cli.focal_pages:
    FOCAL_PAGES = [tuple(t.split(':')) for t in _cli.focal_pages.split(',')]
else:
    FOCAL_PAGES = CONFIG['focal_pages']

if _cli.foreign_writer is not None:
    FOREIGN_WRITER = _cli.foreign_writer

assert DATASET in ('cvl', 'icdar'), f'Unknown dataset: {DATASET!r}'
assert MODE    in ('lines', 'random'), f'Unknown mode: {MODE!r}'

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
THUMB_DIR = os.path.join(OUT_DIR, 'images')
os.makedirs(THUMB_DIR, exist_ok=True)

# ── Resolve dataset-specific config ──────────────────────────────────────────
EXPERIMENT_DIR = CONFIG[f'{DATASET}_experiment_dir']
PATCHES_DIR    = CONFIG[f'{DATASET}_patches_dir']
PAGES_DIR      = CONFIG[f'{DATASET}_pages_dir']
EMBED_PATH     = CONFIG[f'{DATASET}_embed_path']
WRITER_PATH    = CONFIG[f'{DATASET}_writer_path']
RANDOM_LEVELS  = CONFIG['random_pct_levels']

# XML_DIRS: ICDAR uses a list of three colour dirs; CVL uses a single dir wrapped in a list
if DATASET == 'icdar':
    XML_DIRS = CONFIG['icdar_xml_dirs']
else:
    XML_DIRS = [CONFIG['cvl_xml_dir']]

H5_PATH    = os.path.join(EXPERIMENT_DIR, 'raw_patches_test.h5')
INDEX_PATH = os.path.join(EXPERIMENT_DIR, 'patch_index_test.pkl')
PCA_PATH   = os.path.join(EXPERIMENT_DIR, 'pca.pkl')

CVL_PAGE_NS   = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}'
ICDAR_PAGE_NS = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}'

MODE_LABEL = 'lines' if MODE == 'lines' else 'random-patches'
print(f'\n{"─"*60}')
print(f'  Dataset : {DATASET.upper()}')
print(f'  Mode    : {MODE_LABEL}')
print(f'  Out dir : {OUT_DIR}')
print(f'{"─"*60}\n')

POOLING_MAP = {
    'SumPooling-PN0p4':  SumPooling('l2', pn_alpha=0.4),
    'SumPooling-PNsqrt': SumPooling('powernorm'),
    'MaxPooling':        MaxPooling('l2'),
    'LSEPooling':        LSEPooling('l2'),
}
pooling_fn = POOLING_MAP[POOLING_NAME]


# ══════════════════════════════════════════════════════════════════════════════
#  Patch coordinate helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_coords_json(path):
    with open(path) as f:
        return json.load(f)

def _resolve_coords_cvl(writer, page):
    for stem in [f'{int(writer):04d}-{int(page)}-cropped',
                 f'{writer}-{page}-cropped',
                 f'{int(writer):04d}-{int(page)}',
                 f'{writer}-{page}']:
        p = os.path.join(PATCHES_DIR, stem, 'coordinates.json')
        if os.path.isfile(p):
            return p
    return None

def _resolve_coords_icdar(writer, page):
    for stem in [f'{writer}-IMG_MAX_{page}',
                 f'{int(writer):04d}-IMG_MAX_{page}',
                 f'{writer}-{page}',
                 str(page)]:
        p = os.path.join(PATCHES_DIR, stem, 'coordinates.json')
        if os.path.isfile(p):
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  XML helpers — CVL (PAGE-2010)
# ══════════════════════════════════════════════════════════════════════════════

def _find_xml_cvl(writer, page):
    """Search CVL xml dir for the PAGE XML matching (writer, page)."""
    xml_dir = XML_DIRS[0]   # CVL has exactly one xml dir
    candidates = [
        f'{int(writer):04d}-{int(page)}.xml',
        f'{int(writer):04d}-{int(page)}-cropped.xml',
        f'{writer}-{page}.xml',
        f'{writer}-{page}-cropped.xml',
        f'{int(writer):04d}-{int(page)}_attributes.xml',
    ]
    for fname in candidates:
        p = os.path.join(xml_dir, fname)
        if os.path.isfile(p):
            return p
    # Walk fallback for non-standard subdirectory layouts
    for root_d, _, files in os.walk(xml_dir):
        for fn in files:
            if fn.endswith('.xml') and str(writer) in fn and str(page) in fn:
                return os.path.join(root_d, fn)
    return None


def _parse_xml_cvl(xml_path):
    NS = CVL_PAGE_NS
    with open(xml_path, 'rb') as f:
        raw = f.read()
    if raw[:2] == b'\xff\xfe':
        text = raw.decode('utf-16-le', errors='replace')
    elif raw[:2] == b'\xfe\xff':
        text = raw.decode('utf-16-be', errors='replace')
    else:
        try:
            text = raw.decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            text = raw.decode('latin-1', errors='replace')
    text = re.sub(r'<\?xml[^?]*\?>', '', text, count=1)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]', '', text)
    try:
        root = ET.fromstring(text.encode('utf-8'))
    except ET.ParseError:
        return 0, []
    crop_y = 0
    for tl in root.iter(f'{NS}dkTextLines'):
        cy = tl.get('cropY')
        if cy is not None:
            crop_y = int(float(cy))
            break
    lines = []
    for region in root.iter(f'{NS}AttrRegion'):
        if region.get('attrType') == '2' and region.get('fontType') in ('2', None):
            ys = [float(pt.get('y')) for pt in region.iter(f'{NS}Point') if pt.get('y')]
            if ys:
                lines.append((min(ys), max(ys)))
    lines.sort(key=lambda t: t[0])
    return crop_y, lines


def _assign_patches_cvl(coords_json, textlines, crop_y):
    lmap = defaultdict(list)
    for pc in coords_json['patches']:
        py  = pc['y'] + crop_y
        idx = pc['patch_idx']
        ok  = False
        for li, (y1, y2) in enumerate(textlines):
            if y1 <= py <= y2:
                lmap[li].append(idx)
                ok = True
                break
        if not ok and textlines:
            dists = [min(abs(py - y1), abs(py - y2)) for y1, y2 in textlines]
            lmap[int(np.argmin(dists))].append(idx)
    return dict(lmap)


def _ycluster_cvl(patches):
    if not patches:
        return {}, 0
    arr = sorted(patches, key=lambda p: p['y'])
    ys  = np.array([p['y'] for p in arr])
    gaps = np.diff(ys)
    pos  = gaps[gaps > 5]
    thresh = (np.median(pos) if len(pos) else 30) * 0.5
    li, lmap = 0, defaultdict(list)
    lmap[0].append(arr[0]['patch_idx'])
    for i in range(1, len(arr)):
        if ys[i] - ys[i - 1] > thresh:
            li += 1
        lmap[li].append(arr[i]['patch_idx'])
    return dict(lmap), li + 1


def build_line_map_cvl(writer, page):
    coords_p = _resolve_coords_cvl(writer, page)
    if coords_p is None:
        return {}, 0
    cj    = _load_coords_json(coords_p)
    xml_p = _find_xml_cvl(writer, page)
    if xml_p is None:
        lmap, n = _ycluster_cvl(cj['patches'])
        return lmap, n
    crop_y, textlines = _parse_xml_cvl(xml_p)
    if not textlines:
        return {}, 0
    lmap = _assign_patches_cvl(cj, textlines, crop_y)
    return lmap, len(lmap)


# ══════════════════════════════════════════════════════════════════════════════
#  XML helpers — ICDAR (PAGE-2013)
# ══════════════════════════════════════════════════════════════════════════════

def _find_xml_icdar(writer, page):
    """Search all three ICDAR colour dirs for the PAGE XML matching (writer, page)."""
    patterns = [
        f'*_{writer}-IMG_MAX_{page}.xml',
        f'*_{int(writer):04d}-IMG_MAX_{page}.xml',
    ]
    for xml_dir in XML_DIRS:
        for pat in patterns:
            m = glob.glob(os.path.join(xml_dir, pat))
            if m:
                return m[0]
    return None


def _parse_points(s):
    pts = []
    for tok in s.strip().split():
        if ',' in tok:
            x, y = tok.split(',', 1)
            try:
                pts.append((float(x), float(y)))
            except ValueError:
                pass
    return pts


def _parse_xml_icdar(xml_path):
    NS = ICDAR_PAGE_NS
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    lines = []
    for tl in root.iter(f'{NS}TextLine'):
        coords = tl.find(f'{NS}Coords')
        if coords is None:
            continue
        pts_str = coords.get('points')
        if not pts_str:
            continue
        poly = _parse_points(pts_str)
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        lines.append({'id': tl.get('id', ''), 'poly': poly,
                      'bbox': (min(xs), min(ys), max(xs), max(ys))})
    return lines


def _point_in_poly(x, y, poly):
    n, inside, j = len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _bbox_dist2(x, y, bbox):
    mx, my, Mx, My = bbox
    return max(mx - x, 0, x - Mx) ** 2 + max(my - y, 0, y - My) ** 2


def _assign_patches_icdar(coords_json, textlines):
    win  = coords_json.get('win_size', 32)
    lmap = defaultdict(list)
    if not textlines:
        return {}
    for pc in coords_json['patches']:
        cx, cy = pc['x'] + win / 2, pc['y'] + win / 2
        idx    = pc['patch_idx']
        chosen = None
        for i, ln in enumerate(textlines):
            b = ln['bbox']
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                if _point_in_poly(cx, cy, ln['poly']):
                    chosen = i
                    break
        if chosen is None:
            chosen = min(range(len(textlines)),
                         key=lambda i: _bbox_dist2(cx, cy, textlines[i]['bbox']))
        lmap[chosen].append(idx)
    return dict(lmap)


def _ycluster_icdar(patches):
    if not patches:
        return {}, 0
    arr  = sorted(patches, key=lambda p: p['y'])
    ys   = np.array([p['y'] for p in arr])
    gaps = np.diff(ys)
    pos  = gaps[gaps > 5]
    thresh = (np.median(pos) if len(pos) else 30) * 0.5
    li, lmap = 0, defaultdict(list)
    lmap[0].append(arr[0]['patch_idx'])
    for i in range(1, len(arr)):
        if ys[i] - ys[i - 1] > thresh:
            li += 1
        lmap[li].append(arr[i]['patch_idx'])
    return dict(lmap), li + 1


def build_line_map_icdar(writer, page):
    coords_p = _resolve_coords_icdar(writer, page)
    if coords_p is None:
        return {}, 0
    cj    = _load_coords_json(coords_p)
    xml_p = _find_xml_icdar(writer, page)
    if xml_p is None:
        lmap, n = _ycluster_icdar(cj['patches'])
        return lmap, n
    textlines = _parse_xml_icdar(xml_p)
    if not textlines:
        return {}, 0
    lmap = _assign_patches_icdar(cj, textlines)
    return lmap, len(lmap)


# ── Dispatch line-map builder ─────────────────────────────────────────────────
build_line_map = build_line_map_cvl if DATASET == 'cvl' else build_line_map_icdar


# ══════════════════════════════════════════════════════════════════════════════
#  Thumbnail helpers
# ══════════════════════════════════════════════════════════════════════════════
import zipfile

_pages_zf         = None
_pages_zip_lookup = {}


def _init_pages_source():
    global _pages_zf, _pages_zip_lookup
    if not PAGES_DIR or not os.path.exists(PAGES_DIR):
        return
    if PAGES_DIR.endswith('.zip'):
        _pages_zf = zipfile.ZipFile(PAGES_DIR, 'r')
        for name in _pages_zf.namelist():
            bn = os.path.basename(name)
            for ext in ('.tif', '.jpg', '.png'):
                if bn.endswith(f'-cropped{ext}'):
                    stem  = bn.replace(f'-cropped{ext}', '')
                    parts = stem.split('-')
                    if len(parts) == 2:
                        try:
                            _pages_zip_lookup[(int(parts[0]), int(parts[1]))] = name
                        except ValueError:
                            pass
                elif DATASET == 'icdar' and bn.endswith(ext):
                    _pages_zip_lookup[bn] = name


def make_thumb(writer, page, dst, size):
    if os.path.exists(dst):
        return
    img_bytes = None
    w_int = int(writer) if str(writer).isdigit() else 0
    p_int = int(page)   if str(page).isdigit()   else 0

    if _pages_zf is not None:
        zname = _pages_zip_lookup.get((w_int, p_int))
        if zname:
            try:
                img_bytes = _pages_zf.read(zname)
            except Exception:
                pass
    elif PAGES_DIR and os.path.isdir(PAGES_DIR):
        for ext in ('.tif', '.jpg', '.png'):
            # CVL naming
            fp = os.path.join(PAGES_DIR, f'{w_int:04d}-{p_int}-cropped{ext}')
            if os.path.isfile(fp):
                with open(fp, 'rb') as f:
                    img_bytes = f.read()
                break
            # ICDAR naming
            for pat in [f'{writer}-IMG_MAX_{page}{ext}',
                        f'{w_int:04d}-IMG_MAX_{page}{ext}']:
                fp2 = os.path.join(PAGES_DIR, pat)
                if os.path.isfile(fp2):
                    with open(fp2, 'rb') as f:
                        img_bytes = f.read()
                    break
            if img_bytes:
                break

    if img_bytes:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img.thumbnail((size, size))
            img.save(dst)
            return
        except Exception:
            pass
    # grey placeholder
    Image.new('RGB', (size, size), (55, 58, 72)).save(dst)


_init_pages_source()


# ══════════════════════════════════════════════════════════════════════════════
#  Load artifacts
# ══════════════════════════════════════════════════════════════════════════════
print('Loading index and PCA …')
with open(INDEX_PATH, 'rb') as f:
    index = pickle.load(f)
pca = pickle.load(open(PCA_PATH, 'rb'))

wp_to_key       = {(e['writer'], e['page']): e['hdf5_key'] for e in index}
page_n_patches  = {(e['writer'], e['page']): e['patch_end'] - e['patch_start']
                   for e in index}
writer_to_pages = defaultdict(list)
for e in index:
    writer_to_pages[e['writer']].append((e['writer'], e['page']))

all_writers = list(writer_to_pages.keys())
all_wps     = list(wp_to_key.keys())
print(f'  Total pages: {len(all_wps)}   Writers: {len(all_writers)}')

# Background embeddings — load if paths given, else re-encode at level-0
has_bg_embed = bool(EMBED_PATH and os.path.isfile(EMBED_PATH)
                    and WRITER_PATH and os.path.isfile(WRITER_PATH))
if has_bg_embed:
    print('Loading pre-computed background embeddings …')
    bg_encodings  = np.load(EMBED_PATH)
    bg_writer_ids = np.load(WRITER_PATH)
    # PCA + normalise if the stored vectors are pre-PCA
    if bg_encodings.shape[1] == pca.n_features_in_:
        bg_pca = normalize(pca.transform(bg_encodings), axis=1)
    else:
        bg_pca = normalize(bg_encodings, axis=1)
    bg_page_keys = [(e['writer'], e['page']) for e in index]
    assert len(bg_pca) == len(bg_page_keys), (
        f'Background embeddings ({len(bg_pca)}) do not match index ({len(bg_page_keys)})')
    print(f'  {bg_pca.shape[0]} pages, dim={bg_pca.shape[1]} (post-PCA)')
else:
    print('No pre-computed embeddings found — will encode all pages from H5 …')
    bg_pca        = None
    bg_writer_ids = None
    bg_page_keys  = []

bg_wp_to_idx        = {wp: i for i, wp in enumerate(bg_page_keys)} if bg_page_keys else {}
bg_page_writers_arr = np.array([e['writer'] for e in index])


# ══════════════════════════════════════════════════════════════════════════════
#  Build line maps (lines mode) or validate patch counts (random mode)
# ══════════════════════════════════════════════════════════════════════════════
page_line_map  = {}
page_num_lines = {}

if MODE == 'lines':
    print('\nBuilding line maps from PAGE XML …')
    xml_found = xml_missing = coords_missing = 0
    for wp in all_wps:
        lmap, n_lines      = build_line_map(*wp)
        page_line_map[wp]  = lmap
        page_num_lines[wp] = n_lines
        if n_lines > 0 and lmap:
            xml_found += 1
        elif n_lines > 0:
            xml_missing += 1
        else:
            coords_missing += 1

    valid_wps = [wp for wp in all_wps if page_num_lines[wp] >= MIN_LINES]
    print(f'  XML/coords found: {xml_found}   fallback (Y-cluster): {xml_missing}'
          f'   missing: {coords_missing}')
    print(f'  Valid pages (≥ {MIN_LINES} lines): {len(valid_wps)} / {len(all_wps)}')
else:
    valid_wps = all_wps
    print(f'Random mode — {len(valid_wps)} pages available, no XML parsing needed.')

valid_writer_to_pages = defaultdict(list)
for wp in valid_wps:
    valid_writer_to_pages[wp[0]].append(wp)
valid_writers = list(valid_writer_to_pages.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  Focal pages
# ══════════════════════════════════════════════════════════════════════════════
def _coerce_wp(w, p):
    """Try to match (w, p) — possibly string IDs — to the index key format."""
    if (w, p) in wp_to_key:
        return (w, p)
    try:
        wi, pi = int(w), int(p)
        if (wi, pi) in wp_to_key:
            return (wi, pi)
    except (ValueError, TypeError):
        pass
    ws, ps = str(w), str(p)
    if (ws, ps) in wp_to_key:
        return (ws, ps)
    return None


if FOCAL_PAGES:
    focal_wps = []
    for pair in FOCAL_PAGES:
        wp = _coerce_wp(pair[0], pair[1])
        if wp is None:
            print(f'  WARNING: focal page {pair} not found in index — skipping')
        else:
            focal_wps.append(wp)
    if not focal_wps:
        print('FATAL: none of the specified focal pages were found.')
        sys.exit(1)
else:
    chosen    = random.sample(valid_writers, min(N_FOCAL, len(valid_writers)))
    focal_wps = [random.choice(valid_writer_to_pages[w]) for w in chosen]

print(f'\nFocal pages: {focal_wps}')


# ══════════════════════════════════════════════════════════════════════════════
#  Per-focal-page metadata: foreign writer + contamination levels
# ══════════════════════════════════════════════════════════════════════════════
focal_foreign     = {}
focal_levels      = {}
focal_level_label = {}

for wp in focal_wps:
    w = wp[0]
    if FOREIGN_WRITER is not None:
        fw = FOREIGN_WRITER
        if fw not in valid_writer_to_pages:
            # try coercing to int or str
            try:
                fw = int(fw)
            except (ValueError, TypeError):
                fw = str(fw)
        if fw not in valid_writer_to_pages:
            print(f'  WARNING: foreign_writer {FOREIGN_WRITER!r} not found in index '
                  f'— falling back to random')
            fw = random.choice([ow for ow in valid_writers if ow != w])
        else:
            print(f'  Using fixed foreign writer: {fw!r}')
    else:
        fw = random.choice([ow for ow in valid_writers if ow != w])
    fp_wp = random.choice(valid_writer_to_pages[fw])
    focal_foreign[wp] = fp_wp

    if MODE == 'lines':
        n_lines = page_num_lines.get(wp, 0)
        max_lvl = max(1, n_lines // 2)
        levels  = list(range(max_lvl + 1))
        labels  = [f'{round(100 * l / n_lines) if n_lines else 0}%' for l in levels]
        focal_levels[wp]      = levels
        focal_level_label[wp] = labels
        print(f'  ({wp[0]}, {wp[1]})  ← foreign writer {fw}  | '
              f'{n_lines} lines → levels 0..{max_lvl}')
    else:
        n_own  = page_n_patches[wp]
        n_for  = page_n_patches[fp_wp]
        levels = RANDOM_LEVELS
        labels = [f'{p}%' for p in levels]
        focal_levels[wp]      = levels
        focal_level_label[wp] = labels
        print(f'  ({wp[0]}, {wp[1]})  ← foreign writer {fw}  | '
              f'{n_own} own patches, {n_for} foreign patches → levels {levels}')


# ══════════════════════════════════════════════════════════════════════════════
#  Read + L2-normalise patch embeddings from H5
# ══════════════════════════════════════════════════════════════════════════════
def _read_norm(h5, key):
    arr   = h5[key][:].astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms < 1e-8, 1.0, norms)


def _encode(patches):
    enc = pooling_fn.encode([patches])
    return normalize(pca.transform(enc), axis=1)[0]


# ══════════════════════════════════════════════════════════════════════════════
#  If no pre-built background embeddings, encode all pages now
# ══════════════════════════════════════════════════════════════════════════════
if bg_pca is None:
    from tqdm import tqdm
    print('\nEncoding all pages as background …')
    bg_embs = []
    with h5py.File(H5_PATH, 'r') as h5:
        for wp in tqdm(all_wps, desc='encode background'):
            patches = _read_norm(h5, wp_to_key[wp])
            bg_embs.append(_encode(patches))
    bg_pca        = np.stack(bg_embs)
    bg_page_keys  = all_wps
    bg_writer_ids = np.array([wp[0] for wp in all_wps])
    bg_wp_to_idx  = {wp: i for i, wp in enumerate(all_wps)}
    print(f'  Done — {len(bg_pca)} pages')


# ══════════════════════════════════════════════════════════════════════════════
#  Generate contaminated embeddings per focal page × level
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating contaminated embeddings …')
trajectory_embs = {wp: {} for wp in focal_wps}

with h5py.File(H5_PATH, 'r') as h5:
    for wp in focal_wps:
        w, p   = wp
        levels = focal_levels[wp]
        fwp    = focal_foreign[wp]

        # Level 0 — reuse stored background embedding for exact UMAP alignment
        bg_idx = bg_wp_to_idx.get(wp)
        if bg_idx is not None:
            trajectory_embs[wp][levels[0]] = bg_pca[bg_idx]
        else:
            own = _read_norm(h5, wp_to_key[wp])
            trajectory_embs[wp][levels[0]] = _encode(own)

        if len(levels) == 1:
            continue

        own     = _read_norm(h5, wp_to_key[wp])
        n_own   = own.shape[0]
        foreign = _read_norm(h5, wp_to_key[fwp])
        n_for   = foreign.shape[0]

        if MODE == 'lines':
            lmap         = page_line_map.get(wp, {})
            f_lmap       = page_line_map.get(fwp, {})
            sorted_lines   = sorted(lmap.keys())
            f_sorted_lines = sorted(f_lmap.keys())

            for lvl in levels[1:]:
                if not lmap:
                    mixed = own
                else:
                    nr    = min(lvl, len(sorted_lines))
                    rem   = set(random.sample(sorted_lines, nr))
                    keep  = [i for li in sorted_lines if li not in rem
                               for i in lmap[li] if i < n_own]
                    nfp   = min(nr, len(f_sorted_lines))
                    fpick = (random.sample(f_sorted_lines, nfp)
                             if nfp < len(f_sorted_lines) else f_sorted_lines)
                    forn  = [i for fli in fpick for i in f_lmap[fli] if i < n_for]
                    if keep and forn:
                        mixed = np.concatenate([own[keep], foreign[forn]])
                    elif keep:
                        mixed = own[keep]
                    else:
                        mixed = own
                trajectory_embs[wp][lvl] = _encode(mixed)

        else:  # random
            for pct in levels[1:]:
                n_rep = int(round(pct * n_own / 100.0))
                n_rep = min(n_rep, n_own, n_for)
                if n_rep == 0:
                    mixed = own
                else:
                    dest  = np.random.choice(n_own, n_rep, replace=False)
                    src   = np.random.choice(n_for, n_rep, replace=False)
                    mixed = own.copy()
                    mixed[dest] = foreign[src]
                trajectory_embs[wp][pct] = _encode(mixed)

print('  Done.')


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostics — embedding drift + per-page AP
# ══════════════════════════════════════════════════════════════════════════════
print('\nEmbedding L2 drift and cosine sim from level 0:')
for wp in focal_wps:
    levels = focal_levels[wp]
    d0     = trajectory_embs[wp][levels[0]]
    if len(levels) > 1:
        info = '  '.join(
            f'L{l}=dist{np.linalg.norm(trajectory_embs[wp][l] - d0):.3f}'
            f'(cos{float(d0 @ trajectory_embs[wp][l]):.3f})'
            for l in levels[1:])
        print(f'  {wp}: {info}')


def compute_ap(query_emb, gallery_embs, gallery_writers, query_writer, exclude_idx=None):
    sims = gallery_embs @ query_emb
    if exclude_idx is not None:
        sims[exclude_idx] = -2.0
    order    = np.argsort(-sims)
    relevant = gallery_writers[order] == query_writer
    n_rel    = int(relevant.sum())
    if n_rel == 0:
        return 0.0
    hits = prec = 0.0
    for rank, is_rel in enumerate(relevant, 1):
        if is_rel:
            hits += 1
            prec += hits / rank
    return prec / n_rel


print('\nPer-page AP (contaminated query vs full gallery):')
focal_ap = {}
for wp in focal_wps:
    focal_ap[wp] = {}
    bg_idx  = bg_wp_to_idx.get(wp)
    levels  = focal_levels[wp]
    for lvl in levels:
        ap = compute_ap(trajectory_embs[wp][lvl],
                        bg_pca, bg_page_writers_arr, wp[0],
                        exclude_idx=bg_idx)
        focal_ap[wp][lvl] = round(float(ap), 4)
    ap_str = '  '.join(f'L{l}={focal_ap[wp][l]:.3f}' for l in levels)
    drop   = focal_ap[wp][levels[0]] - focal_ap[wp][levels[-1]]
    print(f'  {wp}: {ap_str}  (drop={drop:+.3f})')


"""# ══════════════════════════════════════════════════════════════════════════════
#  UMAP
# ══════════════════════════════════════════════════════════════════════════════
traj_rows  = []
focal_meta = []
for fi, wp in enumerate(focal_wps):
    w, p   = wp
    fwp    = focal_foreign[wp]
    levels = focal_levels[wp]
    labels = focal_level_label[wp]
    n_lines_or_patches = (page_num_lines.get(wp, 0) if MODE == 'lines'
                          else page_n_patches.get(wp, 0))
    for lvl, lbl in zip(levels, labels):
        traj_rows.append(trajectory_embs[wp][lvl])
        focal_meta.append({
            'focal_idx':      fi,
            'writer':         w,
            'page':           p,
            'level':          lvl,
            'label':          lbl,
            'max_level':      levels[-1],
            'n_total':        n_lines_or_patches,
            'ap':             focal_ap[wp][lvl],
            'foreign_writer': fwp[0],
            'foreign_page':   fwp[1],
        })

traj_matrix = np.stack(traj_rows)
n_bg        = len(bg_pca)
n_traj      = len(traj_matrix)

print(f'\nUMAP input: {n_bg} background + {n_traj} trajectory across {len(focal_wps)} focal pages)')
print('Running UMAP … (may take 1–5 min)')
reducer = umap.UMAP(random_state=SEED, n_neighbors=8, min_dist=0.05)
bg_2d   = reducer.fit_transform(bg_pca)
traj_2d = reducer.transform(traj_matrix)

# ── Force level-0 trajectory points to exactly match their background counterpart
for i, fm in enumerate(focal_meta):
    wp = (fm['writer'], fm['page'])
    if fm['level'] == focal_levels[wp][0]:
        bg_idx = bg_wp_to_idx.get(wp)
        if bg_idx is not None:
            traj_2d[i] = bg_2d[bg_idx]

print('  Done.')"""

# ══════════════════════════════════════════════════════════════════════════════
#  UMAP
# ══════════════════════════════════════════════════════════════════════════════
traj_rows  = []
focal_meta = []
for fi, wp in enumerate(focal_wps):
    w, p   = wp
    fwp    = focal_foreign[wp]
    levels = focal_levels[wp]
    labels = focal_level_label[wp]
    n_lines_or_patches = (page_num_lines.get(wp, 0) if MODE == 'lines'
                          else page_n_patches.get(wp, 0))
    for lvl, lbl in zip(levels, labels):
        traj_rows.append(trajectory_embs[wp][lvl])
        focal_meta.append({
            'focal_idx':      fi,
            'writer':         w,
            'page':           p,
            'level':          lvl,
            'label':          lbl,
            'max_level':      levels[-1],
            'n_total':        n_lines_or_patches,
            'ap':             focal_ap[wp][lvl],
            'foreign_writer': fwp[0],
            'foreign_page':   fwp[1],
        })

traj_matrix = np.stack(traj_rows)
all_matrix  = np.vstack([bg_pca, traj_matrix])
n_bg        = len(bg_pca)
n_traj      = len(traj_matrix)

print(f'\nUMAP input: {all_matrix.shape[0]} points '
      f'({n_bg} background + {n_traj} trajectory across {len(focal_wps)} focal pages)')
print('Running UMAP … (may take 1–5 min)')
reducer = umap.UMAP(random_state=SEED, n_neighbors=8, min_dist=0.05)
reduced = reducer.fit_transform(all_matrix)

# ── Force level-0 trajectory points to exactly match their background counterpart
for i, fm in enumerate(focal_meta):
    wp = (fm['writer'], fm['page'])
    if fm['level'] == focal_levels[wp][0]:
        bg_idx = bg_wp_to_idx.get(wp)
        if bg_idx is not None:
            reduced[n_bg + i] = reduced[bg_idx]

bg_2d   = reduced[:n_bg]
traj_2d = reduced[n_bg:]

print('  Done.')


# ══════════════════════════════════════════════════════════════════════════════
#  Thumbnails
# ══════════════════════════════════════════════════════════════════════════════
print('Generating thumbnails …')
bg_thumb_paths = []
for wp in bg_page_keys:
    w, p  = wp
    tname = f'{str(w).zfill(4)}-{p}-bg.png'
    make_thumb(w, p, os.path.join(THUMB_DIR, tname), CONFIG['thumb_size'])
    bg_thumb_paths.append(os.path.join('images', tname))

traj_thumb_paths = []
for fm in focal_meta:
    w, p  = fm['writer'], fm['page']
    tname = f'{str(w).zfill(4)}-{p}-focal-lvl{fm["level"]}.png'
    make_thumb(w, p, os.path.join(THUMB_DIR, tname), CONFIG['thumb_size'] * 2)
    traj_thumb_paths.append(os.path.join('images', tname))

if _pages_zf is not None:
    _pages_zf.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Colours
# ══════════════════════════════════════════════════════════════════════════════
unique_writers = sorted({str(w) for w in bg_writer_ids}, key=lambda x: x)
cmap_bg        = plt.get_cmap('tab20', max(len(unique_writers), 1))
writer_to_hex  = {}
for i, w in enumerate(unique_writers):
    r, g, b, _ = cmap_bg(i % 20)
    writer_to_hex[w] = '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))
bg_colors = [writer_to_hex.get(str(w), '#888888') for w in bg_writer_ids]

focal_hues = plt.get_cmap('Set1', max(len(focal_wps), 1))
cmap_traj  = plt.get_cmap('hsv')


def level_color(lvl, max_lvl):
    return '#7affdd'
    r, g, b, _ = cmap_traj(lvl / max(max_lvl, 1))
    return '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))


def focal_outline(fi):
    r, g, b, _ = focal_hues(fi)
    return '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))


traj_outlines = [focal_outline(fm['focal_idx']) for fm in focal_meta]
traj_colors   = [level_color(fm['level'], fm['max_level']) for fm in focal_meta]


# ══════════════════════════════════════════════════════════════════════════════
#  Build JSON payload
# ══════════════════════════════════════════════════════════════════════════════
bg_data = [
    {'x': float(bg_2d[i][0]), 'y': float(bg_2d[i][1]),
     'writer': str(bg_page_keys[i][0]), 'page': str(bg_page_keys[i][1]),
     'color': bg_colors[i], 'img': bg_thumb_paths[i], 'type': 'background'}
    for i in range(n_bg)
]

traj_data = [
    {'x': float(traj_2d[i][0]), 'y': float(traj_2d[i][1]),
     'writer': str(fm['writer']), 'page': str(fm['page']),
     'level': fm['level'], 'label': fm['label'],
     'max_level': fm['max_level'], 'n_total': fm['n_total'],
     'ap': fm['ap'], 'focal_idx': fm['focal_idx'],
     'foreign_writer': str(fm['foreign_writer']),
     'foreign_page':   str(fm['foreign_page']),
     'color': traj_colors[i], 'outline': traj_outlines[i],
     'img': traj_thumb_paths[i], 'type': 'trajectory'}
    for i, fm in enumerate(focal_meta)
]

focal_info = []
for fi, wp in enumerate(focal_wps):
    w, p   = wp
    fwp    = focal_foreign[wp]
    levels = focal_levels[wp]
    labels = focal_level_label[wp]
    n_tot  = page_num_lines.get(wp, 0) if MODE == 'lines' else page_n_patches.get(wp, 0)
    focal_info.append({
        'writer':         str(w),
        'page':           str(p),
        'n_total':        n_tot,
        'mode':           MODE,
        'max_level':      levels[-1],
        'foreign_writer': str(fwp[0]),
        'foreign_page':   str(fwp[1]),
        'outline':        focal_outline(fi),
        'ap_series': [
            {'level': lvl, 'label': lbl, 'ap': focal_ap[wp][lvl]}
            for lvl, lbl in zip(levels, labels)
        ],
    })

meta = {
    'dataset':    DATASET.upper(),
    'mode':       MODE,
    'mode_label': MODE_LABEL,
    'pooling':    POOLING_NAME,
    'unit':       'lines replaced' if MODE == 'lines' else 'patches replaced (%)',
}

payload = {
    'meta':        meta,
    'background':  bg_data,
    'trajectories': traj_data,
    'focal_info':  focal_info,
}

json_path = os.path.join(OUT_DIR, 'trajectory_data.json')
with open(json_path, 'w') as f:
    json.dump(payload, f, indent=2)
print(f'\nSaved JSON → {json_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  HTML — self-contained interactive UMAP
# ══════════════════════════════════════════════════════════════════════════════
json_inline = json.dumps(payload)
mode_unit   = 'lines replaced' if MODE == 'lines' else '% patches replaced'

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Contamination Trajectory — {DATASET.upper()} / {MODE_LABEL}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  :root{{
    --bg:#ffffff;
    --surface:#f4f5f9;
    --surface2:#eceef4;
    --border:#d0d4e8;
    --text:#1a1d2e;
    --muted:#7880a0;
    --accent:#4f5de8;
    --accent2:#7c5ce8;
    --green:#059669;
    --amber:#d97706;
    --red:#dc2626;
  }}
  html,body{{width:100%;height:100%;background:var(--bg);color:var(--text);
    font-family:'DM Sans',sans-serif;overflow:hidden;font-size:13px}}
  #app{{display:flex;width:100%;height:100%}}
  #sidebar{{
    width:280px;min-width:240px;flex-shrink:0;
    background:var(--surface);border-right:1px solid var(--border);
    display:flex;flex-direction:column;overflow:hidden
  }}
  #sidebar-header{{
    padding:18px 18px 14px;border-bottom:1px solid var(--border);
    background:linear-gradient(135deg,#e8eaf680,#f4f5f9)
  }}
  .ds-badge{{
    display:inline-block;font-family:'JetBrains Mono',monospace;
    font-size:9px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
    padding:3px 8px;border-radius:3px;margin-bottom:8px;
    background:#1e2236;border:1px solid var(--accent);color:var(--accent)
  }}
  .mode-badge{{
    display:inline-block;font-family:'JetBrains Mono',monospace;
    font-size:9px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
    padding:3px 8px;border-radius:3px;margin-bottom:8px;margin-left:4px;
    background:#1a1e2e;border:1px solid var(--accent2);color:var(--accent2)
  }}
  #sidebar-header h1{{font-size:15px;font-weight:500;color:var(--text);line-height:1.3;margin-top:2px}}
  #sidebar-header p{{font-size:11px;color:var(--muted);margin-top:4px}}
  .panel{{padding:12px 16px;border-bottom:1px solid var(--border)}}
  .panel-label{{
    font-size:9px;font-family:'JetBrains Mono',monospace;letter-spacing:.13em;
    text-transform:uppercase;color:var(--muted);margin-bottom:8px;display:block
  }}
  #ramp-wrap{{display:flex;align-items:center;gap:8px}}
  #ramp-gradient{{
    flex:1;height:8px;border-radius:4px;
    background:linear-gradient(to right,#0d0887,#6f01a5,#b12a90,#e16462,#fca636,#f0f921)
  }}
  .ramp-label{{font-size:10px;color:var(--muted);font-family:'JetBrains Mono',monospace}}
  #focal-scroll{{flex:1;overflow-y:auto;padding:10px 12px;display:flex;flex-direction:column;gap:8px}}
  #focal-scroll::-webkit-scrollbar{{width:4px}}
  #focal-scroll::-webkit-scrollbar-track{{background:transparent}}
  #focal-scroll::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
  .focal-card{{
    border:1px solid var(--border);border-radius:7px;padding:10px 12px;
    cursor:pointer;transition:border-color .2s,background .2s;background:var(--surface2)
  }}
  .focal-card:hover{{border-color:var(--accent);background:#181b2c}}
  .focal-card.active{{border-color:#fff;background:#1a1d2f}}
  .fc-head{{display:flex;align-items:center;gap:7px;margin-bottom:4px}}
  .fc-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
  .fc-title{{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600}}
  .fc-sub{{font-size:10px;color:var(--muted);margin-bottom:7px}}
  .ap-rows{{display:flex;flex-direction:column;gap:3px}}
  .ap-row{{display:flex;align-items:center;gap:5px}}
  .ap-row-lbl{{
    font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);
    width:30px;text-align:right;flex-shrink:0
  }}
  .ap-track{{flex:1;height:5px;background:#dde0ed;border-radius:3px;overflow:hidden}}
  .ap-fill{{height:100%;border-radius:3px;transition:width .3s}}
  .ap-val{{font-family:'JetBrains Mono',monospace;font-size:9px;width:36px;text-align:right;flex-shrink:0}}
  #canvas-wrap{{flex:1;position:relative;overflow:hidden;background:var(--bg)}}
  svg{{width:100%;height:100%}}
  #controls{{position:absolute;bottom:16px;right:16px;display:flex;gap:6px}}
  .ctrl-btn{{
    background:var(--surface);border:1px solid var(--border);border-radius:5px;
    color:var(--text);font-family:'JetBrains Mono',monospace;font-size:10px;
    padding:6px 12px;cursor:pointer;transition:border-color .15s,background .15s;letter-spacing:.05em
  }}
  .ctrl-btn:hover{{border-color:var(--accent);background:var(--surface2)}}
  #info-bar{{
    position:absolute;top:12px;left:12px;font-family:'JetBrains Mono',monospace;
    font-size:9px;color:var(--muted);letter-spacing:.06em;pointer-events:none
  }}
  #tip{{
    position:fixed;pointer-events:none;display:none;
    background:#ffffff;border:1px solid var(--border);border-radius:8px;
    padding:10px 12px;font-size:11px;box-shadow:0 12px 40px #000c;z-index:999;max-width:210px
  }}
  #tip img{{width:100%;border-radius:4px;margin-bottom:7px;display:block;border:1px solid var(--border)}}
  .tt-title{{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;color:var(--text);line-height:1.5}}
  .tt-sub{{font-size:10px;color:var(--muted);margin-top:3px;line-height:1.6}}
  .tt-ap{{margin-top:6px;padding:6px 8px;background:#0d0f18;border-radius:5px;font-family:'JetBrains Mono',monospace;font-size:10px}}
  .tt-ap-bar{{height:4px;background:#1a1d2e;border-radius:2px;margin-top:4px;overflow:hidden}}
  .tt-ap-fill{{height:100%;border-radius:2px}}
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <div id="sidebar-header">
      <span class="ds-badge" id="ds-badge">DATASET</span>
      <span class="mode-badge" id="mode-badge">MODE</span>
      <h1>Contamination<br>Trajectory</h1>
      <p id="hdr-sub">UMAP embedding space</p>
    </div>
    <div class="panel">
      <span class="panel-label">Contamination level (colour)</span>
      <div id="ramp-wrap">
        <span class="ramp-label" id="ramp-lo">0</span>
        <div id="ramp-gradient"></div>
        <span class="ramp-label" id="ramp-hi">max</span>
      </div>
      <div style="font-size:10px;color:var(--muted);margin-top:6px" id="unit-note"></div>
    </div>
    <div class="panel">
      <span class="panel-label">Focal pages — click to isolate</span>
    </div>
    <div id="focal-scroll"></div>
  </div>
  <div id="canvas-wrap">
    <svg id="umap-svg"></svg>
    <div id="info-bar">drag · scroll to zoom · click focal card to isolate</div>
    <div id="controls">
      <button class="ctrl-btn" id="btn-all">Show all</button>
      <button class="ctrl-btn" id="btn-reset">Reset view</button>
    </div>
  </div>
</div>
<div id="tip"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const DATA = {json_inline};
const {{meta, background:bg, trajectories:traj, focal_info:focalInfo}} = DATA;

document.getElementById('ds-badge').textContent   = meta.dataset;
document.getElementById('mode-badge').textContent = meta.mode_label;
document.getElementById('hdr-sub').textContent    = `${{meta.pooling}} · UMAP embedding space`;
document.getElementById('unit-note').textContent  =
  `Trajectory colour = ${{meta.mode === 'lines' ? 'n lines replaced' : '% patches replaced'}}`;
document.getElementById('ramp-hi').textContent = meta.mode === 'lines' ? 'max ½' : '50%';
document.getElementById('ramp-lo').textContent = meta.mode === 'lines' ? '0' : '0%';

const focusOutline = {{}};
traj.forEach(d => focusOutline[d.focal_idx] = d.outline);

function apColor(ap, baseline) {{
  if (!baseline) return 'var(--accent)';
  const ratio = ap / baseline;
  if (ratio >= 0.9) return 'var(--green)';
  if (ratio >= 0.6) return 'var(--amber)';
  return 'var(--red)';
}}

const focalListEl = document.getElementById('focal-scroll');
let activeFilter  = null;

focalInfo.forEach((info, fi) => {{
  const col      = focusOutline[fi] || '#fff';
  const baseline = info.ap_series[0]?.ap || 0;
  const subLabel = meta.mode === 'lines'
    ? `${{info.n_total}} lines · levels 0–${{info.max_level}} (≤50%)`
    : `${{info.n_total}} patches · levels ${{info.ap_series.map(s=>s.label).join(' ')}}`;

  const apRowsHTML = info.ap_series.map(s => {{
    const w   = Math.round(s.ap * 100);
    const col = apColor(s.ap, baseline);
    return `<div class="ap-row">
      <span class="ap-row-lbl">${{s.label}}</span>
      <div class="ap-track"><div class="ap-fill" style="width:${{w}}%;background:${{col}}"></div></div>
      <span class="ap-val" style="color:${{col}}">${{s.ap.toFixed(3)}}</span>
    </div>`;
  }}).join('');

  const card = document.createElement('div');
  card.className  = 'focal-card';
  card.dataset.fi = fi;
  card.innerHTML  = `
    <div class="fc-head">
      <div class="fc-dot" style="background:${{col}}"></div>
      <span class="fc-title">W${{info.writer}} · p${{info.page}}</span>
    </div>
    <div class="fc-sub">${{subLabel}}<br>foreign → w${{info.foreign_writer}}</div>
    <div class="ap-rows">${{apRowsHTML}}</div>`;

  card.addEventListener('click', () => {{
    if (activeFilter === fi) {{
      activeFilter = null;
      card.classList.remove('active');
    }} else {{
      document.querySelectorAll('.focal-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      activeFilter = fi;
    }}
    updateVis();
  }});
  focalListEl.appendChild(card);
}});

const svg  = d3.select('#umap-svg');
const wrap = document.getElementById('canvas-wrap');
const W = () => wrap.clientWidth, H = () => wrap.clientHeight;
const g   = svg.append('g');

svg.append('defs').append('marker')
  .attr('id', 'arr').attr('viewBox', '0 -5 10 10')
  .attr('refX', 8).attr('refY', 0)
  .attr('markerWidth', 5).attr('markerHeight', 5)
  .attr('orient', 'auto')
  .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#00000066');

const allX = [...bg, ...traj].map(d => d.x);
const allY = [...bg, ...traj].map(d => d.y);
const pad  = 0.05;
const xE   = d3.extent(allX), yE = d3.extent(allY);
const xR   = xE[1]-xE[0],    yR = yE[1]-yE[0];
const xScl = d3.scaleLinear().domain([xE[0]-pad*xR, xE[1]+pad*xR]).range([0, W()]);
const yScl = d3.scaleLinear().domain([yE[0]-pad*yR, yE[1]+pad*yR]).range([H(), 0]);

const SZ = {{ bgR:2.5, trajR0:9, trajR:7, sw0:2.5, sw:1.8, arrowW:1.5, fs:9 }};
let curK = 1;

function rescale(k) {{
  curK = k;
  const s = 1/k;
  bgG.selectAll('circle').attr('r', SZ.bgR * s);
  trajG.selectAll('.tpt')
    .attr('r',            d => (d.level === focalInfo[d.focal_idx].ap_series[0].level ? SZ.trajR0 : SZ.trajR) * s)
    .attr('stroke-width', d => (d.level === focalInfo[d.focal_idx].ap_series[0].level ? SZ.sw0 : SZ.sw) * s);
  arrG.selectAll('.tarr').attr('stroke-width', SZ.arrowW * s);
  trajG.selectAll('.tlbl').attr('font-size', SZ.fs * s).attr('dy', 3.5 * s);
}}

const zoom = d3.zoom().scaleExtent([0.25, 60]).on('zoom', e => {{
  g.attr('transform', e.transform);
  rescale(e.transform.k);
}});
svg.call(zoom);
document.getElementById('btn-reset').onclick = () =>
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity)
    .on('end', () => rescale(1));
document.getElementById('btn-all').onclick = () => {{
  activeFilter = null;
  document.querySelectorAll('.focal-card').forEach(c => c.classList.remove('active'));
  updateVis();
}};

const bgG = g.append('g');
bgG.selectAll('circle').data(bg).join('circle')
  .attr('cx', d => xScl(d.x)).attr('cy', d => yScl(d.y))
  .attr('r', SZ.bgR).attr('fill', d => d.color).attr('opacity', 0.32)
  .on('mouseover', showTip).on('mousemove', moveTip).on('mouseout', hideTip);

const arrG  = g.append('g');
const trajG = g.append('g');
const byFocal = d3.group(traj, d => d.focal_idx);

byFocal.forEach((pts, fi) => {{
  pts.sort((a, b) => a.level - b.level);
  for (let i = 0; i < pts.length - 1; i++) {{
    arrG.append('line').attr('class', 'tarr').attr('data-fi', fi)
      .attr('x1', xScl(pts[i].x)).attr('y1', yScl(pts[i].y))
      .attr('x2', xScl(pts[i+1].x)).attr('y2', yScl(pts[i+1].y))
      .attr('stroke', focusOutline[fi] || '#fff')
      .attr('stroke-width', SZ.arrowW).attr('stroke-opacity', 0.55)
      .attr('marker-end', 'url(#arr)');
  }}
  trajG.selectAll(null).data(pts).join('circle')
    .attr('class', 'tpt').attr('data-fi', fi)
    .attr('cx', d => xScl(d.x)).attr('cy', d => yScl(d.y))
    .attr('r',            d => (d.level === focalInfo[d.focal_idx].ap_series[0].level ? SZ.trajR0 : SZ.trajR))
    .attr('fill',         d => d.color)
    .attr('stroke', 'none')
    .attr('stroke-width', 0)
    .attr('opacity', 0.92)
    .on('mouseover', showTip).on('mousemove', moveTip).on('mouseout', hideTip);

  trajG.selectAll(null).data(pts).join('text')
    .attr('class', 'tlbl').attr('data-fi', fi)
    .attr('x', d => xScl(d.x)).attr('dy', 3.5)
    .attr('y', d => yScl(d.y)).attr('text-anchor', 'middle')
    .attr('fill', '#111').attr('font-size', SZ.fs)
    .attr('font-family', 'JetBrains Mono, monospace').attr('font-weight', '700')
    .attr('pointer-events', 'none')
    .text(d => d.label);
}});

function updateVis() {{
  const fi = activeFilter;
  if (fi === null) {{
    bgG.selectAll('circle').attr('opacity', 0.32);
    arrG.selectAll('.tarr').attr('stroke-opacity', 0.55);
    trajG.selectAll('.tpt').attr('opacity', 0.92);
    trajG.selectAll('.tlbl').attr('opacity', 1);
  }} else {{
    bgG.selectAll('circle').attr('opacity', 0.08);
    arrG.selectAll('.tarr')
      .attr('stroke-opacity', function() {{ return +this.dataset.fi === fi ? 0.85 : 0.03; }});
    trajG.selectAll('.tpt')
      .attr('opacity', function() {{ return +this.dataset.fi === fi ? 0.95 : 0.05; }});
    trajG.selectAll('.tlbl')
      .attr('opacity', function() {{ return +this.dataset.fi === fi ? 1 : 0; }});
  }}
}}

const tip = document.getElementById('tip');

function showTip(event, d) {{
  let html = d.img ? `<img src="${{d.img}}" onerror="this.style.display='none'">` : '';
  if (d.type === 'background') {{
    html += `<div class="tt-title">Writer ${{d.writer}}</div>
      <div class="tt-sub">Page ${{d.page}}</div>`;
  }} else {{
    const fi       = d.focal_idx;
    const baseline = focalInfo[fi].ap_series[0]?.ap || 0;
    const apCol    = apColor(d.ap, baseline);
    const isClean  = d.level === focalInfo[fi].ap_series[0]?.level;
    html += `<div class="tt-title">Writer ${{d.writer}} · Page ${{d.page}}</div>
      <div class="tt-sub">
        ${{isClean ? `Level ${{d.label}} — original (clean)`
                   : `Level ${{d.label}} — ${{meta.mode === 'lines'
                       ? d.label + ' lines replaced'
                       : d.label + ' patches replaced'}}`}}
        ${{!isClean ? `<br>foreign: w${{d.foreign_writer}} p${{d.foreign_page}}` : ''}}
      </div>
      <div class="tt-ap">
        <span style="color:var(--muted)">AP </span>
        <span style="color:${{apCol}};font-weight:700">${{d.ap.toFixed(4)}}</span>
        ${{!isClean ? `<span style="color:var(--muted);font-size:9px">
          (${{d.ap > baseline ? '+' : ''}}${{(d.ap - baseline).toFixed(4)}} vs clean)</span>` : ''}}
        <div class="tt-ap-bar">
          <div class="tt-ap-fill" style="width:${{Math.round(d.ap*100)}}%;background:${{apCol}}"></div>
        </div>
      </div>`;
  }}
  tip.innerHTML = html;
  tip.style.display = 'block';
  moveTip(event);
}}
function moveTip(event) {{
  const x = event.clientX + 14, y = event.clientY + 14;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = (x + tw > window.innerWidth  ? x - tw - 28 : x) + 'px';
  tip.style.top  = (y + th > window.innerHeight ? y - th - 28 : y) + 'px';
}}
function hideTip() {{ tip.style.display = 'none'; }}
</script>
</body>
</html>"""

html_path = os.path.join(OUT_DIR, 'contamination_trajectory.html')
with open(html_path, 'w') as f:
    f.write(HTML)

print(f'\n{"─"*60}')
print(f'  ✓  HTML   → {html_path}')
print(f'  ✓  JSON   → {json_path}')
print(f'  ✓  Thumbs → {THUMB_DIR}/')
print(f'{"─"*60}')
print(f'\n  Dataset  : {DATASET.upper()}')
print(f'  Mode     : {MODE_LABEL}')
print(f'  Pages    : {n_bg} background + {n_traj} trajectory points')
print(f'  Focal    : {len(focal_wps)} pages × {[len(focal_levels[wp]) for wp in focal_wps]} levels')
print(f'\n  Download the whole  {OUT_DIR}/  folder and open')
print(f'  contamination_trajectory.html  locally  (file:// — no server needed).')