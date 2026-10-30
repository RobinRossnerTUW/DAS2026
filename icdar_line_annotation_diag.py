"""
icdar_line_annotation_diagnostic.py

For N_SAMPLE writers from the ICDAR dataset, prints a per-line breakdown
(line id, bbox, patch count) and reports the image path so you can open
it alongside the annotation.

Usage:
    python icdar_line_annotation_diagnostic.py
    python icdar_line_annotation_diagnostic.py --n 20 --seed 7
"""

import os, sys, json, glob, pickle, random, argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

# ── config ───────────────────────────────────────────────────────────────────
EXPERIMENT_DIR    = '/data/rrossner/DAS2026/icdar23/experiments/test_repository-2026-05-08-12-23'
PATCHES_DIR       = '/data/rrossner/DAS2026/icdar/test_patches_coords'
XML_DIRS          = [
    '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-1/page',
    '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-2/page',
    '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-3/page',
]
IMAGE_DIR         = '/data/rrossner/datasets/ICDAR17/binarized_test/ScriptNet-HistoricalWI-2017-binarized'
PAGE_NS           = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}'
INDEX_PATH        = os.path.join(EXPERIMENT_DIR, 'patch_index_test.pkl')

N_SAMPLE = 10
SEED     = 0

_p = argparse.ArgumentParser()
_p.add_argument('--n',    type=int, default=N_SAMPLE)
_p.add_argument('--seed', type=int, default=SEED)
args = _p.parse_args()
N_SAMPLE, SEED = args.n, args.seed
random.seed(SEED)

# ── helpers (copied verbatim from main script) ───────────────────────────────

def _find_xml_for_page_icdar(writer, page):          # drop xml_dir parameter
    patterns = [f'*_{writer}-IMG_MAX_{page}.xml',
                f'*_{int(writer):04d}-IMG_MAX_{page}.xml']
    for xml_dir in XML_DIRS:                          # loop over all three
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


def _parse_icdar_xml(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    lines = []
    for tl in root.iter(f'{PAGE_NS}TextLine'):
        coords = tl.find(f'{PAGE_NS}Coords')
        if coords is None:
            continue
        pts = _parse_points(coords.get('points', ''))
        if len(pts) < 3:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        lines.append({'id': tl.get('id', ''), 'poly': pts,
                      'bbox': (min(xs), min(ys), max(xs), max(ys))})
    return lines


def _point_in_polygon(x, y, poly):
    n, inside, j = len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _bbox_dist(x, y, bbox):
    mnx, mny, mxx, mxy = bbox
    return max(mnx - x, 0, x - mxx) ** 2 + max(mny - y, 0, y - mxy) ** 2


def _assign_patches_icdar(coords_json, textlines, win_size_default=32):
    win = coords_json.get('win_size', win_size_default)
    patches = coords_json['patches']
    line_to_patches = defaultdict(list)
    n_outside = 0
    if not textlines:
        return {}, len(patches)
    for patch in patches:
        cx = patch['x'] + win / 2.0
        cy = patch['y'] + win / 2.0
        idx = patch['patch_idx']
        cands = [i for i, ln in enumerate(textlines)
                 if ln['bbox'][0] <= cx <= ln['bbox'][2]
                 and ln['bbox'][1] <= cy <= ln['bbox'][3]]
        chosen = None
        for i in cands:
            if _point_in_polygon(cx, cy, textlines[i]['poly']):
                chosen = i; break
        if chosen is None:
            chosen = min(range(len(textlines)),
                         key=lambda i: _bbox_dist(cx, cy, textlines[i]['bbox']))
            n_outside += 1
        line_to_patches[chosen].append(idx)
    return dict(line_to_patches), n_outside


def _resolve_coords_path(patches_dir, writer, page):
    for stem in [f'{writer}-IMG_MAX_{page}',
                 f'{int(writer):04d}-IMG_MAX_{page}',
                 f'{writer}-{page}', str(page)]:
        path = os.path.join(patches_dir, stem, 'coordinates.json')
        if os.path.isfile(path):
            return path
    return None


def find_image(writer, page):
    for stem in [f'{writer}-IMG_MAX_{page}',
                 f'{int(writer):04d}-IMG_MAX_{page}']:
        for ext in ('.jpg', '.png', '.tif'):
            p = os.path.join(IMAGE_DIR, stem + ext)
            if os.path.isfile(p):
                return p
    return None


# ── load index and sample writers ────────────────────────────────────────────

print(f'Loading index from {INDEX_PATH} …')
with open(INDEX_PATH, 'rb') as f:
    index = pickle.load(f)

writer_to_pages = defaultdict(list)
page_num_patches = {}
for e in index:
    wp = (e['writer'], e['page'])
    writer_to_pages[e['writer']].append(wp)
    page_num_patches[wp] = e['patch_end'] - e['patch_start']

all_writers = list(writer_to_pages.keys())
sampled_writers = random.sample(all_writers, min(N_SAMPLE, len(all_writers)))
print(f'Sampled {len(sampled_writers)} / {len(all_writers)} writers  (seed={SEED})\n')

SEP = '─' * 90

# ── per-writer diagnostic ─────────────────────────────────────────────────────

issues = []   # collect pages with problems for a summary at the end

for writer in sampled_writers:
    # pick the page with the most patches (usually most informative)
    pages = writer_to_pages[writer]
    wp    = max(pages, key=lambda w: page_num_patches[w])
    w, p  = wp

    print(SEP)
    print(f'WRITER {w}   PAGE {p}   ({len(pages)} page(s) in index)')

    # ── coordinates.json ────────────────────────────────────────────────────
    coords_path = _resolve_coords_path(PATCHES_DIR, w, p)
    if coords_path is None:
        print(f'  [ERROR] coordinates.json not found under {PATCHES_DIR}')
        issues.append((w, p, 'coords_missing'))
        continue
    with open(coords_path) as f:
        cj = json.load(f)
    n_patches   = len(cj['patches'])
    win_size    = cj.get('win_size', 32)
    img_shape   = cj.get('image_shape', 'unknown')
    print(f'  coords:      {coords_path}')
    print(f'  patches:     {n_patches}  (index says {page_num_patches[wp]})'
          + ('' if n_patches == page_num_patches[wp] else '  ← MISMATCH'))
    print(f'  win_size:    {win_size}   image_shape (H×W): {img_shape}')

    # ── XML ─────────────────────────────────────────────────────────────────
    xml_path = _find_xml_for_page_icdar(w, p)

    if xml_path is None:
        print(f'  [WARN] XML not found in {XML_DIR} — cannot show line annotations')
        issues.append((w, p, 'xml_missing'))
        textlines = []
    else:
        print(f'  xml:         {xml_path}')
        textlines = _parse_icdar_xml(xml_path)

    # ── assignment ──────────────────────────────────────────────────────────
    if textlines:
        line_map, n_outside = _assign_patches_icdar(cj, textlines)
        print(f'  textlines:   {len(textlines)}   '
              f'lines with ≥1 patch: {len(line_map)}   '
              f'patches assigned to nearest (outside poly): {n_outside}')

        # per-line table
        print(f'\n  {"Line":>4}  {"ID":>30}  {"bbox (x0,y0,x1,y1)":>38}  {"#patches":>8}')
        print(f'  {"─"*4}  {"─"*30}  {"─"*38}  {"─"*8}')
        for li, ln in enumerate(textlines):
            bx = '({:5.0f},{:5.0f},{:5.0f},{:5.0f})'.format(*ln['bbox'])
            n_p = len(line_map.get(li, []))
            flag = '  ← 0 patches' if n_p == 0 else ''
            print(f'  {li:>4}  {ln["id"]:>30}  {bx:>38}  {n_p:>8}{flag}')

        empty_lines = sum(1 for li in range(len(textlines)) if li not in line_map)
        if empty_lines:
            print(f'\n  [WARN] {empty_lines} / {len(textlines)} lines have 0 patches assigned')
            issues.append((w, p, f'{empty_lines}_empty_lines'))
    else:
        print('  (no textlines parsed — skipping assignment table)')

    # ── image path ──────────────────────────────────────────────────────────
    img_path = find_image(w, p)
    if img_path:
        print(f'\n  image:       {img_path}')
    else:
        print(f'\n  [WARN] image not found — searched: {IMAGE_DIR}')
        print( '          Update IMAGE_ROOTS at the top of this script if needed.')
        issues.append((w, p, 'image_missing'))

    print()

# ── summary ──────────────────────────────────────────────────────────────────
print(SEP)
if issues:
    print(f'\nISSUES SUMMARY ({len(issues)} pages):\n')
    for w, p, kind in issues:
        print(f'  writer={w}  page={p}  → {kind}')
else:
    print('\nNo issues found across sampled pages.')