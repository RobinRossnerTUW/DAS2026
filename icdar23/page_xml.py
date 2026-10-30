"""
page_xml.py
───────────
Dataset-specific PAGE-XML parsing and patch-to-textline assignment.
Supports CVL (PAGE-2010) and ICDAR HistoricalWI-2017 (PAGE-2013).

Public API
──────────
build_all_line_maps(all_wps)
    Iterate over every (writer, page) pair, parse its PAGE XML (or fall back
    to y-cluster grouping if the XML is missing), and return:
        page_line_map  : dict  { wp -> {line_idx: [patch_idx, ...]} }
        page_num_lines : dict  { wp -> int }
        stats          : dict  with xml_found/xml_missing/coords_missing/total_outside
"""

import os, re, json, glob
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import config

# ══════════════════════════════════════════════════════════════════════════════
# CVL helpers  (PAGE-2010, dkTextLines / AttrRegion schema)
# ══════════════════════════════════════════════════════════════════════════════

def _find_xml_cvl(writer_id, text_id):
    xml_dir  = config.XML_DIRS[0]
    patterns = [
        f'{writer_id}-{text_id}.xml',
        f'{writer_id}-{text_id}-cropped.xml',
        f'{int(writer_id):04d}-{int(text_id)}.xml',
        f'{int(writer_id):04d}-{text_id}.xml',
        f'{int(writer_id):04d}-{int(text_id)}_attributes.xml',
    ]
    for pat in patterns:
        path = os.path.join(xml_dir, pat)
        if os.path.isfile(path):
            return path
    for root_dir, _, files in os.walk(xml_dir):
        for fname in files:
            if fname.endswith('.xml') and str(writer_id) in fname and str(text_id) in fname:
                return os.path.join(root_dir, fname)
    return None


def _parse_cvl_xml(xml_path):
    """Return (crop_offset_y, [(y_min, y_max), ...]) for all textlines."""
    with open(xml_path, 'rb') as f:
        raw = f.read()
    if   raw[:2] == b'\xff\xfe': text = raw.decode('utf-16-le',  errors='replace')
    elif raw[:2] == b'\xfe\xff': text = raw.decode('utf-16-be',  errors='replace')
    else:
        try:    text = raw.decode('utf-8',    errors='strict')
        except UnicodeDecodeError:
                text = raw.decode('latin-1',  errors='replace')

    text = re.sub(r'<\?xml[^?]*\?>', '', text, count=1)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]', '', text)
    try:
        root = ET.fromstring(text.encode('utf-8'))
    except ET.ParseError:
        return 0, []

    ns            = config.PAGE_NS
    crop_offset_y = 0
    for tl in root.iter(f'{ns}dkTextLines'):
        cy = tl.get('cropY')
        if cy is not None:
            crop_offset_y = int(float(cy))
            break

    textlines = []
    for region in root.iter(f'{ns}AttrRegion'):
        if region.get('attrType') == '2' and region.get('fontType') in ('2', None):
            ys = [float(pt.get('y')) for pt in region.iter(f'{ns}Point')
                  if pt.get('y') is not None]
            if ys:
                textlines.append((min(ys), max(ys)))
    textlines.sort(key=lambda t: t[0])
    return crop_offset_y, textlines


def _assign_patches_cvl(coords_json, textlines, crop_offset_y):
    line_to_patches = defaultdict(list)
    n_outside       = 0
    for patch in coords_json['patches']:
        py_full  = patch['y'] + crop_offset_y
        assigned = False
        for li, (y_min, y_max) in enumerate(textlines):
            if y_min <= py_full <= y_max:
                line_to_patches[li].append(patch['patch_idx'])
                assigned = True
                break
        if not assigned:
            if textlines:
                dists = [min(abs(py_full - y0), abs(py_full - y1))
                         for y0, y1 in textlines]
                line_to_patches[int(np.argmin(dists))].append(patch['patch_idx'])
            n_outside += 1
    return dict(line_to_patches), n_outside


def _ycluster_cvl(patches):
    """Fallback: group patches into pseudo-lines by y-coordinate gaps."""
    if not patches:
        return {}, 0
    arr  = sorted(patches, key=lambda pc: pc['y'])
    ys   = np.array([pc['y'] for pc in arr])
    gaps = np.diff(ys)
    pos  = gaps[gaps > 5]
    thr  = (np.median(pos) if len(pos) else 30) * 0.5
    li   = 0
    lm   = defaultdict(list)
    lm[0].append(arr[0]['patch_idx'])
    for i in range(1, len(arr)):
        if ys[i] - ys[i - 1] > thr:
            li += 1
        lm[li].append(arr[i]['patch_idx'])
    return dict(lm), li + 1


def _resolve_coords_cvl(writer, page):
    for stem in [f'{int(writer):04d}-{int(page)}-cropped',
                 f'{writer}-{page}-cropped',
                 f'{int(writer):04d}-{int(page)}',
                 f'{writer}-{page}']:
        path = os.path.join(config.PATCHES_DIR, stem, 'coordinates.json')
        if os.path.isfile(path):
            return path
    return None


def _build_line_map_cvl(w, p, diag=False):
    coords_path = _resolve_coords_cvl(w, p)
    if coords_path is None:
        return {}, 0, 0, 'coords_missing'

    with open(coords_path) as f:
        cj = json.load(f)

    xml_path = _find_xml_cvl(f'{int(w):04d}', str(int(p)))
    if xml_path is None:
        lm, n = _ycluster_cvl(cj['patches'])
        return lm, n, 0, 'xml_missing'

    crop_y, textlines = _parse_cvl_xml(xml_path)
    if diag:
        _print_diag(w, p, coords_path, xml_path, len(textlines), cj, crop_y=crop_y)

    lm, n_out = _assign_patches_cvl(cj, textlines, crop_y)
    n_lines   = len(lm) if lm else len(textlines)
    return lm, n_lines, n_out, 'found'


# ══════════════════════════════════════════════════════════════════════════════
# ICDAR helpers  (PAGE-2013, Transkribus TextLine polygon schema)
# ══════════════════════════════════════════════════════════════════════════════

def _find_xml_icdar(writer, page):
    for xml_dir in config.XML_DIRS:
        for pat in [f'*_{writer}-IMG_MAX_{page}.xml',
                    f'*_{int(writer):04d}-IMG_MAX_{page}.xml']:
            matches = glob.glob(os.path.join(xml_dir, pat))
            if matches:
                return matches[0]
    return None


def _parse_points(s):
    pts = []
    for tok in s.strip().split():
        if ',' in tok:
            x, y = tok.split(',', 1)
            try:    pts.append((float(x), float(y)))
            except ValueError: pass
    return pts


def _parse_icdar_xml(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    ns    = config.PAGE_NS
    lines = []
    for tl in root.iter(f'{ns}TextLine'):
        coords = tl.find(f'{ns}Coords')
        if coords is None: continue
        pts_str = coords.get('points')
        if not pts_str: continue
        poly = _parse_points(pts_str)
        if len(poly) < 3: continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        lines.append({'id': tl.get('id', ''), 'poly': poly,
                      'bbox': (min(xs), min(ys), max(xs), max(ys))})
    return lines


def _point_in_polygon(x, y, poly):
    inside = False
    j      = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _bbox_dist(x, y, bbox):
    minx, miny, maxx, maxy = bbox
    return max(minx - x, 0, x - maxx) ** 2 + max(miny - y, 0, y - maxy) ** 2


def _assign_patches_icdar(coords_json, textlines):
    win             = coords_json.get('win_size', 32)
    line_to_patches = defaultdict(list)
    n_outside       = 0
    if not textlines:
        return {}, len(coords_json['patches'])
    for patch in coords_json['patches']:
        cx  = patch['x'] + win / 2.0
        cy  = patch['y'] + win / 2.0
        idx = patch['patch_idx']
        cands = [i for i, ln in enumerate(textlines)
                 if ln['bbox'][0] <= cx <= ln['bbox'][2]
                 and ln['bbox'][1] <= cy <= ln['bbox'][3]]
        chosen = next((i for i in cands
                       if _point_in_polygon(cx, cy, textlines[i]['poly'])), None)
        if chosen is None:
            chosen    = min(range(len(textlines)),
                            key=lambda i: _bbox_dist(cx, cy, textlines[i]['bbox']))
            n_outside += 1
        line_to_patches[chosen].append(idx)
    return dict(line_to_patches), n_outside


def _ycluster_icdar(patches):
    if not patches:
        return {}, 0
    arr = np.array([(pc['y'], pc['patch_idx']) for pc in patches],
                   dtype=[('y', float), ('idx', int)])
    arr.sort(order='y')
    ys, idx = arr['y'], arr['idx']
    gaps = np.diff(ys); pos = gaps[gaps > 5]
    thr  = (np.median(pos) if len(pos) else 30) * 0.5
    li   = 0; lm = defaultdict(list)
    lm[0].append(int(idx[0]))
    for i in range(1, len(idx)):
        if ys[i] - ys[i - 1] > thr: li += 1
        lm[li].append(int(idx[i]))
    return dict(lm), li + 1


def _resolve_coords_icdar(writer, page):
    for stem in [f'{writer}-IMG_MAX_{page}',
                 f'{int(writer):04d}-IMG_MAX_{page}',
                 f'{writer}-{page}',
                 str(page)]:
        path = os.path.join(config.PATCHES_DIR, stem, 'coordinates.json')
        if os.path.isfile(path):
            return path
    return None


def _build_line_map_icdar(w, p, diag=False):
    coords_path = _resolve_coords_icdar(w, p)
    if coords_path is None:
        return {}, 0, 0, 'coords_missing'

    with open(coords_path) as f:
        cj = json.load(f)

    xml_path = _find_xml_icdar(w, p)
    if xml_path is None:
        lm, n = _ycluster_icdar(cj['patches'])
        return lm, n, 0, 'xml_missing'

    textlines = _parse_icdar_xml(xml_path)
    if diag:
        _print_diag(w, p, coords_path, xml_path, len(textlines), cj)

    lm, n_out = _assign_patches_icdar(cj, textlines)
    return lm, len(lm), n_out, 'found'


# ══════════════════════════════════════════════════════════════════════════════
# Shared diagnostic printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_diag(w, p, coords_path, xml_path, n_lines, cj, crop_y=None):
    print(f'\n--- diagnostic ({w}, {p}) ---')
    print(f'  coords: {coords_path}')
    print(f'  xml:    {xml_path}')
    if crop_y is not None:
        print(f'  crop_offset_y: {crop_y}')
    print(f'  num_patches:   {cj.get("num_patches")}')
    print(f'  n textlines:   {n_lines}')


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch + public API
# ══════════════════════════════════════════════════════════════════════════════

_BUILD_LINE_MAP = _build_line_map_cvl if config.DATASET == 'cvl' else _build_line_map_icdar


def build_all_line_maps(all_wps):
    """
    Parse PAGE XML for every (writer, page) pair and assign patches to lines.

    Returns
    ───────
    page_line_map  : dict { wp -> {line_idx: [patch_idx, ...]} }
    page_num_lines : dict { wp -> int }
    stats          : dict {
        'xml_found', 'xml_missing', 'coords_missing', 'total_outside'
    }
    """
    page_line_map  = {}
    page_num_lines = {}
    xml_found = xml_missing = coords_missing = total_outside = 0
    diag_done = False

    for wp in tqdm(all_wps, desc='build line map'):
        w, p = wp
        lm, n_lines, n_out, status = _BUILD_LINE_MAP(w, p, diag=not diag_done)
        page_line_map[wp]  = lm
        page_num_lines[wp] = n_lines
        total_outside += n_out
        if status == 'found':
            xml_found += 1; diag_done = True
        elif status == 'xml_missing':
            xml_missing += 1
        else:
            coords_missing += 1

    stats = dict(xml_found=xml_found, xml_missing=xml_missing,
                 coords_missing=coords_missing, total_outside=total_outside)
    return page_line_map, page_num_lines, stats
