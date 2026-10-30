"""
make_contamination_viz.py
─────────────────────────
Produces five images directly from the GPFS zips/tar — no job needed.

    python make_contamination_viz.py
"""

import io, json, os, pickle, random, re, tarfile, time, zipfile, xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

# ── Paths (all on GPFS, no local scratch needed) ──────────────────────────────
TAR_PATH   = '/gpfs/data/fs71186/rossner/datasets/cvl-database/patches/cvl_test_patches_7663856.tar'
CROPPED_ZIP= '/gpfs/data/fs71186/rossner/datasets/cvl-database/cvl-database-cropped-1-1.zip'
XML_ZIP    = '/gpfs/data/fs71186/rossner/datasets/cvl-database/cvl-database-1-1.zip'
INDEX_PATH = '/gpfs/data/fs71186/rossner/datasets/cvl-database/experiments/7663856/experiments/cvl_baseline-2026-04-02-22-05/patch_index_test.pkl'
OUT_DIR    = '/gpfs/data/fs71186/rossner/icdar23/comparison'

ORIG_WRITER, ORIG_PAGE   = 395, 4    # change if you want a different page
FOREIGN_WRITER           = None      # None = auto-pick
LINE_TO_REPLACE          = 2         # 0-indexed line index to contaminate
SEED                     = 42

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ── XML parsing (same namespace logic as contamination_eval_cvl.py) ───────────
def parse_cvl_xml(raw_bytes):
    NS = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}'
    raw = re.sub(rb'<\?xml[^?]*\?>', b'', raw_bytes, count=1)
    raw = re.sub(rb'[^\x09\x0A\x0D\x20-\xFF]', b'', raw)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return 0, []

    crop_offset_y = 0
    for tl in root.iter(f'{NS}dkTextLines'):
        cy = tl.get('cropY')
        if cy is not None:
            crop_offset_y = int(float(cy)); break

    textlines = []
    for region in root.iter(f'{NS}AttrRegion'):
        if region.get('attrType') == '2':
            ys = [float(pt.get('y')) for pt in region.iter(f'{NS}Point') if pt.get('y')]
            if ys:
                textlines.append((min(ys), max(ys)))

    textlines.sort(key=lambda t: t[0])
    return crop_offset_y, textlines


def assign_patches_to_lines(patches, textlines, crop_offset_y):
    line_map = defaultdict(list)
    for p in patches:
        y_full = p['y'] + crop_offset_y
        assigned = False
        for li, (y0, y1) in enumerate(textlines):
            if y0 <= y_full <= y1:
                line_map[li].append(p); assigned = True; break
        if not assigned and textlines:
            dists = [min(abs(y_full - y0), abs(y_full - y1)) for y0, y1 in textlines]
            line_map[int(np.argmin(dists))].append(p)
    return dict(line_map)


# ── Load index ────────────────────────────────────────────────────────────────
print('Loading index...')
with open(INDEX_PATH, 'rb') as f:
    index = pickle.load(f)

writer_to_entries = defaultdict(list)
for e in index:
    writer_to_entries[e['writer']].append(e)

orig_entry = next(e for e in index if e['writer'] == ORIG_WRITER and e['page'] == ORIG_PAGE)

# ── Pre-scan: collect all foreign pages that have valid textlines ─────────────
print('Pre-scanning XMLs for valid foreign pages...')
valid_foreign = []  

with zipfile.ZipFile(XML_ZIP) as zf:
    xml_names = set(zf.namelist())
    for e in index:
        if e['writer'] == ORIG_WRITER:
            continue
        xml_member = f'cvl-database-1-1/testset/xml/{e["writer"]:04d}-{e["page"]}_attributes.xml'
        if xml_member not in xml_names:
            continue
        try:
            raw = zf.read(xml_member)
            offset, lines = parse_cvl_xml(raw)
            if len(lines) > 0:
                valid_foreign.append((e['writer'], e, offset, lines))
        except Exception:
            continue

print(f'  Found {len(valid_foreign)} valid foreign pages')

# Pick one
FOREIGN_WRITER, foreign_entry, foreign_offset, foreign_lines = random.choice(valid_foreign)
print(f'Original : writer={ORIG_WRITER}  page={ORIG_PAGE}')
print(f'Foreign  : writer={FOREIGN_WRITER}  page={foreign_entry["page"]}')

page_name      = f'{ORIG_WRITER:04d}-{ORIG_PAGE}-cropped'
foreign_name   = f'{FOREIGN_WRITER:04d}-{foreign_entry["page"]}-cropped'


# ── Helper: read a file from tar by path ──────────────────────────────────────
print('Indexing tar (one-time scan)...')
t0 = time.time()
_tar_handle = tarfile.open(TAR_PATH)
_tar_index  = {m.name: m for m in _tar_handle.getmembers()}
print(f'  Indexed {len(_tar_index)} members in {time.time()-t0:.1f}s')

def read_from_tar(tar_path, member):
    return _tar_handle.extractfile(_tar_index[member]).read()

def read_coords(tar_path, page_name):
    member = f'cvl_test_patches/{page_name}/coordinates.json'
    return json.loads(read_from_tar(tar_path, member))

def read_patch_png(tar_path, page_name, filename):
    member = f'cvl_test_patches/{page_name}/{filename}'
    return Image.open(io.BytesIO(read_from_tar(tar_path, member))).convert('RGB')

def read_cropped_tif(zip_path, page_name):
    member = f'cvl-database-cropped-1-1/{page_name}.tif'
    with zipfile.ZipFile(zip_path) as zf:
        return Image.open(io.BytesIO(zf.read(member))).convert('RGB')

def read_xml(zip_path, writer, page):
    name = f'{writer:04d}-{page}_attributes.xml'
    member = f'cvl-database-1-1/testset/xml/{name}'
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(member)


# ── Load coords + XMLs ────────────────────────────────────────────────────────
print('Reading coordinates.json...')
orig_cj    = read_coords(TAR_PATH, page_name)
foreign_cj = read_coords(TAR_PATH, foreign_name)

print('Parsing XMLs...')
orig_xml_bytes = read_xml(XML_ZIP, ORIG_WRITER, ORIG_PAGE)
orig_offset, orig_lines = parse_cvl_xml(orig_xml_bytes)

print(f'  Orig    textlines: {len(orig_lines)}  crop_offset_y={orig_offset}')
print(f'  Foreign textlines: {len(foreign_lines)}  crop_offset_y={foreign_offset}')

orig_line_map    = assign_patches_to_lines(orig_cj['patches'],    orig_lines,    orig_offset)
foreign_line_map = assign_patches_to_lines(foreign_cj['patches'], foreign_lines, foreign_offset)

# pick a valid line to replace
line_idx = LINE_TO_REPLACE if LINE_TO_REPLACE in orig_line_map else sorted(orig_line_map.keys())[len(orig_line_map)//2]
print(f'  Replacing line index {line_idx} '
      f'({len(orig_line_map.get(line_idx,[]))} patches) '
      f'with foreign line {line_idx} '
      f'({len(foreign_line_map.get(line_idx,[]))} patches)')

win = orig_cj['win_size']  # 32
half = win // 2
page_img = read_cropped_tif(CROPPED_ZIP, page_name)

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 1 — Original Continuous Image
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding Image 1 (Original Unchanged)...')
out1 = os.path.join(OUT_DIR, f'image1_original_w{ORIG_WRITER}p{ORIG_PAGE}.png')
page_img.save(out1)
print(f'  Saved: {out1}')


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 2 — Original Page Reconstructed entirely from 32x32 Patches
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding Image 2 (Original as Patches)...')
img2 = page_img.copy()
total_orig_patches = len(orig_cj['patches'])

print(f'  Pasting all {total_orig_patches} original patches...')
for i, p in enumerate(orig_cj['patches']):
    if i % 200 == 0:
        print(f'    patch {i}/{total_orig_patches}...', flush=True)
    try:
        patch_img = read_patch_png(TAR_PATH, page_name, p['filename'])
        img2.paste(patch_img, (p['x'] - half, p['y'] - half))
    except Exception:
        pass

out2 = os.path.join(OUT_DIR, f'image2_original_patches_w{ORIG_WRITER}p{ORIG_PAGE}.png')
img2.save(out2)
print(f'  Saved: {out2}')


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 3 — Patch Reconstructed Page with Contaminated Line
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding Image 3 (Contaminated Patch Page)...')
img3 = img2.copy()
draw = ImageDraw.Draw(img3, 'RGBA')   # define draw HERE, after img3 exists

foreign_patches_in_line = foreign_line_map.get(line_idx, [])
if not foreign_patches_in_line and foreign_line_map:
    foreign_patches_in_line = foreign_line_map.get(sorted(foreign_line_map.keys())[0], [])

# Soft red tint over replaced line — keeps TIF background visible beneath
if orig_lines and line_idx < len(orig_lines):
    y0, y1 = orig_lines[line_idx]
    draw.rectangle([0, y0 - orig_offset - 2, img3.width, y1 - orig_offset + 2],
                   fill=(255, 180, 180, 100))   # semi-transparent, NO white wipe

# Compute y-shift for foreign patches
orig_y0, orig_y1 = orig_lines[line_idx]
orig_line_center = (orig_y0 + orig_y1) / 2 - orig_offset
foreign_y_vals   = [p['y'] for p in foreign_patches_in_line]
foreign_line_center = np.mean(foreign_y_vals) if foreign_y_vals else orig_line_center
y_shift = orig_line_center - foreign_line_center

# Paste foreign patches
print(f'  Pasting {len(foreign_patches_in_line)} foreign patches (y_shift={y_shift:.1f}px)...')
for i, fp in enumerate(foreign_patches_in_line):
    if i % 20 == 0:
        print(f'    patch {i}...', flush=True)
    try:
        patch_img = read_patch_png(TAR_PATH, foreign_name, fp['filename'])
        x = fp['x']
        y = int(fp['y'] + y_shift)
        img3.paste(patch_img, (x - half, y - half))
    except Exception:
        pass

# Red outline box around contaminated line
if orig_lines and line_idx < len(orig_lines):
    y0, y1 = orig_lines[line_idx]
    d2 = ImageDraw.Draw(img3)
    d2.rectangle([1, y0 - orig_offset - 2, img3.width - 2, y1 - orig_offset + 2],
                 outline=(220, 30, 30), width=3)

# Label bar
d3 = ImageDraw.Draw(img3)
label = f'Writer {ORIG_WRITER} | Line {line_idx} replaced with Writer {FOREIGN_WRITER} (keypoint patches on TIF background)'
d3.rectangle([0, 0, img3.width, 22], fill=(30, 30, 30))
d3.text((6, 4), label, fill=(255, 255, 100))

out3 = os.path.join(OUT_DIR, f'image3_contaminated_patches_w{ORIG_WRITER}p{ORIG_PAGE}.png')
img3.save(out3)
print(f'  Saved: {out3}')


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 4 — Foreign Writer Full Page Reconstructed from Patches
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding Image 4 (Foreign Page as Patches)...')
foreign_h, foreign_w = foreign_cj['image_shape']
foreign_page_img = read_cropped_tif(CROPPED_ZIP, foreign_name)
img4 = foreign_page_img.copy()
total_foreign_patches = len(foreign_cj['patches'])

print(f'  Pasting all {total_foreign_patches} foreign patches...')
for i, p in enumerate(foreign_cj['patches']):
    if i % 200 == 0:
        print(f'    patch {i}/{total_foreign_patches}...', flush=True)
    try:
        patch_img = read_patch_png(TAR_PATH, foreign_name, p['filename'])
        img4.paste(patch_img, (p['x'] - half, p['y'] - half))
    except Exception:
        pass

out4 = os.path.join(OUT_DIR, f'image4_foreign_patches_w{FOREIGN_WRITER}p{foreign_entry["page"]}.png')
img4.save(out4)
print(f'  Saved: {out4}')


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 5 — Foreign Continuous Image
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding Image 5 (Foreign Normal Page)...')
try:
    foreign_page_img = read_cropped_tif(CROPPED_ZIP, foreign_name)
    out5 = os.path.join(OUT_DIR, f'image5_foreign_original_w{FOREIGN_WRITER}p{foreign_entry["page"]}.png')
    foreign_page_img.save(out5)
    print(f'  Saved: {out5}')
except Exception as e:
    print(f'  WARNING: Could not load foreign continuous image: {e}')

print(f'\nDone! Check {OUT_DIR}/')
_tar_handle.close()