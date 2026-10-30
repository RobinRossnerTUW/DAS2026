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
TAR_PATH   = '/data/rrossner/datasets/CVL/7663856/cvl_test_patches_7663856.tar'
CROPPED_ZIP= '/data/rrossner/datasets/CVL/cvl-database-cropped-1-1.zip'
XML_ZIP    = '/data/rrossner/datasets/CVL/cvl-database-1-1.zip'
INDEX_PATH = '/data/rrossner/datasets/CVL/7663856/patch_index_test.pkl'
OUT_DIR    = '/data/rrossner/DAS2026/visualize'

ORIG_WRITER, ORIG_PAGE   = 80, 1    # change if you want a different page
FOREIGN_WRITER           = 189      # None = auto-pick
LINE_TO_REPLACE          = 2         # 0-indexed line index to contaminate
FOREIGN_LINE             = 2
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
match = next(((w, e, o, l) for w, e, o, l in valid_foreign if w == FOREIGN_WRITER and e['page'] == 4), None)
if match is None:
    raise ValueError(f'Writer {FOREIGN_WRITER} not found in valid_foreign list')
FOREIGN_WRITER, foreign_entry, foreign_offset, foreign_lines = match
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
available_lines = sorted(orig_line_map.keys())
line_idx = available_lines[LINE_TO_REPLACE] if LINE_TO_REPLACE < len(available_lines) else available_lines[len(available_lines)//2]
available_foreign_lines = sorted(foreign_line_map.keys())
foreign_line_key = available_foreign_lines[FOREIGN_LINE] if FOREIGN_LINE < len(available_foreign_lines) else available_foreign_lines[-1]
print(f'  Available line keys in orig_line_map: {sorted(orig_line_map.keys())}')
print(f'  line_idx resolved to: {line_idx}')
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
img2 = Image.new('RGB', page_img.size, (255, 255, 255))
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
# IMAGE 3 — Original patches with line replaced by foreign keypoint patches
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding Image 3 (Contaminated Patch Page)...')
img3 = img2.copy()

# Wipe the original line band white so no original patches bleed through
orig_y0, orig_y1 = orig_lines[line_idx]
band_y0 = int(orig_y0 - orig_offset) - half
band_y1 = int(orig_y1 - orig_offset) + half
draw = ImageDraw.Draw(img3)
draw.rectangle([0, band_y0, img3.width, band_y1], fill=(255, 255, 255))

# Paste foreign keypoint patches, shifted to align with the original line centre
orig_line_center    = (orig_y0 + orig_y1) / 2 - orig_offset
foreign_patches_in_line = foreign_line_map.get(foreign_line_key, [])
if not foreign_patches_in_line:
    raise ValueError(f'No patches found for foreign line {FOREIGN_LINE}')

foreign_y_vals      = [p['y'] for p in foreign_patches_in_line]
foreign_line_center = np.mean(foreign_y_vals)
y_shift = orig_line_center - foreign_line_center

print(f'  Pasting {len(foreign_patches_in_line)} foreign patches (y_shift={y_shift:.1f}px)...')
occupied = []
for i, fp in enumerate(foreign_patches_in_line):
    if i % 20 == 0:
        print(f'    patch {i}...', flush=True)
    try:
        x = fp['x']
        y = int(fp['y'] + y_shift)
        box = (x - half, y - half, x + half, y + half)
        if any(box[0] < ox1 and box[2] > ox0 and box[1] < oy1 and box[3] > oy0
               for ox0, oy0, ox1, oy1 in occupied):
            continue
        occupied.append(box)
        patch_img = read_patch_png(TAR_PATH, foreign_name, fp['filename'])
        pd = ImageDraw.Draw(patch_img)
        pd.rectangle([0, 0, patch_img.width - 1, patch_img.height - 1], outline=(0, 0, 0), width=1)
        img3.paste(patch_img, (x - half, y - half))
    except Exception:
        pass

d2 = ImageDraw.Draw(img3)
d2.rectangle([1, band_y0, img3.width - 2, band_y1], outline=(220, 30, 30), width=3)

out3 = os.path.join(OUT_DIR, f'image3_contaminated_patches_w{ORIG_WRITER}p{ORIG_PAGE}.png')
img3.save(out3)
print(f'  Saved: {out3}')


# ══════════════════════════════════════════════════════════════════════════════
# FINAL IMAGES — saved to visualize/final/
# ══════════════════════════════════════════════════════════════════════════════
FINAL_DIR = os.path.join(OUT_DIR, 'final')
os.makedirs(FINAL_DIR, exist_ok=True)

# ── Compute crop bounds so all three finals show the same number of lines ─────
N_LINES = len(available_foreign_lines)  # foreign drives the line count

# Foreign: crop from first to last available line
f_top = foreign_lines[available_foreign_lines[0]]
f_bot = foreign_lines[available_foreign_lines[-1]]
foreign_crop_y0 = max(0, int(f_top[0] - foreign_offset) - half)
foreign_crop_y1 = int(f_bot[1] - foreign_offset) + half

# Original: N_LINES consecutive lines centred on line_idx
pos   = available_lines.index(line_idx)
start = max(0, pos - N_LINES // 2)
end   = min(len(available_lines), start + N_LINES)
start = max(0, end - N_LINES)  # re-clamp if end hit the ceiling
selected_orig_lines = available_lines[start:end]

o_top = orig_lines[selected_orig_lines[0]]
o_bot = orig_lines[selected_orig_lines[-1]]
orig_crop_y0 = max(0, int(o_top[0] - orig_offset) - half)
orig_crop_y1 = int(o_bot[1] - orig_offset) + half

print(f'\n  Final crop: orig y={orig_crop_y0}-{orig_crop_y1}, '
      f'foreign y={foreign_crop_y0}-{foreign_crop_y1}, N_LINES={N_LINES}')

# ── Final A — TIF background, only target line band wiped white, patches with borders
print('\nBuilding Final A...')
imgA = page_img.copy()
draw = ImageDraw.Draw(imgA)
draw.rectangle([0, band_y0, imgA.width, band_y1], fill=(255, 255, 255))
orig_patches_in_line = orig_line_map.get(line_idx, [])
occupied = []
for fp in orig_patches_in_line:
    try:
        x, y = fp['x'], fp['y']
        box = (x - half, y - half, x + half, y + half)
        if any(box[0] < ox1 and box[2] > ox0 and box[1] < oy1 and box[3] > oy0
               for ox0, oy0, ox1, oy1 in occupied):
            continue
        occupied.append(box)
        patch_img = read_patch_png(TAR_PATH, page_name, fp['filename'])
        pd = ImageDraw.Draw(patch_img)
        pd.rectangle([0, 0, patch_img.width - 1, patch_img.height - 1], outline=(0, 0, 0), width=1)
        imgA.paste(patch_img, (x - half, y - half))
    except Exception:
        pass
dA = ImageDraw.Draw(imgA)
dA.rectangle([1, band_y0, imgA.width - 2, band_y1], outline=(220, 30, 30), width=3)
imgA = imgA.crop((0, orig_crop_y0, imgA.width, orig_crop_y1))
outA = os.path.join(FINAL_DIR, f'finalA_orig_line_patches_w{ORIG_WRITER}p{ORIG_PAGE}.png')
imgA.save(outA)
print(f'  Saved: {outA}')

# ── Final B — TIF background, only foreign line band wiped white, patches with borders
print('\nBuilding Final B...')
foreign_page_img = read_cropped_tif(CROPPED_ZIP, foreign_name)
imgB = foreign_page_img.copy()
f_y0, f_y1 = foreign_lines[foreign_line_key]
foreign_band_y0 = int(f_y0 - foreign_offset) - half
foreign_band_y1 = int(f_y1 - foreign_offset) + half
draw = ImageDraw.Draw(imgB)
draw.rectangle([0, foreign_band_y0, imgB.width, foreign_band_y1], fill=(255, 255, 255))
occupied = []
for fp in foreign_patches_in_line:
    try:
        x, y = fp['x'], fp['y']
        box = (x - half, y - half, x + half, y + half)
        if any(box[0] < ox1 and box[2] > ox0 and box[1] < oy1 and box[3] > oy0
               for ox0, oy0, ox1, oy1 in occupied):
            continue
        occupied.append(box)
        patch_img = read_patch_png(TAR_PATH, foreign_name, fp['filename'])
        pd = ImageDraw.Draw(patch_img)
        pd.rectangle([0, 0, patch_img.width - 1, patch_img.height - 1], outline=(0, 0, 0), width=1)
        imgB.paste(patch_img, (x - half, y - half))
    except Exception:
        pass
dB = ImageDraw.Draw(imgB)
dB.rectangle([1, foreign_band_y0, imgB.width - 2, foreign_band_y1], outline=(220, 30, 30), width=3)
imgB = imgB.crop((0, foreign_crop_y0, imgB.width, foreign_crop_y1))
outB = os.path.join(FINAL_DIR, f'finalB_foreign_line_patches_w{FOREIGN_WRITER}p{foreign_entry["page"]}.png')
imgB.save(outB)
print(f'  Saved: {outB}')

# ── Final C — Image 3 cropped to same orig bounds
print('\nBuilding Final C (contaminated, cropped)...')
imgC = img3.crop((0, orig_crop_y0, img3.width, orig_crop_y1))
outC = os.path.join(FINAL_DIR, f'finalC_contaminated_w{ORIG_WRITER}p{ORIG_PAGE}.png')
imgC.save(outC)
print(f'  Saved: {outC}')

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 4 — Foreign Writer Full Page Reconstructed from Patches
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding Image 4 (Foreign Page as Patches)...')
foreign_page_img = read_cropped_tif(CROPPED_ZIP, foreign_name)
foreign_h, foreign_w = foreign_cj['image_shape']
img4 = Image.new('RGB', foreign_page_img.size, (255, 255, 255))
total_foreign_patches = len(foreign_cj['patches'])


print(f'  band_y0={band_y0}, band_y1={band_y1}')
print(f'  orig_line_center={orig_line_center:.1f}')
print(f'  foreign_line_center={foreign_line_center:.1f}')
print(f'  y_shift={y_shift:.1f}')
print(f'  sample foreign patch y values: {foreign_y_vals[:5]}')
print(f'  orig_offset={orig_offset}, orig_y0={orig_y0}, orig_y1={orig_y1}')
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