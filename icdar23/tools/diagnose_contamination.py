"""
diagnose_contamination.py
─────────────────────────
Diagnostic script to investigate why CVL patch-block replacement degrades
mAP much faster than CVL line replacement, and faster than ICDAR patch.

Run this BEFORE re-running the contamination experiments.

Usage:
    python diagnose_contamination.py \
        --h5_path      <exp_dir>/raw_patches_test.h5 \
        --index_path   <exp_dir>/patch_index_test.pkl \
        --pca_path     <exp_dir>/pca.pkl \
        --patches_dir  <cvl_scratch>/cvl_test_patches \
        --xml_dir      <cvl_scratch>/xml \
        --n_pages      20 \
        --lines_per_page 8
"""

import argparse, os, sys, pickle, json, re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import h5py
from sklearn.preprocessing import normalize

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--h5_path',        required=True)
parser.add_argument('--index_path',     required=True)
parser.add_argument('--pca_path',       required=True)
parser.add_argument('--patches_dir',    required=True,
                    help='dir with per-page subdirs containing coordinates.json')
parser.add_argument('--xml_dir',        required=True)
parser.add_argument('--n_pages',        type=int, default=20,
                    help='number of pages to inspect deeply')
parser.add_argument('--lines_per_page', type=int, default=8)
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Helpers (copied from contamination_eval_cvl.py)
# ══════════════════════════════════════════════════════════════════════════════
def find_xml_for_page(xml_dir, writer_id, text_id):
    patterns = [
        f'{writer_id}-{text_id}.xml',
        f'{writer_id}-{text_id}-cropped.xml',
        f'{int(writer_id):04d}-{int(text_id)}.xml',
        f'{int(writer_id):04d}-{text_id}.xml',
    ]
    for pat in patterns:
        p = os.path.join(xml_dir, pat)
        if os.path.isfile(p):
            return p
    for root, dirs, files in os.walk(xml_dir):
        for f in files:
            if f.endswith('.xml') and writer_id in f and str(text_id) in f:
                return os.path.join(root, f)
    return None


def parse_cvl_xml(xml_path):
    NS = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}'
    with open(xml_path, 'rb') as f:
        raw = f.read()
    raw = re.sub(rb'<\?xml[^?]*\?>', b'', raw, count=1)
    raw = re.sub(rb'[^\x09\x0A\x0D\x20-\xFF]', b'', raw)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return 0, []
    crop_offset_y = 0
    for tl in root.iter(f'{NS}dkTextLines'):
        cy = tl.get('cropY')
        if cy is not None:
            crop_offset_y = int(float(cy))
            break
    textlines = []
    for region in root.iter(f'{NS}AttrRegion'):
        if region.get('attrType') == '2' and region.get('fontType') in ('2', None):
            ys = []
            for pt in region.iter(f'{NS}Point'):
                y = pt.get('y')
                if y is not None:
                    ys.append(float(y))
            if ys:
                textlines.append((min(ys), max(ys)))
    textlines.sort(key=lambda t: t[0])
    return crop_offset_y, textlines


def get_block_indices(n_patches, n_blocks):
    """Reproduce contamination_eval_patches_cvl.py's block division."""
    blocks = []
    step = n_patches / n_blocks
    for i in range(n_blocks):
        start = int(round(i * step))
        end   = int(round((i + 1) * step))
        blocks.append(list(range(start, min(end, n_patches))))
    return blocks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Load index + pick sample pages
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SECTION 1 — Index / page inventory')
print('='*70)

with open(args.index_path, 'rb') as f:
    index = pickle.load(f)

wp_to_key = {(e['writer'], e['page']): e['hdf5_key'] for e in index}
writer_to_pages = defaultdict(list)
for e in index:
    writer_to_pages[e['writer']].append((e['writer'], e['page']))
all_writers = list(writer_to_pages.keys())
all_wps     = list(wp_to_key.keys())

print(f'Total pages   : {len(all_wps)}')
print(f'Total writers : {len(all_writers)}')
pages_per_writer = [len(v) for v in writer_to_pages.values()]
print(f'Pages/writer  : mean={np.mean(pages_per_writer):.1f}  '
      f'min={min(pages_per_writer)}  max={max(pages_per_writer)}')

# patch counts from index
patch_counts = [e['patch_end'] - e['patch_start'] for e in index]
print(f'Patch count   : mean={np.mean(patch_counts):.0f}  '
      f'median={np.median(patch_counts):.0f}  '
      f'std={np.std(patch_counts):.0f}  '
      f'min={min(patch_counts)}  max={max(patch_counts)}')

sample_wps = all_wps[:args.n_pages]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Patch norm check (are H5 patches already unit-norm?)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SECTION 2 — Patch L2-norm distribution (are patches already normalised?)')
print('='*70)

with h5py.File(args.h5_path, 'r') as h5f:
    for (w, p) in sample_wps[:5]:
        key = wp_to_key[(w, p)]
        raw = h5f[key][:50].astype(np.float32)     # first 50 patches
        norms = np.linalg.norm(raw, axis=1)
        print(f'  page {w}-{p}:  norm  mean={norms.mean():.4f}  '
              f'std={norms.std():.4f}  '
              f'min={norms.min():.4f}  '
              f'max={norms.max():.4f}')

print()
print('  → If mean ≈ 1.0 and std ≈ 0, patches ARE already unit-normalised in the H5.')
print('    In that case, the per-patch re-normalisation in the eval scripts is a no-op.')
print('    But the ICDAR double-load bug (see Section 5) would still be an error.')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Spatial ordering of patches inside H5
#   Key question: are patch indices stored in Y-coordinate order?
#   If not, "contiguous index blocks" ≠ "spatial stripes".
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SECTION 3 — Spatial ordering of patches in H5 vs coordinates.json')
print('='*70)
print('  Computing Spearman rank correlation between patch index and patch Y coord.')
print('  corr ≈ 1 → sorted top-to-bottom (blocks = spatial bands)')
print('  corr ≈ 0 → random / extraction order (blocks = arbitrary subsets)')
print()

from scipy.stats import spearmanr

spatial_corrs = []
pages_checked = 0

for (w, p) in all_wps:
    page_name = f'{int(w):04d}-{int(p)}-cropped'
    coords_path = os.path.join(args.patches_dir, page_name, 'coordinates.json')
    if not os.path.isfile(coords_path):
        page_name = f'{w}-{p}-cropped'
        coords_path = os.path.join(args.patches_dir, page_name, 'coordinates.json')
    if not os.path.isfile(coords_path):
        continue

    with open(coords_path) as f:
        cj = json.load(f)

    patches_info = cj.get('patches', [])
    if len(patches_info) < 10:
        continue

    # patch_idx is the H5 row, 'y' is the top-left Y in cropped-image space
    idxs = [pc['patch_idx'] for pc in patches_info]
    ys   = [pc['y']         for pc in patches_info]
    if len(set(ys)) < 5:
        continue

    corr, _ = spearmanr(idxs, ys)
    spatial_corrs.append(corr)
    pages_checked += 1
    if pages_checked >= 200:       # enough for a reliable estimate
        break

if spatial_corrs:
    print(f'  Pages checked : {pages_checked}')
    print(f'  Spearman corr (patch_idx vs Y) :')
    print(f'    mean = {np.mean(spatial_corrs):.4f}')
    print(f'    std  = {np.std(spatial_corrs):.4f}')
    print(f'    min  = {np.min(spatial_corrs):.4f}')
    print(f'    max  = {np.max(spatial_corrs):.4f}')
    print()

    # histogram
    bins = [-1, -0.5, -0.1, 0.1, 0.5, 0.9, 1.01]
    hist, _ = np.histogram(spatial_corrs, bins=bins)
    labels = ['(-1,-0.5)', '(-0.5,-0.1)', '(-0.1,0.1)', '(0.1,0.5)', '(0.5,0.9)', '(0.9,1.0]']
    print('  Distribution of correlation values:')
    for label, count in zip(labels, hist):
        bar = '█' * (count * 40 // max(hist, default=1))
        print(f'    {label:>12}  {bar}  {count}')
else:
    print('  Could not compute spatial correlations — coordinates.json not found.')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Block vs Line spatial overlap analysis
#   For pages where we have both coordinates.json and XML:
#   how much do index-based blocks overlap with actual text lines?
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SECTION 4 — Block-vs-line spatial mismatch (sample pages)')
print('='*70)
print('  For each block, we show the Y-range of patches it contains')
print('  vs the Y-range of actual text lines.')
print()

detailed_pages = 0
block_line_purity_scores = []    # fraction of each block that falls in ONE line

for (w, p) in all_wps:
    if detailed_pages >= 5:
        break

    page_name = f'{int(w):04d}-{int(p)}-cropped'
    coords_path = os.path.join(args.patches_dir, page_name, 'coordinates.json')
    if not os.path.isfile(coords_path):
        continue

    with open(coords_path) as f:
        cj = json.load(f)
    patches_info = cj.get('patches', [])
    if len(patches_info) < 50:
        continue

    xml_path = find_xml_for_page(args.xml_dir, f'{int(w):04d}', str(int(p)))
    if xml_path is None:
        continue

    crop_offset_y, textlines = parse_cvl_xml(xml_path)
    if not textlines:
        continue

    # Build patch_idx -> Y mapping
    idx_to_y = {pc['patch_idx']: pc['y'] for pc in patches_info}
    n_patches = len(patches_info)

    # Index-based blocks
    n_blocks = min(args.lines_per_page, n_patches)
    blocks   = get_block_indices(n_patches, n_blocks)

    print(f'  Page {w}-{p}:  {n_patches} patches, {len(textlines)} XML lines, '
          f'{n_blocks} index blocks')
    print(f'  XML textlines (Y range in full-image coords, after offset {crop_offset_y}):')
    for li, (y1, y2) in enumerate(textlines):
        # count how many patches fall in this line
        count = sum(1 for pc in patches_info
                    if y1 <= (pc['y'] + crop_offset_y) <= y2)
        print(f'    line {li:2d}: y=[{y1:.0f}, {y2:.0f}]  patches≈{count}')

    print(f'  Index blocks (Y range of patches inside each block):')
    for bi, block in enumerate(blocks):
        ys_in_block = [idx_to_y.get(idx, None) for idx in block]
        ys_in_block = [y for y in ys_in_block if y is not None]
        if not ys_in_block:
            print(f'    block {bi}: no coords found')
            continue
        y_min_b = min(ys_in_block)
        y_max_b = max(ys_in_block)

        # what fraction of patches in this block are in each text line?
        line_counts = defaultdict(int)
        for y in ys_in_block:
            y_full = y + crop_offset_y
            for li, (y1, y2) in enumerate(textlines):
                if y1 <= y_full <= y2:
                    line_counts[li] += 1
                    break
        dominant_line = max(line_counts, key=line_counts.get, default=-1)
        purity = line_counts[dominant_line] / len(ys_in_block) if ys_in_block else 0
        block_line_purity_scores.append(purity)
        print(f'    block {bi:2d}: patch_idx=[{block[0]}, {block[-1]}]  '
              f'y=[{y_min_b:.0f}, {y_max_b:.0f}]  '
              f'dominant_line={dominant_line}  purity={purity:.2f}')
    print()
    detailed_pages += 1

if block_line_purity_scores:
    print(f'  Overall block-line purity (fraction of block patches in dominant XML line):')
    print(f'    mean = {np.mean(block_line_purity_scores):.3f}')
    print(f'    std  = {np.std(block_line_purity_scores):.3f}')
    print()
    print('  → Purity ≈ 1.0 means index blocks align well with actual text lines.')
    print('  → Purity ≈ 1/n_lines means blocks are randomly mixed across lines.')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ICDAR double-load bug
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SECTION 5 — ICDAR double-load bug (contamination_eval_icdar.py)')
print('='*70)
print("""
  BUG CONFIRMED: In contamination_eval_icdar.py, patches are normalised and then
  immediately overwritten by a second raw load from the H5:

      patches = h5f[key][:].astype(np.float32)
      norms   = np.linalg.norm(patches, axis=1, keepdims=True)
      patches = patches / np.where(norms < 1e-8, 1.0, norms)
      patches = h5f[key][:].astype(np.float32)   ← OVERWRITES WITH RAW PATCHES!

  The same bug occurs for f_patches (foreign page).

  Effect: ICDAR contamination eval mixes UN-normalised native patches with
  UN-normalised foreign patches, while CVL patch eval mixes NORMALISED patches.
  This makes the two experiments INCOMPARABLE for any cross-dataset conclusion.

  The fix (two lines to DELETE — mark with ← DELETE):
""")

icdar_bug_lines = [
    ("patches = h5f[key][:].astype(np.float32)      ← DELETE (overwrite of normalised patches)",
     "  This is the second load of native patches. Remove it."),
    ("f_patches = h5f[fkey][:max_foreign + 1].astype(np.float32)  ← DELETE",
     "  This is the second load of foreign patches. Remove it."),
]
for line, comment in icdar_bug_lines:
    print(f'    {line}')
    print(f'    {comment}')
    print()

print('  After the fix, verify the patch norms as shown in Section 2.')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Actual contamination ratio at each level
#   For a sample of pages, compute the ACTUAL fraction of foreign patches
#   that ends up in the mixed set, for both block and line replacement.
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SECTION 6 — Actual contamination fractions vs nominal (5 sample pages)')
print('='*70)

import random
random.seed(0)
np.random.seed(0)

foreign_map_sample = {}
for (w, p) in sample_wps:
    other_writers = [ow for ow in all_writers if ow != w]
    fw = random.choice(other_writers)
    fp = random.choice(writer_to_pages[fw])[1]
    foreign_map_sample[(w, p)] = (fw, fp)

levels = list(range(0, args.lines_per_page // 2 + 1))
avg_patches = np.mean([e['patch_end'] - e['patch_start'] for e in index])
patches_per_line_icdar = max(1, int(round(avg_patches / args.lines_per_page)))

print(f'  levels          : {levels}')
print(f'  nominal block % : {[f"{l/args.lines_per_page*100:.1f}" for l in levels]}')
print(f'  patches/block   : ~{int(avg_patches/args.lines_per_page)}')
print()
print(f'  {"Level":>5}  {"Nom%":>6}  {"Actual native":>14}  {"Actual foreign":>14}  {"Actual pct foreign":>18}')
print(f'  {"-"*65}')

with h5py.File(args.h5_path, 'r') as h5f:
    for lvl in levels:
        n_native_list, n_foreign_list, total_list = [], [], []
        for (w, p) in sample_wps:
            key = wp_to_key[(w, p)]
            n_patches = h5f[key].shape[0]
            n_blocks  = min(args.lines_per_page, n_patches)
            blocks    = get_block_indices(n_patches, n_blocks)

            (fw, fp)  = foreign_map_sample[(w, p)]
            fkey = wp_to_key[(fw, fp)]
            n_f_patches = h5f[fkey].shape[0]
            n_foreign_blocks = min(args.lines_per_page, n_f_patches)
            foreign_blocks   = get_block_indices(n_f_patches, n_foreign_blocks)

            if lvl == 0:
                n_native  = n_patches
                n_foreign = 0
            else:
                n_replace = min(lvl, len(blocks))
                blocks_to_remove = set(random.sample(range(len(blocks)), n_replace))
                keep_indices = []
                for bi, block in enumerate(blocks):
                    if bi not in blocks_to_remove:
                        keep_indices.extend(block)
                n_foreign_pick = min(n_replace, len(foreign_blocks))
                foreign_picks  = random.sample(range(len(foreign_blocks)), n_foreign_pick)
                foreign_indices = []
                for fbi in foreign_picks:
                    foreign_indices.extend(foreign_blocks[fbi])
                n_native  = len(keep_indices)
                n_foreign = len(foreign_indices)

            n_native_list.append(n_native)
            n_foreign_list.append(n_foreign)
            total_list.append(n_native + n_foreign)

        nom_pct    = lvl / args.lines_per_page * 100
        actual_pct = np.mean(n_foreign_list) / np.mean(total_list) * 100
        print(f'  {lvl:>5}  {nom_pct:>6.1f}%  '
              f'{np.mean(n_native_list):>14.0f}  '
              f'{np.mean(n_foreign_list):>14.0f}  '
              f'{actual_pct:>17.1f}%')

print()
print('  → Nominal% should match Actual% if block division is working correctly.')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Conceptual diagnosis summary
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SECTION 7 — Root cause summary & interpretation')
print('='*70)
print("""
  RESULTS TO EXPLAIN
  ──────────────────
  Level  CVL-Line  CVL-PatchBlock  ICDAR-Patch
    0    0.8808      0.8808         0.6192
    1    0.7557      0.6737         0.5405
    2    0.5435      0.3051         0.3504
    3    0.4002      0.1237         0.1821
    4    0.2969      0.0499         0.0942

  LIKELY ROOT CAUSES
  ──────────────────

  [A] Spatial incoherence of index-based blocks (CVL Patch >> CVL Line drop)
      ─────────────────────────────────────────────────────────────────────────
      contamination_eval_patches_cvl.py divides patches by H5 index position,
      NOT by their Y-coordinate / spatial line assignment.

      If patches are stored in non-Y-sorted order (e.g. word-extraction order,
      random scan, or grouped by word bounding box), then an "index block" spans
      patches from MANY different text lines, not a single one.

      Consequence for SumPooling:
        • Line replacement:   remove 1 entire line, keep N-1 complete lines.
          The kept lines are 100% pure native → SumPool is still a strong,
          coherent writer descriptor from (N-1)/N of the writing.
        • Index-block removal: remove ~1/8 of patches drawn from ALL lines.
          Every line loses some patches and gains some foreign patches.
          SumPool is contaminated everywhere, not just in one spatial band.
          This "distributed" contamination is far more destructive.

      Section 3 (Spearman correlation) tells you definitively whether this is
      the case. If mean corr << 1, index blocks ≠ spatial lines.

  [B] ICDAR double-load bug (cross-dataset comparison invalid)
      ─────────────────────────────────────────────────────────
      contamination_eval_icdar.py re-loads patches from H5 AFTER normalising
      them, discarding the normalisation. ICDAR mixes raw (un-normalised)
      patches; CVL patch mixes L2-normalised patches. You cannot directly
      compare the drop rates between these two experiments.

      Fix: delete the two duplicate load lines (see Section 5).

  [C] Effective contamination fraction mismatch (minor)
      ──────────────────────────────────────────────────
      CVL pages have mean 6.3 XML lines but the line eval uses lines_per_page=8
      as the denominator. Replacing 4 lines from a page with 6.3 mean lines
      removes ≈63% of content, not 50%. The patch block eval correctly removes
      exactly 50%. This makes CVL-Line worse at high levels than it looks.
      Section 6 confirms the actual fractions.

  RECOMMENDED FIXES
  ─────────────────
  1. Fix contamination_eval_icdar.py: remove the two duplicate load lines
     (see Section 5 for the exact lines).

  2. In contamination_eval_patches_cvl.py, replace the index-based block
     division with a Y-coordinate-sorted block division:
       a. Load coordinates.json per page.
       b. Sort patches by Y coordinate (ascending).
       c. Divide the sorted list into n_blocks equal groups.
       d. Repeat for the foreign page.
     This ensures "block i" covers a spatial horizontal stripe, matching
     the line-replacement semantics.

  3. For a fair comparison, normalise contamination % by actual line patch
     counts (not by fixed lines_per_page=8), or report as absolute
     "fraction of patches replaced" rather than "lines replaced."
""")

print('Diagnostics done.')
