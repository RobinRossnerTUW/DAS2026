"""
visualize_contamination.py
──────────────────────────
Generates visual panels showing how retrieval results degrade
as lines are contaminated with foreign writer patches.

For selected query pages, shows:
  - Query page thumbnail
  - Top-K retrieved pages at each contamination level
  - Green border = correct writer, Red border = wrong writer

Logs to wandb as image artifacts and/or saves to disk.

Usage:
    python visualize_contamination.py \
        --contam_pkl   contamination_results_cvl.pkl \
        --patches_dir  /path/to/cvl_test_patches \
        --images_dir   /path/to/cvl-database-cropped-1-1 \
        --out_dir      contamination_vis \
        --top_k 10 \
        --num_queries 5 \
        --seed 42
"""

import argparse, os, sys, pickle, random, json, re, glob
from collections import defaultdict

import numpy as np
import h5py
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--h5_path',       required=True, help='raw_patches_test.h5')
parser.add_argument('--index_path',    required=True, help='patch_index_test.pkl')
parser.add_argument('--pca_path',      required=True, help='pca.pkl')
parser.add_argument('--patches_dir',   required=True, help='extracted patches with coordinates.json')
parser.add_argument('--xml_dir',       required=True, help='CVL XML files')
parser.add_argument('--images_dir',    required=True, help='original cropped CVL images (tif)')
parser.add_argument('--out_dir',       default='contamination_vis')
parser.add_argument('--top_k',         type=int, default=10)
parser.add_argument('--num_queries',   type=int, default=8,
                    help='number of query pages to visualize')
parser.add_argument('--n_trials',      type=int, default=1,
                    help='just 1 trial is fine for visualization')
parser.add_argument('--seed',          type=int, default=42)
parser.add_argument('--no_wandb',      action='store_true')
parser.add_argument('--wandb_project', type=str, default='cvl-contamination-vis')
parser.add_argument('--wandb_run',     type=str, default=None)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)

# lazy imports for visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ── wandb ─────────────────────────────────────────────────────────────────────
if not args.no_wandb:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
else:
    wandb = None

# ══════════════════════════════════════════════════════════════════════════════
# Reuse XML parsing from contamination_eval_cvl.py
# ══════════════════════════════════════════════════════════════════════════════
import xml.etree.ElementTree as ET

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
        if region.get('attrType') == '2' and region.get('fontType') == '2':
            ys = []
            for pt in region.iter(f'{NS}Point'):
                y = pt.get('y')
                if y is not None:
                    ys.append(float(y))
            if ys:
                textlines.append((min(ys), max(ys)))
    textlines.sort(key=lambda t: t[0])
    return crop_offset_y, textlines


def find_xml_for_page(xml_dir, writer_id, text_id):
    patterns = [
        f'{writer_id}-{text_id}_attributes.xml',
        f'{writer_id}-{text_id}.xml',
    ]
    for pat in patterns:
        path = os.path.join(xml_dir, pat)
        if os.path.isfile(path):
            return path
    for root_d, dirs, files in os.walk(xml_dir):
        for f in files:
            if f.endswith('.xml') and writer_id in f and str(text_id) in f:
                return os.path.join(root_d, f)
    return None


def assign_patches_to_lines(coords_json, textlines, crop_offset_y):
    patches = coords_json['patches']
    line_to_patches = defaultdict(list)
    for patch in patches:
        patch_y_full = patch['y'] + crop_offset_y
        assigned = False
        for line_idx, (y_min, y_max) in enumerate(textlines):
            if y_min <= patch_y_full <= y_max:
                line_to_patches[line_idx].append(patch['patch_idx'])
                assigned = True
                break
        if not assigned and textlines:
            dists = [min(abs(patch_y_full - y_min), abs(patch_y_full - y_max))
                     for y_min, y_max in textlines]
            nearest = np.argmin(dists)
            line_to_patches[nearest].append(patch['patch_idx'])
    return dict(line_to_patches)


from page_encodings import SumPooling

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
print('Loading artifacts...')
with open(args.index_path, 'rb') as f:
    index = pickle.load(f)

pca = pickle.load(open(args.pca_path, 'rb'))
pooling = SumPooling('l2', pn_alpha=0.4)

wp_to_key = {(e['writer'], e['page']): e['hdf5_key'] for e in index}
writer_to_pages = defaultdict(list)
for e in index:
    writer_to_pages[e['writer']].append((e['writer'], e['page']))

all_writers = list(writer_to_pages.keys())
all_wps = sorted(wp_to_key.keys())

# Build line map
print('Building line assignments...')
page_line_map = {}
page_num_lines = {}
for (w, p) in tqdm(all_wps, desc='Line map'):
    page_name = f'{int(w):04d}-{int(p)}-cropped'
    coords_path = os.path.join(args.patches_dir, page_name, 'coordinates.json')
    if not os.path.isfile(coords_path):
        page_line_map[(w, p)] = {}
        page_num_lines[(w, p)] = 0
        continue

    with open(coords_path) as f:
        coords_json = json.load(f)

    xml_path = find_xml_for_page(args.xml_dir, f'{int(w):04d}', str(int(p)))
    if xml_path is None:
        page_line_map[(w, p)] = {}
        page_num_lines[(w, p)] = 0
        continue

    crop_offset_y, textlines = parse_cvl_xml(xml_path)
    line_map = assign_patches_to_lines(coords_json, textlines, crop_offset_y)
    page_line_map[(w, p)] = line_map
    page_num_lines[(w, p)] = len(textlines)

# ══════════════════════════════════════════════════════════════════════════════
# Select query pages (pages with enough lines and whose writer has 5 pages)
# ══════════════════════════════════════════════════════════════════════════════
eligible = [(w, p) for (w, p) in all_wps
            if page_num_lines.get((w, p), 0) >= 6
            and len(writer_to_pages[w]) >= 5]
random.shuffle(eligible)
query_pages = eligible[:args.num_queries]
print(f'Selected {len(query_pages)} query pages for visualization')

# contamination levels
avg_lines = np.mean([n for n in page_num_lines.values() if n > 0])
max_replace = int(avg_lines / 2)
line_levels = list(range(0, max_replace + 1))

# ══════════════════════════════════════════════════════════════════════════════
# Helper: load page thumbnail
# ══════════════════════════════════════════════════════════════════════════════
def load_page_thumbnail(writer_id, page_id, max_height=200):
    """Load original cropped TIF and resize for display."""
    page_name = f'{int(writer_id):04d}-{int(page_id)}-cropped'
    for ext in ['.tif', '.tiff', '.png', '.jpg']:
        path = os.path.join(args.images_dir, page_name + ext)
        if os.path.isfile(path):
            img = Image.open(path).convert('L')
            ratio = max_height / img.height
            img = img.resize((int(img.width * ratio), max_height), Image.LANCZOS)
            return np.array(img)
    # fallback: gray placeholder
    return np.ones((max_height, int(max_height * 0.7)), dtype=np.uint8) * 200

def load_page_image_full(writer_id, page_id):
    """Load original cropped image at full resolution as grayscale numpy array."""
    page_name = f'{int(writer_id):04d}-{int(page_id)}-cropped'
    for ext in ['.tif', '.tiff', '.png', '.jpg']:
        path = os.path.join(args.images_dir, page_name + ext)
        if os.path.isfile(path):
            img = Image.open(path).convert('L')
            return np.array(img)
    return np.ones((500, 350), dtype=np.uint8) * 255

# ══════════════════════════════════════════════════════════════════════════════
# Run contamination + collect rankings for selected queries
# ══════════════════════════════════════════════════════════════════════════════
print('\nComputing contaminated embeddings...')

# We need the full gallery at each level to compute rankings
# Strategy: compute level-0 (clean) embeddings for all pages,
# then for query pages, recompute at each contamination level

# 1. Compute clean embeddings for ALL pages
clean_page_embs = {}
with h5py.File(args.h5_path, 'r') as h5f:
    for (w, p) in tqdm(all_wps, desc='Clean embeddings'):
        key = wp_to_key[(w, p)]
        patches = h5f[key][:].astype(np.float32)
        norms = np.linalg.norm(patches, axis=1, keepdims=True)
        patches = patches / np.where(norms < 1e-8, 1.0, norms)
        enc = pooling.encode([patches])
        clean_page_embs[(w, p)] = enc.flatten()

# PCA transform all clean embeddings
emb_matrix = np.stack([clean_page_embs[wp] for wp in all_wps])
emb_matrix = pca.transform(emb_matrix)
emb_matrix = normalize(emb_matrix, axis=1)
for i, wp in enumerate(all_wps):
    clean_page_embs[wp] = emb_matrix[i]

writers_array = np.array([w for w, p in all_wps])

# 2. For each query page × each level, compute contaminated embedding
# and find top-K from the clean gallery
print('Computing rankings at each contamination level...')

# results: query_rankings[(w,p)][level] = list of (rank_writer, rank_page, is_correct)
query_rankings = {}

with h5py.File(args.h5_path, 'r') as h5f:
    for (qw, qp) in tqdm(query_pages, desc='Query pages'):
        query_rankings[(qw, qp)] = {}

        key = wp_to_key[(qw, qp)]
        patches = h5f[key][:].astype(np.float32)
        norms = np.linalg.norm(patches, axis=1, keepdims=True)
        patches = patches / np.where(norms < 1e-8, 1.0, norms)

        line_map = page_line_map.get((qw, qp), {})
        sorted_lines = sorted(line_map.keys())

        # pick a foreign writer (consistent across levels)
        other_writers = [ow for ow in all_writers if ow != qw]
        foreign_w = random.choice(other_writers)
        foreign_p = random.choice(writer_to_pages[foreign_w])[1]
        fkey = wp_to_key[(foreign_w, foreign_p)]
        f_patches = h5f[fkey][:].astype(np.float32)
        fn = np.linalg.norm(f_patches, axis=1, keepdims=True)
        f_patches = f_patches / np.where(fn < 1e-8, 1.0, fn)
        f_line_map = page_line_map.get((foreign_w, foreign_p), {})
        f_sorted_lines = sorted(f_line_map.keys())

        for lvl in line_levels:
            if lvl == 0 or not line_map:
                mixed = patches
            else:
                n_replace = min(lvl, len(sorted_lines))
                lines_to_replace = set(random.sample(sorted_lines, n_replace))

                keep_indices = []
                for li in sorted_lines:
                    if li not in lines_to_replace:
                        keep_indices.extend(line_map[li])

                foreign_indices = []
                n_foreign_lines = min(n_replace, len(f_sorted_lines))
                if n_foreign_lines > 0:
                    foreign_line_picks = random.sample(f_sorted_lines, n_foreign_lines)
                    for fli in foreign_line_picks:
                        foreign_indices.extend(f_line_map[fli])

                keep_indices = [i for i in keep_indices if i < patches.shape[0]]
                foreign_indices = [i for i in foreign_indices if i < f_patches.shape[0]]

                if keep_indices and foreign_indices:
                    mixed = np.concatenate([patches[keep_indices], f_patches[foreign_indices]])
                elif keep_indices:
                    mixed = patches[keep_indices]
                else:
                    mixed = patches

            # encode + PCA
            enc = pooling.encode([mixed])
            enc_tf = pca.transform(enc.reshape(1, -1))
            enc_tf = normalize(enc_tf, axis=1).flatten()

            # compute similarities to entire gallery
            sims = emb_matrix @ enc_tf
            # exclude self
            query_idx = all_wps.index((qw, qp))
            sims[query_idx] = -999

            ranked_indices = np.argsort(-sims)
            top_k_results = []
            for rank_idx in ranked_indices[:args.top_k]:
                rw, rp = all_wps[rank_idx]
                top_k_results.append({
                    'writer': int(rw),
                    'page': int(rp),
                    'correct': int(rw) == int(qw),
                    'similarity': float(sims[rank_idx]),
                })
            # get textline ranges for visualization overlay
            xml_path_q = find_xml_for_page(args.xml_dir, f'{int(qw):04d}', str(int(qp)))
            tl_ranges = []
            crop_y_q = 0
            if xml_path_q:
                crop_y_q, tl_ranges = parse_cvl_xml(xml_path_q)

            # figure out original image height for scaling
            page_name_q = f'{int(qw):04d}-{int(qp)}-cropped'
            coords_path_q = os.path.join(args.patches_dir, page_name_q, 'coordinates.json')
            img_h = 1
            if os.path.isfile(coords_path_q):
                with open(coords_path_q) as cf:
                    cj = json.load(cf)
                    img_h = cj.get('image_shape', [1])[0]

            # get foreign page textlines for splicing
            xml_path_f = find_xml_for_page(args.xml_dir, f'{int(foreign_w):04d}', str(int(foreign_p)))
            f_tl_ranges = []
            f_crop_y = 0
            if xml_path_f:
                f_crop_y, f_tl_ranges = parse_cvl_xml(xml_path_f)

            query_rankings[(qw, qp)][lvl] = {
                'top_k': top_k_results,
                'foreign_writer': int(foreign_w),
                'foreign_page': int(foreign_p),
                'replaced_lines': lines_to_replace if lvl > 0 and line_map else set(),
                'textline_ranges': tl_ranges,
                'crop_offset_y': crop_y_q,
                'img_height': img_h,
                'foreign_textline_ranges': f_tl_ranges,
                'foreign_crop_offset_y': f_crop_y,
            }

# ══════════════════════════════════════════════════════════════════════════════
# Generate visualization panels
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating panels...')

THUMB_H = 120  # thumbnail height in pixels

for qi, (qw, qp) in enumerate(query_pages):
    n_levels = len(line_levels)
    n_cols = 1 + args.top_k  # query + top-K
    n_rows = n_levels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    n_writer_pages = len(writer_to_pages[qw])
    fig.suptitle(f'Query: Writer {int(qw):04d} Page {int(qp)} ({n_writer_pages} pages by this writer, {n_writer_pages-1} in gallery)',
                 fontsize=14, fontweight='bold', y=1.01)

    for row, lvl in enumerate(line_levels):
        ranking = query_rankings[(qw, qp)][lvl]
        top_k = ranking['top_k']
        contam_pct = lvl / avg_lines * 100

        # query column
        # query column — splice foreign lines + red border outline
        ax = axes[row, 0]

        replaced_lines = query_rankings[(qw, qp)][lvl].get('replaced_lines', set())
        tl_ranges = query_rankings[(qw, qp)][lvl].get('textline_ranges', [])
        crop_y = query_rankings[(qw, qp)][lvl].get('crop_offset_y', 0)
        img_h_orig = query_rankings[(qw, qp)][lvl].get('img_height', 1)
        fw = query_rankings[(qw, qp)][lvl].get('foreign_writer', 0)
        fp = query_rankings[(qw, qp)][lvl].get('foreign_page', 0)

        # load full-res query and foreign images for splicing
        qimg_full = load_page_image_full(qw, qp)
        if replaced_lines and fw:
            fimg_full = load_page_image_full(fw, fp)
        else:
            fimg_full = None

        # splice foreign lines into query at full resolution
        # splice foreign lines into query at full resolution
        if replaced_lines and fimg_full is not None and tl_ranges:
            # get foreign page's textline ranges
            f_tl_ranges = query_rankings[(qw, qp)][lvl].get('foreign_textline_ranges', [])
            f_crop_y = query_rankings[(qw, qp)][lvl].get('foreign_crop_offset_y', 0)

            # fallback if foreign XML had no handwritten textlines
            if not f_tl_ranges:
                print(f'  WARNING: no foreign textlines for W{int(fw):04d}-{int(fp)}, '
                      f'falling back to Y-proportional copy')
                for li in sorted(replaced_lines):
                    if li < len(tl_ranges):
                        y_min, y_max = tl_ranges[li]
                        ratio_min = (y_min - crop_y) / max(img_h_orig, 1)
                        ratio_max = (y_max - crop_y) / max(img_h_orig, 1)
                        fy1 = int(ratio_min * fimg_full.shape[0])
                        fy2 = int(ratio_max * fimg_full.shape[0])
                        f_tl_ranges.append((fy1 + f_crop_y, fy2 + f_crop_y))

            f_sorted = sorted(f_tl_ranges, key=lambda t: t[0]) if f_tl_ranges else []
            replaced_sorted = sorted(replaced_lines)

            for i, li in enumerate(replaced_sorted):
                if li >= len(tl_ranges):
                    continue
                y_min, y_max = tl_ranges[li]
                # convert to cropped image coords
                dst_y1 = int(y_min - crop_y)
                dst_y2 = int(y_max - crop_y)
                dst_y1 = max(0, min(dst_y1, qimg_full.shape[0]-1))
                dst_y2 = max(0, min(dst_y2, qimg_full.shape[0]-1))
                dst_h = dst_y2 - dst_y1

                if dst_h <= 0:
                    continue

                # pick corresponding foreign line
                if i < len(f_sorted):
                    fy_min, fy_max = f_sorted[i]
                    src_y1 = int(fy_min - f_crop_y)
                    src_y2 = int(fy_max - f_crop_y)
                    src_y1 = max(0, min(src_y1, fimg_full.shape[0]-1))
                    src_y2 = max(0, min(src_y2, fimg_full.shape[0]-1))

                    if src_y2 > src_y1:
                        foreign_strip = fimg_full[src_y1:src_y2, :]
                        # resize to match destination dimensions
                        strip_pil = Image.fromarray(foreign_strip)
                        strip_pil = strip_pil.resize(
                            (qimg_full.shape[1], dst_h), Image.LANCZOS)
                        qimg_full[dst_y1:dst_y2, :] = np.array(strip_pil)[:dst_h, :qimg_full.shape[1]]

        # now resize to thumbnail
        thumb_pil = Image.fromarray(qimg_full)
        ratio = THUMB_H / thumb_pil.height
        thumb_w = int(thumb_pil.width * ratio)
        thumb_pil = thumb_pil.resize((thumb_w, THUMB_H), Image.LANCZOS)
        qimg_thumb = np.array(thumb_pil)

        # convert to RGB
        if qimg_thumb.ndim == 2:
            qimg_rgb = np.stack([qimg_thumb]*3, axis=-1)
        else:
            qimg_rgb = qimg_thumb.copy()

        # draw red border rectangles around replaced lines
        if replaced_lines and tl_ranges:
            scale = THUMB_H / max(img_h_orig, 1)
            border_w = 2
            for li in replaced_lines:
                if li >= len(tl_ranges):
                    continue
                y_min, y_max = tl_ranges[li]
                y1 = int((y_min - crop_y) * scale)
                y2 = int((y_max - crop_y) * scale)
                y1 = max(0, min(y1, THUMB_H-1))
                y2 = max(0, min(y2, THUMB_H-1))
                # top border
                qimg_rgb[max(0,y1-border_w):y1, :] = [255, 0, 0]
                # bottom border
                qimg_rgb[y2:min(THUMB_H,y2+border_w), :] = [255, 0, 0]
                # left border
                qimg_rgb[y1:y2, :border_w] = [255, 0, 0]
                # right border
                qimg_rgb[y1:y2, -border_w:] = [255, 0, 0]

        ax.imshow(qimg_rgb, aspect='auto')
        ax.set_title(f'{lvl} lines\n({contam_pct:.0f}%)', fontsize=8, fontweight='bold')
        ax.axis('off')
        # blue border for query
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#2196F3')
            spine.set_linewidth(3)

        # top-K columns
        for col_idx, result in enumerate(top_k):
            ax = axes[row, 1 + col_idx]
            rimg = load_page_thumbnail(result['writer'], result['page'], max_height=THUMB_H)
            ax.imshow(rimg, cmap='gray', aspect='auto')

            is_correct = result['correct']
            color = '#4CAF50' if is_correct else '#F44336'
            label = f"W{result['writer']:04d}" if result['writer'] != int(qw) else f"W{result['writer']:04d}✓"

            ax.set_title(f'#{col_idx+1} {label}\n{result["similarity"]:.3f}',
                        fontsize=6, color=color, fontweight='bold')
            ax.axis('off')

            # colored border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)

    plt.tight_layout()

    # save to disk
    panel_path = os.path.join(args.out_dir, f'query_{int(qw):04d}_{int(qp)}.png')
    fig.savefig(panel_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {panel_path}')

    # log to wandb
    if wandb is not None:
        wandb.log({
            f'query_{int(qw):04d}_{int(qp)}': wandb.Image(panel_path,
                caption=f'Writer {int(qw):04d} Page {int(qp)} — {n_levels} contamination levels × Top-{args.top_k}')
        })

    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Summary panel: mAP degradation curve
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating summary degradation curve...')
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# compute mAP per level from the query rankings (approximation from selected queries)
for lvl in line_levels:
    correct_counts = []
    for (qw, qp) in query_pages:
        top_k = query_rankings[(qw, qp)][lvl]['top_k']
        n_correct = sum(1 for r in top_k if r['correct'])
        correct_counts.append(n_correct)

contam_pcts = [lvl / avg_lines * 100 for lvl in line_levels]
# compute precision@K per level
prec_at_k = []
for lvl in line_levels:
    precisions = []
    for (qw, qp) in query_pages:
        top_k = query_rankings[(qw, qp)][lvl]['top_k']
        n_correct = sum(1 for r in top_k if r['correct'])
        precisions.append(n_correct / len(top_k))
    prec_at_k.append(np.mean(precisions))

ax.plot(contam_pcts, prec_at_k, 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.fill_between(contam_pcts, prec_at_k, alpha=0.15, color='#2196F3')
ax.set_xlabel('Contamination (%)', fontsize=12)
ax.set_ylabel(f'Precision@{args.top_k}', fontsize=12)
ax.set_title(f'Retrieval Degradation Under Line Contamination (Top-{args.top_k})', fontsize=13)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# annotate
for i, (pct, prec) in enumerate(zip(contam_pcts, prec_at_k)):
    ax.annotate(f'{prec:.2f}', (pct, prec), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

summary_path = os.path.join(args.out_dir, 'degradation_curve.png')
fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {summary_path}')

if wandb is not None:
    wandb.log({'degradation_curve': wandb.Image(summary_path)})
    wandb.finish()

plt.close(fig)

# save rankings for later use
pickle.dump({
    'query_rankings': {f'{w}_{p}': v for (w, p), v in query_rankings.items()},
    'query_pages': [(int(w), int(p)) for w, p in query_pages],
    'line_levels': line_levels,
    'top_k': args.top_k,
}, open(os.path.join(args.out_dir, 'rankings.pkl'), 'wb'))

print(f'\nDone! All outputs in {args.out_dir}')
