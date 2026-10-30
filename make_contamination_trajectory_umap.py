#!/usr/bin/env python3
"""
make_contamination_trajectory_umap.py
──────────────────────────────────────
Visualises how contaminating a page moves it through UMAP embedding space.

QUICK START
  1. Edit the CONFIG block below (all paths and settings live there).
  2. Run:  python make_contamination_trajectory_umap.py
  3. Download the whole OUT_DIR folder to your local machine.
  4. Open contamination_trajectory.html in any browser (file:// works, no server needed).

CLI args still work and override CONFIG values if provided.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit this block, then just run:  python make_contamination_trajectory_umap.py
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = dict(

    # ── Model / experiment artefacts ─────────────────────────────────────────

    # HDF5 of raw patch embeddings
    h5_path     = "/data/rrossner/datasets/CVL/7663856/raw_patches_test.h5",

    # Pickle listing (writer, page, hdf5_key, patch_start, patch_end) per page
    index_path  = "/data/rrossner/datasets/CVL/7663856/patch_index_test.pkl",

    # Pre-computed page-level embeddings  (N, D) — used as UMAP background
    embed_path  = "/data/rrossner/datasets/CVL/7663856/embeddings_SumPooling-PN0p4_512.npy",

    # Writer IDs matching embed_path row-for-row  (N,)
    writer_path = "/data/rrossner/datasets/CVL/7663856/writers_SumPooling-PN0p4_512.npy",

    # Fitted sklearn PCA saved with pickle
    pca_path    = "/data/rrossner/datasets/CVL/7663856/pca.pkl",

    # ── Dataset layout ────────────────────────────────────────────────────────

    # Patch coordinates — point to the .tar archive OR to the already-extracted dir.
    # The script auto-extracts archives on first run into out_dir/_extracted/.
    # Archive:   cvl_test_patches_7663856.tar
    # Extracted: a dir containing {writer:04d}-{page}-cropped/coordinates.json
    patches_dir = "/data/rrossner/datasets/CVL/7663856/cvl_test_patches_7663856.tar",

    # XML annotations — point to the .zip archive OR to the already-extracted dir.
    # Archive:   cvl-database-1-1.zip  (contains *_attributes.xml files)
    xml_dir     = "/data/rrossner/datasets/CVL/cvl-database-1-1.zip",

    # Full-page images — point to the .zip archive OR to the already-extracted dir.
    # Archive:   cvl-database-cropped-1-1.zip  (contains {writer:04d}-{page}-cropped.tif)
    pages_dir   = "/data/rrossner/datasets/CVL/cvl-database-cropped-1-1.zip",

    # ── Output ────────────────────────────────────────────────────────────────

    # All outputs go here: contamination_trajectory.html  +  images/  +  trajectory_data.json
    # Download the whole folder; open the HTML locally with file://
    out_dir     = "/data/rrossner/DAS2026/umap",

    # ── Focal page selection ──────────────────────────────────────────────────

    # Option A — pick N pages automatically (one per randomly chosen writer)
    n_focal     = 5,

    # Option B — pin specific pages as list of (writer, page) int tuples.
    #            Set to None to use n_focal instead.
    # Example:   focal_pages = [(1001, 42), (1002, 17), (1003, 88)]
    focal_pages = [(205, 1), (213, 2), (324, 2), (168, 4)],

    # ── Contamination settings ────────────────────────────────────────────────

    # Max lines to replace; levels 0 → max_lines are all shown as trajectory points
    max_lines      = 4,

    # Fix the foreign writer (int ID) so all focal pages borrow from the same writer.
    # Set to None to pick a different random foreign writer per focal page.
    foreign_writer = None,

    # Which pooling to use when re-encoding contaminated patches.
    # Options: "SumPooling-PN0p4" | "SumPooling-PNsqrt" | "MaxPooling" | "LSEPooling"
    pooling        = "SumPooling-PN0p4",

    # ── Misc ──────────────────────────────────────────────────────────────────
    thumb_size  = 100,  # max thumbnail px
    seed        = 42,
)
# ══════════════════════════════════════════════════════════════════════════════
#  END CONFIG
# ══════════════════════════════════════════════════════════════════════════════

import os, sys, json, pickle, random, re, argparse, glob
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
from page_encodings import SumPooling, GMP, MaxPooling, LSEPooling

# ── CLI (overrides CONFIG when provided) ──────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Contamination trajectory UMAP. "
                "Edit CONFIG at the top of this file, or pass CLI args to override.")
for k in ["h5_path","index_path","embed_path","writer_path","pca_path",
          "patches_dir","xml_dir","pages_dir","out_dir","pooling"]:
    parser.add_argument(f"--{k}")
parser.add_argument("--focal_pages",    help="e.g. '1001:42,1002:17'")
parser.add_argument("--n_focal",        type=int)
parser.add_argument("--foreign_writer", type=int)
parser.add_argument("--max_lines",      type=int)
parser.add_argument("--thumb_size",     type=int)
parser.add_argument("--seed",           type=int)
cli = parser.parse_args()

def C(key):
    v = getattr(cli, key, None)
    return v if v is not None else CONFIG[key]

h5_path        = C("h5_path")
index_path     = C("index_path")
embed_path     = C("embed_path")
writer_path    = C("writer_path")
pca_path       = C("pca_path")
patches_dir    = C("patches_dir")
xml_dir        = C("xml_dir")
pages_dir      = C("pages_dir")
out_dir        = C("out_dir")
max_lines      = C("max_lines")
thumb_size     = C("thumb_size")
pooling_name   = C("pooling")
seed           = C("seed")
foreign_writer = C("foreign_writer")
n_focal        = C("n_focal")

if cli.focal_pages:
    focal_pages = [(int(t.split(":")[0]), int(t.split(":")[1]))
                   for t in cli.focal_pages.split(",")]
else:
    focal_pages = CONFIG["focal_pages"]   # list of (w,p) tuples or None

random.seed(seed)
np.random.seed(seed)
os.makedirs(out_dir, exist_ok=True)
THUMB_DIR = os.path.join(out_dir, "images")
os.makedirs(THUMB_DIR, exist_ok=True)

# ── Load archives into memory (zero disk extraction, zero inodes) ────────────
import tarfile, zipfile, io

def _load_coords_from_tar(tar_path):
    """Read all coordinates.json files from a tar into memory.
    Returns dict: (writer_int, page_int) -> parsed json dict.
    """
    cache = {}
    print(f"  Reading coordinates.json from {os.path.basename(tar_path)} …")
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.name.endswith("coordinates.json"):
                continue
            # path like: something/0196-6-cropped/coordinates.json
            dirname = member.name.split("/")[-2]          # "0196-6-cropped"
            stem    = dirname.replace("-cropped", "")     # "0196-6"
            parts   = stem.split("-")
            if len(parts) != 2:
                continue
            try:
                w, p = int(parts[0]), int(parts[1])
                fobj = tf.extractfile(member)
                if fobj:
                    cache[(w, p)] = json.load(fobj)
            except (ValueError, json.JSONDecodeError):
                pass
    print(f"    {len(cache)} coordinates.json entries loaded")
    return cache

def _load_xml_from_zip(zip_path):
    """Read all XML files from a zip into memory.
    Returns dict: basename -> bytes.
    """
    cache = {}
    print(f"  Reading XML files from {os.path.basename(zip_path)} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".xml"):
                cache[os.path.basename(name)] = zf.read(name)
    print(f"    {len(cache)} XML files loaded")
    return cache

def _open_pages_zip(zip_path):
    """Return an open ZipFile for on-demand image reading. Caller must close."""
    return zipfile.ZipFile(zip_path, "r")

def _load_coords_from_dir(dir_path):
    """Fallback: build same-shaped cache from an already-extracted directory."""
    cache = {}
    for root, _, files in os.walk(dir_path):
        if "coordinates.json" in files:
            dirname = os.path.basename(root)
            stem    = dirname.replace("-cropped", "")
            parts   = stem.split("-")
            if len(parts) == 2:
                try:
                    w, p = int(parts[0]), int(parts[1])
                    with open(os.path.join(root, "coordinates.json")) as f:
                        cache[(w, p)] = json.load(f)
                except (ValueError, json.JSONDecodeError):
                    pass
    return cache

def _load_xml_from_dir(dir_path):
    """Fallback: build same-shaped cache from an already-extracted directory."""
    cache = {}
    for root, _, files in os.walk(dir_path):
        for fn in files:
            if fn.endswith(".xml"):
                with open(os.path.join(root, fn), "rb") as f:
                    cache[fn] = f.read()
    return cache

# ── Validate all input paths ──────────────────────────────────────────────────
missing = [(lbl, p) for lbl, p in [
    ("h5_path",     h5_path),    ("index_path",  index_path),
    ("embed_path",  embed_path), ("writer_path", writer_path),
    ("pca_path",    pca_path),   ("patches_dir", patches_dir),
    ("xml_dir",     xml_dir),    ("pages_dir",   pages_dir),
] if not os.path.exists(p)]
if missing:
    print("ERROR — these paths don't exist (update CONFIG):")
    for lbl, p in missing: print(f"  {lbl}: {p}")
    sys.exit(1)

# ── Build in-memory caches ────────────────────────────────────────────────────
print("Loading dataset into memory …")

coords_cache = (_load_coords_from_tar(patches_dir)
                if patches_dir.endswith((".tar", ".tar.gz", ".tgz"))
                else _load_coords_from_dir(patches_dir))

xml_cache = (_load_xml_from_zip(xml_dir)
             if xml_dir.endswith(".zip")
             else _load_xml_from_dir(xml_dir))

# pages_dir: keep as path string; thumbnails are generated later on-demand
pages_is_zip = pages_dir.endswith(".zip")
if pages_is_zip:
    _pages_zf = zipfile.ZipFile(pages_dir, "r")
    # Build name lookup: (writer, page) -> zip entry name
    pages_zip_lookup = {}
    for name in _pages_zf.namelist():
        bn = os.path.basename(name)
        if bn.endswith("-cropped.tif"):
            stem  = bn.replace("-cropped.tif", "")
            parts = stem.split("-")
            if len(parts) == 2:
                try: pages_zip_lookup[(int(parts[0]), int(parts[1]))] = name
                except ValueError: pass
    print(f"  {len(pages_zip_lookup)} page images indexed from zip")

POOLING_MAP = {
    "SumPooling-PN0p4":  SumPooling("l2", pn_alpha=0.4),
    "SumPooling-PNsqrt": SumPooling("powernorm"),
    "MaxPooling":        MaxPooling("l2"),
    "LSEPooling":        LSEPooling("l2"),
}
pooling = POOLING_MAP[pooling_name]

# ── Load background embeddings ────────────────────────────────────────────────
print("Loading background embeddings …")
bg_encodings  = np.load(embed_path)
bg_writer_ids = np.load(writer_path)
print(f"  {bg_encodings.shape[0]} pages, dim={bg_encodings.shape[1]}")

# ── Load index + PCA ──────────────────────────────────────────────────────────
print("Loading index and PCA …")
with open(index_path, "rb") as f:
    index = pickle.load(f)
pca = pickle.load(open(pca_path, "rb"))

wp_to_key       = {(e["writer"], e["page"]): e["hdf5_key"] for e in index}
writer_to_pages = defaultdict(list)
for e in index:
    writer_to_pages[e["writer"]].append((e["writer"], e["page"]))
all_writers = list(writer_to_pages.keys())
all_wps     = list(wp_to_key.keys())

# ── Build patch_num → H5 row mapping ─────────────────────────────────────────
# H5 rows are in lexicographic filename order (_0,_1,_10,_100,...), but
# patch_idx in coordinates.json is the numerical patch number from the filename.
# patch_idx=9 → H5 row 1739, patch_idx=10 → H5 row 2, etc.
# We must translate before indexing into h5f[key][:].
print("Building patch_num → H5 row mappings from index filepaths …")
wp_to_h5map = {}   # (w,p) -> {patch_num: h5_row}
n_mapped = 0
for e in index:
    fps = e.get("filepaths", [])
    if not fps:
        continue
    mapping = {}
    for h5_row, fp in enumerate(fps):
        try:
            patch_num = int(os.path.basename(fp).rsplit("_", 1)[1].replace(".png", ""))
            mapping[patch_num] = h5_row
        except (ValueError, IndexError):
            pass
    if mapping:
        wp_to_h5map[(e["writer"], e["page"])] = mapping
        n_mapped += 1
print(f"  {n_mapped}/{len(index)} pages have filepath-based H5 mappings")
# Quick sanity: patch_idx 10 should NOT map to h5 row 10
_sample_map = wp_to_h5map.get((213, 2), {})
if _sample_map:
    print(f"  Sanity W213/p2: patch_num 10 → H5 row {_sample_map.get(10,'?')}  "
          f"(correct=2, wrong=10)")

# ── XML helpers — operate on in-memory caches ─────────────────────────────────
def find_xml_bytes(writer_id, text_id):
    """Return raw XML bytes from xml_cache, or None if not found."""
    for pat in [f"{writer_id}-{text_id}.xml",
                f"{writer_id}-{text_id}-cropped.xml",
                f"{int(writer_id):04d}-{int(text_id)}.xml",
                f"{int(writer_id):04d}-{text_id}.xml"]:
        if pat in xml_cache:
            return xml_cache[pat]
    # fuzzy fallback
    for fn, data in xml_cache.items():
        if fn.endswith(".xml") and writer_id in fn and str(text_id) in fn:
            return data
    return None

def parse_cvl_xml_bytes(raw_bytes):
    NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}"
    raw = re.sub(rb"<\?xml[^?]*\?>", b"", raw_bytes, count=1)
    raw = re.sub(rb"[^\x09\x0A\x0D\x20-\xFF]", b"", raw)
    try:   root = ET.fromstring(raw)
    except ET.ParseError: return 0, []
    crop_y = 0
    for tl in root.iter(f"{NS}dkTextLines"):
        cy = tl.get("cropY")
        if cy is not None: crop_y = int(float(cy)); break
    lines = []
    for region in root.iter(f"{NS}AttrRegion"):
        if region.get("attrType") == "2" and region.get("fontType") in ("2", None):
            ys = [float(pt.get("y")) for pt in region.iter(f"{NS}Point") if pt.get("y")]
            if ys: lines.append((min(ys), max(ys)))
    lines.sort(key=lambda t: t[0])
    return crop_y, lines

def assign_patches_to_lines(coords_json, textlines, crop_offset_y, h5map=None):
    """
    Assign each patch to a text line by y-coordinate.
    Returns {line_idx: [h5_row, ...]} — values are H5 row indices, not patch_idx,
    when h5map is provided.  Falls back to raw patch_idx if h5map is None.
    """
    ltop = defaultdict(list)
    for pc in coords_json["patches"]:
        y  = pc["y"] + crop_offset_y
        # Translate patch_idx (numerical filename number) → H5 row (lex order)
        h5_row = h5map[pc["patch_idx"]] if (h5map and pc["patch_idx"] in h5map) else pc["patch_idx"]
        ok = False
        for li, (y1, y2) in enumerate(textlines):
            if y1 <= y <= y2: ltop[li].append(h5_row); ok = True; break
        if not ok and textlines:
            dists = [min(abs(y-y1), abs(y-y2)) for y1, y2 in textlines]
            ltop[int(np.argmin(dists))].append(h5_row)
    return dict(ltop)

def load_line_map(w, p):
    """Load (line_idx -> [h5_row, ...]) with correct H5 row translation."""
    cj = coords_cache.get((w, p)) or coords_cache.get((int(w), int(p)))
    if cj is None:
        return {}, 0
    xb = find_xml_bytes(f"{int(w):04d}", str(int(p)))
    if xb is None:
        return {}, 0
    crop_y, textlines = parse_cvl_xml_bytes(xb)
    h5map = wp_to_h5map.get((w, p)) or wp_to_h5map.get((int(w), int(p)))
    lmap  = assign_patches_to_lines(cj, textlines, crop_y, h5map=h5map)
    return lmap, len(textlines)

# Sanity check on a few pages
_hits = sum(1 for (w, p) in all_wps[:20] if load_line_map(w, p)[1] > 0)
print(f"Cache sanity check: {_hits}/20 sampled pages have valid line maps "
      f"({'OK' if _hits > 15 else 'WARNING — low hit rate, check archive structure'})")

# ── Focal pages ───────────────────────────────────────────────────────────────
if focal_pages:
    focal_wps = list(focal_pages)
else:
    chosen = random.sample(all_writers, min(n_focal, len(all_writers)))
    focal_wps = [random.choice(writer_to_pages[w]) for w in chosen]
print(f"Focal pages: {focal_wps}")

focal_foreign   = {}
focal_n_lines   = {}   # (w,p) -> actual XML line count for this page
focal_max_levels= {}   # (w,p) -> levels to compute = [0, 1, …, n_lines//2]

for (w, p) in focal_wps:
    fw = foreign_writer if foreign_writer is not None else random.choice(
        [ow for ow in all_writers if ow != w])
    fp = random.choice(writer_to_pages[fw])[1]
    focal_foreign[(w, p)] = (fw, fp)

    # Determine per-page line count → half of it = max contamination depth
    _, n_lines = load_line_map(w, p)
    max_lvl    = max(1, n_lines // 2)
    focal_n_lines[(w, p)]    = n_lines
    focal_max_levels[(w, p)] = list(range(max_lvl + 1))   # [0, 1, …, max_lvl]
    print(f"  Focal ({w},{p})  ←foreign—  ({fw},{fp})  "
          f"| {n_lines} lines → levels 0..{max_lvl} (50% contamination)")

# ── Diagnose embedding space ─────────────────────────────────────────────────
print(f"\nDiagnostic — bg_encodings shape: {bg_encodings.shape}")
print(f"Diagnostic — pca expects input dim: {pca.n_features_in_}, "
      f"outputs dim: {pca.n_components_}")

if bg_encodings.shape[1] == pca.n_features_in_:
    print("  bg_encodings is pre-PCA → applying pca.transform + normalize")
    bg_pca = normalize(pca.transform(bg_encodings), axis=1)
else:
    print("  bg_encodings is post-PCA → normalizing only")
    bg_pca = normalize(bg_encodings, axis=1)

bg_wp_to_idx = {(e["writer"], e["page"]): i for i, e in enumerate(index)}

# ── Contaminated embeddings (per-page depth = n_lines // 2) ──────────────────
print("\nGenerating contaminated embeddings (per-page depths) …")
print("  Level 0 taken directly from stored bg_encodings (exact alignment).")
trajectory_embs = {wp: {} for wp in focal_wps}

with h5py.File(h5_path, "r") as h5f:
    for (w, p) in focal_wps:
        levels = focal_max_levels[(w, p)]

        # Level 0 — pull from stored bg embedding for exact alignment
        bg_idx = bg_wp_to_idx.get((w, p))
        if bg_idx is not None:
            trajectory_embs[(w, p)][0] = bg_pca[bg_idx]
        else:
            raw = h5f[wp_to_key[(w, p)]][:].astype(np.float32)
            raw = raw / np.where(np.linalg.norm(raw,axis=1,keepdims=True)<1e-8, 1.0,
                                 np.linalg.norm(raw,axis=1,keepdims=True))
            trajectory_embs[(w, p)][0] = normalize(
                pca.transform(pooling.encode([raw])), axis=1)[0]

        if len(levels) == 1:
            continue   # max_lvl == 0, nothing to contaminate

        patches = h5f[wp_to_key[(w, p)]][:].astype(np.float32)
        norms   = np.linalg.norm(patches, axis=1, keepdims=True)
        patches = patches / np.where(norms < 1e-8, 1.0, norms)

        lmap, _      = load_line_map(w, p)
        sorted_lines = sorted(lmap.keys())

        fw, fp    = focal_foreign[(w, p)]
        f_patches = h5f[wp_to_key[(fw, fp)]][:].astype(np.float32)
        fn        = np.linalg.norm(f_patches, axis=1, keepdims=True)
        f_patches = f_patches / np.where(fn < 1e-8, 1.0, fn)
        f_lmap, _ = load_line_map(fw, fp)
        f_sorted  = sorted(f_lmap.keys())

        for lvl in levels[1:]:   # skip 0, already done
            if not lmap:
                mixed = patches
            else:
                nr     = min(lvl, len(sorted_lines))
                rem    = set(random.sample(sorted_lines, nr))
                keep   = [i for li in sorted_lines if li not in rem
                          for i in lmap[li] if i < len(patches)]
                nfp    = min(nr, len(f_sorted))
                fpicks = random.sample(f_sorted, nfp) if nfp < len(f_sorted) else f_sorted
                forn   = [i for fli in fpicks for i in f_lmap[fli] if i < len(f_patches)]
                mixed  = (np.concatenate([patches[keep], f_patches[forn]])
                          if keep and forn else (patches[keep] if keep else patches))

            emb = normalize(pca.transform(pooling.encode([mixed])), axis=1)
            trajectory_embs[(w, p)][lvl] = emb[0]

print("  Done.")

# ── Diagnostics ───────────────────────────────────────────────────────────────
print("\nEmbedding L2 distances + cosine sim from level 0 (in PCA space):")
for (w, p) in focal_wps:
    levels = focal_max_levels[(w, p)]
    d0 = trajectory_embs[(w, p)][0]
    dists   = [np.linalg.norm(trajectory_embs[(w, p)][l] - d0) for l in levels[1:]]
    cosines = [float(d0 @ trajectory_embs[(w, p)][l])           for l in levels[1:]]
    print(f"  W{w}/p{p} ({focal_n_lines[(w,p)]} lines, max lvl {levels[-1]}): "
          + "  ".join(f"L{l}={d:.3f}(cos {c:.3f})" for l,d,c in zip(levels[1:],dists,cosines)))

# ── Per-page AP for each focal page at each contamination level ──────────────
print("\nComputing per-page AP (query = contaminated embedding, gallery = all bg pages) …")
bg_page_writers_arr = np.array([e["writer"] for e in index])

def compute_ap(query_emb, gallery_embs, gallery_writers, query_writer, exclude_idx=None):
    """AP for a single query against a fixed gallery."""
    sims = gallery_embs @ query_emb          # cosine sim (galleries already L2-normed)
    if exclude_idx is not None:
        sims[exclude_idx] = -2.0             # push query page to bottom
    order    = np.argsort(-sims)
    relevant = gallery_writers[order] == query_writer
    n_rel    = int(relevant.sum())
    if n_rel == 0:
        return 0.0
    hits, prec_sum = 0, 0.0
    for rank, is_rel in enumerate(relevant, 1):
        if is_rel:
            hits += 1
            prec_sum += hits / rank
    return prec_sum / n_rel

focal_ap = {}
for (w, p) in focal_wps:
    focal_ap[(w, p)] = {}
    bg_idx = bg_wp_to_idx.get((w, p))
    levels = focal_max_levels[(w, p)]
    for lvl in levels:
        ap = compute_ap(trajectory_embs[(w, p)][lvl],
                        bg_pca, bg_page_writers_arr, w,
                        exclude_idx=bg_idx)
        focal_ap[(w, p)][lvl] = round(float(ap), 4)
    ap_str = "  ".join(f"L{l}={focal_ap[(w,p)][l]:.3f}" for l in levels)
    drop   = focal_ap[(w,p)][0] - focal_ap[(w,p)][levels[-1]]
    print(f"  W{w}/p{p}: {ap_str}  (drop={drop:+.3f})")

traj_rows, focal_meta = [], []
for fi, (w, p) in enumerate(focal_wps):
    for lvl in focal_max_levels[(w, p)]:
        traj_rows.append(trajectory_embs[(w, p)][lvl])
        focal_meta.append({"focal_idx": fi, "writer": w, "page": p,
                           "level": lvl,
                           "n_lines": focal_n_lines[(w, p)],
                           "max_level": focal_max_levels[(w, p)][-1],
                           "pct": round(100 * lvl / focal_n_lines[(w, p)]) if focal_n_lines[(w, p)] else 0,
                           "ap": focal_ap[(w, p)][lvl]})

traj_matrix = np.stack(traj_rows)
all_matrix  = np.vstack([bg_pca, traj_matrix])
print(f"\nUMAP input: {all_matrix.shape[0]} points "
      f"({bg_pca.shape[0]} bg + {traj_matrix.shape[0]} trajectory across {len(focal_wps)} focal pages)")
print("Running UMAP … (slow step, ~1–5 min)")
reduced = umap.UMAP(random_state=seed, n_neighbors=8, min_dist=0.05).fit_transform(all_matrix)
print("  Done.")

bg_2d   = reduced[:len(bg_pca)]
traj_2d = reduced[len(bg_pca):]

# ── Thumbnails ────────────────────────────────────────────────────────────────
def make_thumb(w, p, dst, size):
    """Generate thumbnail from zip or directory, save to dst. Gray placeholder on miss."""
    if os.path.exists(dst): return
    img_bytes = None
    if pages_is_zip:
        zname = pages_zip_lookup.get((w, p)) or pages_zip_lookup.get((int(w), int(p)))
        if zname:
            try: img_bytes = _pages_zf.read(zname)
            except Exception: pass
    else:
        # directory: scan for matching file
        for ext in (".tif", ".jpg", ".png"):
            fp = os.path.join(pages_dir, f"{int(w):04d}-{int(p)}-cropped{ext}")
            if os.path.isfile(fp):
                with open(fp, "rb") as f: img_bytes = f.read()
                break
    if img_bytes:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img.thumbnail((size, size)); img.save(dst); return
        except Exception: pass
    Image.new("RGB", (size, size), (210, 210, 210)).save(dst)

bg_page_keys = [(e["writer"], e["page"]) for e in index]
assert len(bg_page_keys) == len(bg_encodings)

print("Generating thumbnails …")
bg_thumb_paths = []
for (w, p) in bg_page_keys:
    tname = f"{w:04d}-{p}-bg.png"
    make_thumb(w, p, os.path.join(THUMB_DIR, tname), thumb_size)
    bg_thumb_paths.append(os.path.join("images", tname))

traj_thumb_paths = []
for fm in focal_meta:
    w, p  = fm["writer"], fm["page"]
    tname = f"{w:04d}-{p}-focal-lvl{fm['level']}.png"
    make_thumb(w, p, os.path.join(THUMB_DIR, tname), thumb_size * 2)
    traj_thumb_paths.append(os.path.join("images", tname))

if pages_is_zip:
    _pages_zf.close()

# ── Colours ───────────────────────────────────────────────────────────────────
unique_writers = sorted(set(str(int(w)) for w in bg_writer_ids), key=lambda x: int(x))
cmap_bg = plt.get_cmap("tab20", max(len(unique_writers), 1))
writer_to_hex = {}
for i, w in enumerate(unique_writers):
    r,g,b,_ = cmap_bg(i)
    writer_to_hex[w] = "#{:02X}{:02X}{:02X}".format(int(r*255),int(g*255),int(b*255))
bg_colors = [writer_to_hex[str(int(w))] for w in bg_writer_ids]

focal_hues = plt.get_cmap("Set1", max(len(focal_wps), 1))
cmap_traj  = plt.get_cmap("plasma")

def level_color(lvl, max_lvl):
    """Colour by fraction of contamination: 0 = dark blue, 1.0 = bright yellow."""
    r,g,b,_ = cmap_traj(lvl / max(max_lvl, 1))
    return "#{:02X}{:02X}{:02X}".format(int(r*255),int(g*255),int(b*255))
def focal_outline(fi):
    r,g,b,_ = focal_hues(fi)
    return "#{:02X}{:02X}{:02X}".format(int(r*255),int(g*255),int(b*255))

# ── JSON ─────────────────────────────────────────────────────────────────────
bg_data = [{"x":float(bg_2d[i][0]),"y":float(bg_2d[i][1]),
            "writer":str(int(bg_page_keys[i][0])),"page":int(bg_page_keys[i][1]),
            "color":bg_colors[i],"img":bg_thumb_paths[i],"type":"background"}
           for i in range(len(bg_2d))]

traj_data = []
for i, fm in enumerate(focal_meta):
    fw, fp = focal_foreign[(fm["writer"], fm["page"])]
    traj_data.append({"x":float(traj_2d[i][0]),"y":float(traj_2d[i][1]),
        "writer":fm["writer"],"page":fm["page"],"level":fm["level"],
        "n_lines":fm["n_lines"],"max_level":fm["max_level"],"pct":fm["pct"],
        "ap":fm["ap"],
        "focal_idx":fm["focal_idx"],"foreign_writer":fw,"foreign_page":fp,
        "color":level_color(fm["level"], fm["max_level"]),
        "outline":focal_outline(fm["focal_idx"]),
        "img":traj_thumb_paths[i],"type":"trajectory"})

focal_info = [
    {"writer":int(w),"page":int(p),
     "n_lines":focal_n_lines[(w,p)],
     "max_level":focal_max_levels[(w,p)][-1],
     "foreign_writer":int(focal_foreign[(w,p)][0]),
     "foreign_page":int(focal_foreign[(w,p)][1]),
     "ap_series": [{"level":l, "pct": round(100*l/focal_n_lines[(w,p)]) if focal_n_lines[(w,p)] else 0,
                    "ap": focal_ap[(w,p)][l]}
                   for l in focal_max_levels[(w,p)]]}
    for w,p in focal_wps
]
payload = {"background":bg_data,"trajectories":traj_data,
           "n_focal":len(focal_wps),"focal_info":focal_info}

json_path = os.path.join(out_dir, "trajectory_data.json")
with open(json_path,"w") as f: json.dump(payload, f, indent=2)
print(f"Saved JSON → {json_path}")

# ── Self-contained HTML ───────────────────────────────────────────────────────
html_path   = os.path.join(out_dir, "contamination_trajectory.html")
json_inline = json.dumps(payload)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Contamination Trajectory — UMAP</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500&display=swap');
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  :root{{--bg:#ffffff;--surface:#f5f5f7;--border:#d1d1d6;--text:#1a1a1a;--muted:#6e6e73;--accent:#5856d6}}
  html,body{{width:100%;height:100%;background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',sans-serif;overflow:hidden}}
  #app{{display:flex;width:100%;height:100%}}
  #sidebar{{width:260px;min-width:220px;flex-shrink:0;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}}
  #sidebar h1{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);padding:18px 18px 10px;border-bottom:1px solid var(--border)}}
  .panel{{padding:14px 18px;border-bottom:1px solid var(--border)}}
  .panel label{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);display:block;margin-bottom:8px}}
  #level-legend{{display:flex;flex-direction:column;gap:5px}}
  .legend-row{{display:flex;align-items:center;gap:8px;font-size:12px}}
  .legend-swatch{{width:14px;height:14px;border-radius:50%;flex-shrink:0}}
  #focal-list{{display:flex;flex-direction:column;gap:6px;overflow-y:auto;flex:1;padding:14px 18px}}
  .focal-card{{border:1px solid var(--border);border-radius:6px;padding:10px 12px;cursor:pointer;transition:border-color .15s,background .15s;font-size:11px;font-family:'IBM Plex Mono',monospace}}
  .focal-card:hover{{border-color:var(--accent);background:#20223a}}
  .focal-card.active{{border-color:#fff;background:#1e2030}}
  .focal-card .fw{{font-weight:600;margin-bottom:2px}}
  .focal-card .fmeta{{color:var(--muted)}}
  .ap-chart{{margin-top:7px;display:flex;flex-direction:column;gap:3px}}
  .ap-row{{display:flex;align-items:center;gap:5px;font-size:9px;color:var(--muted)}}
  .ap-bar-wrap{{flex:1;height:5px;background:#1e2030;border-radius:3px;overflow:hidden}}
  .ap-bar{{height:100%;border-radius:3px;transition:width .3s}}
  .ap-val{{width:30px;text-align:right;font-family:'IBM Plex Mono',monospace}}
  #tooltip{{position:fixed;pointer-events:none;display:none;background:#1a1c28;border:1px solid var(--border);border-radius:8px;padding:10px 12px;font-size:12px;box-shadow:0 8px 32px #000a;z-index:999;max-width:220px}}
  #tooltip img{{width:100%;border-radius:4px;margin-bottom:6px;display:block}}
  #tooltip .tt-label{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--text);line-height:1.5}}
  #tooltip .tt-level{{font-size:10px;color:var(--muted);margin-top:3px}}
  #canvas-wrap{{flex:1;position:relative;overflow:hidden}}
  svg{{width:100%;height:100%}}
  #controls{{position:absolute;bottom:18px;right:18px;display:flex;gap:8px}}
  button{{background:var(--surface);border:1px solid var(--border);border-radius:5px;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:11px;padding:6px 12px;cursor:pointer;transition:border-color .15s}}
  button:hover{{border-color:var(--accent)}}
  #info-bar{{position:absolute;top:14px;left:14px;font-size:10px;font-family:'IBM Plex Mono',monospace;color:var(--muted)}}
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h1>Contamination<br>Trajectory</h1>
    <div class="panel"><label>Contamination Level</label><div id="level-legend"></div></div>
    <div class="panel"><label>Focal Pages — click to isolate</label></div>
    <div id="focal-list"></div>
  </div>
  <div id="canvas-wrap">
    <svg id="umap-svg"></svg>
    <div id="info-bar">drag to pan · scroll to zoom · click a focal card to isolate</div>
    <div id="controls">
      <button id="btn-reset">Reset view</button>
      <button id="btn-all">Show all</button>
    </div>
  </div>
</div>
<div id="tooltip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const DATA={json_inline};
const bg=DATA.background, traj=DATA.trajectories, focalInfo=DATA.focal_info;
const focalOutlines={{}};
traj.forEach(d=>{{ focalOutlines[d.focal_idx]=d.outline; }});

// Legend: plasma ramp 0%→50% with tick marks at 0, 25, 50
const legendEl=document.getElementById("level-legend");
[["0%","#0d0887"],["~25%","#9c179e"],["~50%","#ed7953"]].forEach(([label,col])=>{{
  const row=document.createElement("div");row.className="legend-row";
  row.innerHTML=`<span class="legend-swatch" style="background:${{col}}"></span><span>${{label}} contamination</span>`;
  legendEl.appendChild(row);
}});
const noteRow=document.createElement("div");
noteRow.style="font-size:10px;color:var(--muted);margin-top:4px;line-height:1.4";
noteRow.textContent="Depth = page-specific (n lines ÷ 2)";
legendEl.appendChild(noteRow);

const focalListEl=document.getElementById("focal-list");
let activeFilter=null;
focalInfo.forEach((info,fi)=>{{
  const card=document.createElement("div");card.className="focal-card";card.dataset.fi=fi;
  const col=focalOutlines[fi]||"#fff";
  // Build AP bar chart rows
  const apRows = info.ap_series.map(s => {{
    const pct0 = info.ap_series[0].ap;
    const barW  = Math.round(s.ap * 100);
    const barCol = s.level===0 ? "#7c6af7"
                 : s.ap >= pct0*0.9 ? "#4ade80"
                 : s.ap >= pct0*0.6 ? "#facc15"
                 : "#f87171";
    return `<div class="ap-row">
      <span style="width:22px;font-size:9px">${{s.pct}}%</span>
      <div class="ap-bar-wrap"><div class="ap-bar" style="width:${{barW}}%;background:${{barCol}}"></div></div>
      <span class="ap-val">${{s.ap.toFixed(3)}}</span>
    </div>`;
  }}).join("");
  card.innerHTML=`
    <div class="fw" style="color:${{col}}">▐ Writer ${{info.writer}}  ·  page ${{info.page}}</div>
    <div class="fmeta">${{info.n_lines}} lines · levels 0–${{info.max_level}} · foreign w${{info.foreign_writer}}</div>
    <div class="ap-chart">
      <div class="ap-row" style="margin-bottom:1px">
        <span style="width:22px"></span><span style="flex:1;font-size:9px;color:var(--muted)">AP (query→gallery)</span>
      </div>
      ${{apRows}}
    </div>`;
  card.addEventListener("click",()=>{{
    if(activeFilter===fi){{activeFilter=null;card.classList.remove("active");updateVis();}}
    else{{document.querySelectorAll(".focal-card").forEach(c=>c.classList.remove("active"));card.classList.add("active");activeFilter=fi;updateVis();}}
  }});
  focalListEl.appendChild(card);
}});

const svg=d3.select("#umap-svg");
const wrap=document.getElementById("canvas-wrap");
const W=()=>wrap.clientWidth,H=()=>wrap.clientHeight;
const g=svg.append("g");

svg.append("defs").append("marker").attr("id","arrow")
  .attr("viewBox","0 -5 10 10").attr("refX",8).attr("refY",0)
  .attr("markerWidth",6).attr("markerHeight",6).attr("orient","auto")
  .append("path").attr("d","M0,-5L10,0L0,5").attr("fill","#ffffff55");

const allX=[...bg,...traj].map(d=>d.x),allY=[...bg,...traj].map(d=>d.y);
const pad=0.05;
const xE=d3.extent(allX),yE=d3.extent(allY),xR=xE[1]-xE[0],yR=yE[1]-yE[0];
const xScl=d3.scaleLinear().domain([xE[0]-pad*xR,xE[1]+pad*xR]).range([0,W()]);
const yScl=d3.scaleLinear().domain([yE[0]-pad*yR,yE[1]+pad*yR]).range([H(),0]);

// Base sizes — all rescaling is done relative to these
const BASE = {{ bgR:2.5, trajR0:9, trajR:7, arrowW:1.5, trajSW0:2.5, trajSW:1.8, fontSize:9 }};
let currentK = 1;

function rescale(k) {{
  currentK = k;
  const s = 1 / k;
  bgG.selectAll("circle").attr("r", BASE.bgR * s);
  trajG.selectAll(".tpt").attr("r",  d => (d.level===0 ? BASE.trajR0 : BASE.trajR) * s)
                         .attr("stroke-width", d => (d.level===0 ? BASE.trajSW0 : BASE.trajSW) * s);
  arrG.selectAll(".tarrow").attr("stroke-width", BASE.arrowW * s);
  trajG.selectAll(".tlbl").attr("font-size", BASE.fontSize * s)
                          .attr("y", d => yScl(d.y) + 4 * s);
}}

const zoom=d3.zoom().scaleExtent([0.3,40]).on("zoom", e=>{{
  g.attr("transform", e.transform);
  rescale(e.transform.k);
}});
svg.call(zoom);
document.getElementById("btn-reset").onclick=()=>svg.transition().duration(500)
  .call(zoom.transform, d3.zoomIdentity)
  .on("end", ()=>rescale(1));
document.getElementById("btn-all").onclick=()=>{{activeFilter=null;document.querySelectorAll(".focal-card").forEach(c=>c.classList.remove("active"));updateVis();}};

const bgG=g.append("g");
bgG.selectAll("circle").data(bg).join("circle")
  .attr("cx",d=>xScl(d.x)).attr("cy",d=>yScl(d.y))
  .attr("r",BASE.bgR).attr("fill",d=>d.color).attr("opacity",0.35)
  .on("mouseover",showTip).on("mousemove",moveTip).on("mouseout",hideTip);

const arrG=g.append("g"),trajG=g.append("g");
const byFocal=d3.group(traj,d=>d.focal_idx);
byFocal.forEach((pts,fi)=>{{
  pts.sort((a,b)=>a.level-b.level);
  for(let i=0;i<pts.length-1;i++){{
    arrG.append("line").attr("class","tarrow").attr("data-fi",fi)
      .attr("x1",xScl(pts[i].x)).attr("y1",yScl(pts[i].y))
      .attr("x2",xScl(pts[i+1].x)).attr("y2",yScl(pts[i+1].y))
      .attr("stroke",focalOutlines[fi]||"#fff").attr("stroke-width",BASE.arrowW)
      .attr("stroke-opacity",0.6).attr("marker-end","url(#arrow)");
  }}
  trajG.selectAll(null).data(pts).join("circle").attr("class","tpt").attr("data-fi",d=>d.focal_idx)
    .attr("cx",d=>xScl(d.x)).attr("cy",d=>yScl(d.y))
    .attr("r",d=>d.level===0?BASE.trajR0:BASE.trajR).attr("fill",d=>d.color)
    .attr("stroke",d=>focalOutlines[d.focal_idx]||"#fff")
    .attr("stroke-width",d=>d.level===0?BASE.trajSW0:BASE.trajSW).attr("opacity",0.92)
    .on("mouseover",showTip).on("mousemove",moveTip).on("mouseout",hideTip);
  trajG.selectAll(null).data(pts).join("text").attr("class","tlbl").attr("data-fi",d=>d.focal_idx)
    .attr("x",d=>xScl(d.x)).attr("y",d=>yScl(d.y)+4).attr("text-anchor","middle")
    .attr("fill","#fff").attr("font-size",BASE.fontSize).attr("font-family","IBM Plex Mono,monospace")
    .attr("font-weight","600").attr("pointer-events","none").text(d=>d.level);
}});

function updateVis(){{
  const fi=activeFilter;
  if(fi===null){{
    bgG.selectAll("circle").attr("opacity",0.35);
    arrG.selectAll("[data-fi]").attr("stroke-opacity",0.6);
    trajG.selectAll(".tpt").attr("opacity",0.92);
    trajG.selectAll(".tlbl").attr("opacity",1);
  }}else{{
    bgG.selectAll("circle").attr("opacity",0.1);
    arrG.selectAll("[data-fi]").attr("stroke-opacity",function(){{return+this.dataset.fi===fi?0.85:0.04;}});
    trajG.selectAll(".tpt").attr("opacity",function(){{return+this.dataset.fi===fi?0.95:0.06;}});
    trajG.selectAll(".tlbl").attr("opacity",function(){{return+this.dataset.fi===fi?1:0;}});
  }}
}}

const tip=document.getElementById("tooltip");
function showTip(event,d){{
  let html=d.img?`<img src="${{d.img}}" onerror="this.style.display='none'">`:"";
  if(d.type==="background"){{html+=`<div class="tt-label">Writer ${{d.writer}}<br>Page ${{d.page}}</div>`;}}
  else{{
    const apBar = Math.round(d.ap*100);
    const apCol = d.level===0?"#7c6af7":d.ap>=0.7?"#4ade80":d.ap>=0.4?"#facc15":"#f87171";
    html+=`<div class="tt-label">Writer ${{d.writer}} · Page ${{d.page}}</div>
      <div class="tt-level">
        ${{d.level===0?"Level 0 — original (0%)"
          :`Level ${{d.level}} — ${{d.level}}/${{d.n_lines}} lines (${{d.pct}}%)`}}
        ${{d.level>0?`<br>foreign: w${{d.foreign_writer}}`:""}}<br>
        <span style="font-size:10px">AP: </span>
        <span style="font-size:11px;font-weight:600;color:${{apCol}}">${{d.ap.toFixed(3)}}</span>
        <div style="height:4px;background:#1e2030;border-radius:2px;margin-top:3px;overflow:hidden">
          <div style="height:100%;width:${{apBar}}%;background:${{apCol}};border-radius:2px"></div>
        </div>
      </div>`;
  }}
  tip.innerHTML=html;tip.style.display="block";moveTip(event);
}}
function moveTip(event){{
  const x=event.clientX+14,y=event.clientY+14,tw=tip.offsetWidth,th=tip.offsetHeight;
  tip.style.left=(x+tw>window.innerWidth?x-tw-28:x)+"px";
  tip.style.top=(y+th>window.innerHeight?y-th-28:y)+"px";
}}
function hideTip(){{tip.style.display="none";}}
</script>
</body>
</html>"""

with open(html_path, "w") as f:
    f.write(HTML)

print(f"\n✓  HTML   → {html_path}")
print(f"✓  JSON   → {json_path}")
print(f"✓  Thumbs → {THUMB_DIR}/")
print(f"\nDownload the whole '{out_dir}/' folder and open contamination_trajectory.html")
print("locally with file:// — no server needed.")
print(f"\nSummary: {len(bg_data)} background pages, "
      f"{len(focal_wps)} focal pages × {max_lines+1} levels = {len(traj_data)} trajectory points")
