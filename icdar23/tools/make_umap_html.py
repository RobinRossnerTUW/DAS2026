#!/usr/bin/env python3
"""
make_umap_html.py
-----------------
Loads page-level embeddings produced by main.py, runs UMAP,
generates thumbnails from FULL PAGE images (not patches),
and produces a self-contained hover HTML.

Page image filename format in the zip:
    {writer}-IMG_MAX_{page}.jpg
e.g. 1001-IMG_MAX_116403.jpg  →  writer=1001, page=116403
"""

import os, json, pickle, subprocess, argparse, glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import umap

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir",   required=True,
                    help="Experiment log dir with embeddings_*.npy + patch_index_test.pkl")
parser.add_argument("--out_dir",   required=True,
                    help="Output dir for umap_data.json, images/, and HTML")
parser.add_argument("--scripts",   required=True,
                    help="Repo root where create_html.py lives")
parser.add_argument("--pages_dir", required=True,
                    help="Dir containing full page jpgs ({writer}-IMG_MAX_{page}.jpg)")
parser.add_argument("--thumb_size", type=int, default=100,
                    help="Max thumbnail dimension in pixels (default: 100)")
args = parser.parse_args()

LOG_DIR     = args.log_dir
OUT_DIR     = args.out_dir
SCRIPTS_DIR = args.scripts
PAGES_DIR   = args.pages_dir
THUMB_SIZE  = args.thumb_size

THUMB_DIR   = os.path.join(OUT_DIR, "images")
OUTPUT_JSON = os.path.join(OUT_DIR, "umap_data.json")
OUTPUT_HTML = os.path.join(OUT_DIR, "WR-umap-hover.html")

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

# ── LOAD EMBEDDINGS & WRITER LABELS ───────────────────────────────────────────
print("Loading embeddings ...")
encodings  = np.load(os.path.join(LOG_DIR, "embeddings_SumPooling-PN0p4_512.npy"))
writer_ids = np.load(os.path.join(LOG_DIR, "writers_SumPooling-PN0p4_512.npy"))
labels     = [str(int(w)) for w in writer_ids]
print(f"  {encodings.shape[0]} pages, dim={encodings.shape[1]}")

# ── BUILD (writer, page) → full page filepath LOOKUP ─────────────────────────
# Filename format: {writer}-IMG_MAX_{page}.jpg
print("Building page image lookup ...")
page_lookup = {}   # (writer_int, page_int) -> filepath
for fp in glob.glob(os.path.join(PAGES_DIR, "*-cropped.tif")):
    name = Path(fp).stem                        # e.g. "0395-4-cropped"
    # strip trailing "-cropped" then split on "-"
    name = name.replace("-cropped", "")         # "0395-4"
    parts = name.split("-")
    if len(parts) != 2:
        continue
    try:
        w, p = int(parts[0]), int(parts[1])
        page_lookup[(w, p)] = fp
    except ValueError:
        continue
print(f"  {len(page_lookup)} page images indexed")

# ── LOAD PATCH INDEX to get (writer, page) per embedding row ─────────────────
print("Loading patch index ...")
patch_index = pickle.load(open(os.path.join(LOG_DIR, "patch_index_test.pkl"), "rb"))

# patch_index[i] has 'writer' and 'page' matching encodings[i]
page_keys = [(entry["writer"], entry["page"]) for entry in patch_index]

assert len(page_keys) == len(encodings), (
    f"Mismatch: {len(page_keys)} entries in patch_index vs "
    f"{len(encodings)} rows in embeddings."
)

# ── GENERATE THUMBNAILS from full page images ─────────────────────────────────
print(f"Generating {len(page_keys)} thumbnails from full page images ...")
missing = 0
thumb_paths = []   # relative path for each point, used in JSON

for (w, p) in page_keys:
    fp = page_lookup.get((w, p))
    thumb_name = f"{w:04d}-{p}-cropped.png"
    thumb_path = os.path.join(THUMB_DIR, thumb_name)
    thumb_paths.append(os.path.join("images", thumb_name))  # relative for file://

    if os.path.exists(thumb_path):
        continue

    if fp is None:
        # shouldn't happen if zip matches test set — create a blank placeholder
        img = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), color=(200, 200, 200))
        img.save(thumb_path)
        missing += 1
        continue

    try:
        img = Image.open(fp).convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        img.save(thumb_path)
    except Exception as e:
        print(f"  WARNING: could not thumbnail {fp}: {e}")

if missing:
    print(f"  WARNING: {missing} pages had no matching image in pages_dir")
print("  thumbnails done")

# ── ASSIGN COLORS (one per unique writer) ─────────────────────────────────────
unique_writers = sorted(set(labels), key=lambda x: int(x))
cmap = plt.get_cmap("tab20", max(len(unique_writers), 1))
writer_to_color = {w: cmap(i) for i, w in enumerate(unique_writers)}

def rgba_to_hex(rgba):
    r, g, b, _ = rgba
    return "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))

colors = [rgba_to_hex(writer_to_color[l]) for l in labels]

# ── RUN UMAP ──────────────────────────────────────────────────────────────────
print("Running UMAP ...")
reducer = umap.UMAP(random_state=42, n_neighbors=12, min_dist=0.1)
reduced = reducer.fit_transform(encodings)
print("  UMAP done")

# ── BUILD JSON ────────────────────────────────────────────────────────────────
data = []
for i in range(len(encodings)):
    w, p = page_keys[i]
    data.append({
        "x":     float(reduced[i][0]),
        "y":     float(reduced[i][1]),
        "label": f"writer {labels[i]}  page {p}",   # richer tooltip
        "color": colors[i],
        "img":   thumb_paths[i],
    })

with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved JSON -> {OUTPUT_JSON}")

# ── CALL create_html.py ───────────────────────────────────────────────────────
create_html = os.path.join(SCRIPTS_DIR, "create_html.py")
subprocess.run([
    "python3", create_html,
    OUTPUT_JSON,
    OUTPUT_HTML,
    "--title", "Writer Retrieval UMAP Visualization",
    "--max-thumb", str(THUMB_SIZE),
], check=True)
print(f"HTML written -> {OUTPUT_HTML}")
print("Open it in a browser (file:// works, no server needed).")
