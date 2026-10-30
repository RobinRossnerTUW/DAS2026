"""
scale_sweep_patches.py
──────────────────────
Compare patch content over several --scale values. For a few random pages that
reached the patch cap, runs the keypoint detection + filtering of
extract_patches_only_color.py at each scale and writes one image per page:
one row per scale, N random patches per row.

Usage
─────
    python helpers/scale_sweep_patches.py \
        --in_dir /data/rrossner/datasets/Patrick-data/staged \
        --patches_dir /data/rrossner/datasets/Patrick-data_patches \
        --out_dir experiments/maslots_skt_scale_sweep
"""

import os, sys, json, glob, argparse, random
from types import SimpleNamespace

import cv2
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_patches_only_color as epc


def patches_for(filename, params, scale, n, seed):
    p = SimpleNamespace(**{**vars(params), 'scale': scale})
    epc.args = p                           # calc_features reads the module-level `args`
    tup = epc.calc_features(filename, p)
    img = cv2.imread(filename)
    if scale != -1:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_CUBIC)
    points = [pt for pt, _ in tup]
    picked = random.Random(seed).sample(points, min(n, len(points)))
    rois = [epc.get_image_patch(img, x, y, p) for x, y in picked]
    return len(points), [r for r in rois if r is not None]


def make_sheet(page, scales, results, n, zoom=4, pad=4, label_w=330):
    cell = 32 * zoom
    rows = []
    for scale, (n_kp, rois) in zip(scales, results):
        row = np.full((cell + 2 * pad, label_w + n * (cell + pad), 3), 255, np.uint8)
        cv2.putText(row, f'scale {1.0 if scale == -1 else scale:.2f}', (8, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(row, f'{n_kp} keypoints kept', (8, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 1)
        for i, roi in enumerate(rois):
            x = label_w + i * (cell + pad)
            row[pad:pad + cell, x:x + cell] = cv2.resize(roi, (cell, cell), interpolation=cv2.INTER_NEAREST)
        rows.append(row)
    title = np.full((60, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(title, page, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return np.vstack([title] + rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='patch contact sheets over several scales')
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--patches_dir', type=str, required=True,
                        help='existing patch set, for its parameters and per-page patch counts')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--scales', type=float, nargs='+', default=[-1, 0.75, 0.5, 0.35, 0.25])
    parser.add_argument('--num_pages', type=int, default=5)
    parser.add_argument('--num_patches', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    params = SimpleNamespace(**json.load(open(os.path.join(args.patches_dir, 'db-creation-parameters.json'))))
    files = sorted(glob.glob(os.path.join(args.in_dir, '*.*')))
    full = [f for f in files
            if len(glob.glob(os.path.join(args.patches_dir, os.path.splitext(os.path.basename(f))[0], '*.png')))
            >= params.patches_per_page]
    # one random page per writer (first number of the name), so every hand is covered
    rng = random.Random(args.seed)
    by_writer = {}
    for f in full:
        by_writer.setdefault(os.path.basename(f).split('_')[0], []).append(f)
    pages = [rng.choice(v) for _, v in sorted(by_writer.items())][:args.num_pages]

    jobs = [(f, s) for f in pages for s in args.scales]
    results = Parallel(n_jobs=len(jobs), verbose=10)(
        delayed(patches_for)(f, params, s, args.num_patches, args.seed) for f, s in jobs)

    os.makedirs(args.out_dir, exist_ok=True)
    for i, f in enumerate(pages):
        page = os.path.splitext(os.path.basename(f))[0]
        res = results[i * len(args.scales):(i + 1) * len(args.scales)]
        title = f'{page}  ({os.path.relpath(os.path.realpath(f), os.path.dirname(args.in_dir))})'
        out = os.path.join(args.out_dir, f'{page}.png')
        cv2.imwrite(out, make_sheet(title, args.scales, res, args.num_patches))
        print(out, [r[0] for r in res])
