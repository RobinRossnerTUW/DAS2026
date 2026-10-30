"""
make_coords_for_patches.py
──────────────────────────
Write coordinates.json files for an EXISTING patch set produced by
extract_patches_only.py (which saves patches but no coordinates).

Keypoint detection + filtering + linspace subsampling in extract_patches_only.py
is deterministic, so re-running calc_features with the parameters stored in the
patch set's db-creation-parameters.json reproduces the patch_idx -> (x, y)
mapping exactly. Only JSON files are written (to --out_dir); the patch set is
read-only. Every page is verified by cropping a few patches and comparing them
pixel-for-pixel with the existing PNGs.

Usage
─────
    python helpers/make_coords_for_patches.py \
        --in_dir  /data/rrossner/datasets/CVL/cvl_test_binarized \
        --patches_dir /data/rrossner/datasets/CVL/cvl_test_patches_bin \
        --out_dir /data/rrossner/bot_dir/projects/DAS2026/cvl_test_patches_bin_coords
"""

import os, sys, json, glob, argparse
from types import SimpleNamespace

import cv2
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_patches_only as epo


def coords_for_page(filename, params, patches_dir, out_dir, n_check):
    epo.args = params                      # calc_features reads the module-level `args`
    page_name = os.path.splitext(os.path.basename(filename))[0]
    n_existing = len(glob.glob(os.path.join(patches_dir, page_name, '*.png')))
    try:
        tup = epo.calc_features(filename, params)
    except Exception as e:                 # no keypoints survive filtering -> sift.compute returns None
        print(f'No keypoints for {page_name} ({type(e).__name__}), skipping')
        return page_name, 0, n_existing, 0
    if len(tup) == 0:
        return page_name, 0, n_existing, 0
    if len(tup) > params.patches_per_page and params.patches_per_page != -1:
        idx = np.linspace(0, len(tup) - 1, params.patches_per_page, dtype=np.int32)
        tup = [tup[i] for i in idx]
    points = [p for p, _ in tup]

    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

    coords, count = [], 0
    for p in points:                       # same skip logic as extract_patches()
        if epo.get_image_patch(img, p[0], p[1], params) is None:
            continue
        coords.append({'patch_idx': count, 'x': p[0], 'y': p[1],
                       'filename': f'{page_name}_{count}.png'})
        count += 1

    # verify against the existing patch PNGs
    mismatches = 0
    for c in [coords[i] for i in np.linspace(0, count - 1, min(n_check, count), dtype=int)]:
        ref = cv2.imread(os.path.join(patches_dir, page_name, c['filename']), cv2.IMREAD_GRAYSCALE)
        roi = epo.get_image_patch(img, c['x'], c['y'], params)
        if ref is None or not np.array_equal(ref, roi):
            mismatches += 1

    os.makedirs(os.path.join(out_dir, page_name), exist_ok=True)
    with open(os.path.join(out_dir, page_name, 'coordinates.json'), 'w') as f:
        json.dump({'page': page_name, 'source_image': os.path.basename(filename),
                   'image_shape': list(img.shape[:2]), 'win_size': params.win_size,
                   'num_patches': count, 'patches': coords}, f)
    return page_name, count, n_existing, mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='/data/rrossner/datasets/CVL/cvl_test_binarized', help='binarized page images the patch set was cut from')
    parser.add_argument('--patches_dir', default='/data/rrossner/datasets/CVL/cvl_test_patches_bin', help='existing patch set (has db-creation-parameters.json)')
    parser.add_argument('--out_dir', default='/data/rrossner/bot_dir/projects/DAS2026/cvl_test_patches_bin_coords')
    parser.add_argument('--n_check', default=20, type=int, help='patches verified per page')
    parser.add_argument('--n_jobs', default=16, type=int)
    parser.add_argument('--limit', default=None, type=int, help='only process the first N pages (for testing)')
    a = parser.parse_args()

    params = SimpleNamespace(**json.load(open(os.path.join(a.patches_dir, 'db-creation-parameters.json'))))
    files = sorted(f for f in glob.glob(a.in_dir + '/**/*.*', recursive=True)
                   if os.path.isfile(f) and epo.is_image_file(f))[:a.limit]
    print(f'{len(files)} pages, params: {vars(params)}')

    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(coords_for_page)(f, params, a.patches_dir, a.out_dir, a.n_check) for f in files)

    bad = [r for r in res if r[1] != r[2] or r[3] > 0]
    for name, n, n_ex, mm in bad:
        print(f'MISMATCH {name}: coords={n} existing_pngs={n_ex} pixel_mismatches={mm}/{a.n_check}')
    print(f'\n{len(res) - len(bad)} / {len(res)} pages verified OK -> {a.out_dir}')


if __name__ == '__main__':
    main()
