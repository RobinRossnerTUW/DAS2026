"""Copy of extract_patches_only_color.py with one addition: --scale_csv gives every page its own scale
(csv with columns `staged` = image file name and `scale`, e.g. from char_height_calibration.py scales).
Pages missing in the csv are an error. Without --scale_csv it behaves like the original (--scale)."""
import csv
import shutil
import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
import glob
import logging
import json
import sklearn.preprocessing
from sklearn.decomposition import PCA
from tqdm import tqdm


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

_SCALES = {}


def read_scales(path):
    # plain dict cache: an lru_cache wrapper defined in __main__ cannot be sent to the joblib workers
    if path not in _SCALES:
        _SCALES[path] = {r['staged']: float(r['scale']) for r in csv.DictReader(open(path))}
    return _SCALES[path]


def page_scale(filename, arguments):
    if arguments.scale_csv is None:
        return arguments.scale
    return read_scales(arguments.scale_csv)[os.path.basename(filename)]


def edge_pixels(mask):
    return np.sum(mask == 255)

def calc_features(input_img, arguments):
    img = cv2.imread(input_img)

    scale = page_scale(input_img, arguments)
    if scale != -1:
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(img, 30, 200)

    sift = cv2.SIFT_create(sigma=args.sigma)
    kp = sift.detect(img_gray, None)


    def filter_keypoint(point):
        return point


    new_kp = []
    uniq_kp = []
    seen_kp = set()

    num_pixel = arguments.win_size ** 2 

    for p in kp:
        point = (int(p.pt[0]), int(p.pt[1]))

        roi = get_image_patch(img, point[0], point[1], arguments)
        mask_roi = get_image_patch(mask, point[0], point[1], arguments)

        if roi is None:
            continue

        if arguments.edge_pixels != -1:
            if edge_pixels(mask_roi) < (arguments.edge_pixels * num_pixel):
                continue 

        if not point in seen_kp and filter_keypoint(p):
            new_kp.append(p)
            uniq_kp.append(point)
            seen_kp.add(point)

    _, desc = sift.compute(img_gray, new_kp)

    if not new_kp:
        desc = np.zeros((1,128))
        kp_tupple = [( (int(img.shape[1]/2), int(img.shape[0]/2)), desc)]
        print(f'Nothing found for {input_img}')
        return kp_tupple

    desc = sklearn.preprocessing.normalize(desc, norm='l1')
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    desc = sklearn.preprocessing.normalize(desc, norm='l2')

    kp_tupple = [(p, desc[i]) for i, p in enumerate(uniq_kp)]


    return kp_tupple


def get_image_patch(img, px, py, arguments):
    half_win_size = int(arguments.win_size / 2)
    if not (half_win_size < px < img.shape[1] - half_win_size and half_win_size < py < img.shape[0] - half_win_size):
        return None

    roi = img[py - half_win_size:py + half_win_size, px - half_win_size:px + half_win_size]
    assert roi.shape[:2] == (half_win_size * 2, half_win_size * 2), 'shape of the roi is not (%d,%d). It is (%d,%d)' % \
                                                                (half_win_size * 2, half_win_size * 2,
                                                                 roi.shape[0], roi.shape[1])
    return roi


def extract_patches(filename, tup, args):
    if len(tup) > args.patches_per_page and args.patches_per_page != -1:
        idx = np.linspace(0, len(tup) - 1, args.patches_per_page, dtype=np.int32)
        tup = [tup[i] for i in idx]

    points, _ = zip(*tup)

    img = cv2.imread(filename)

    scale = page_scale(filename, args)
    if scale != -1:
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    page_name = os.path.splitext(os.path.basename(filename))[0]
    page_dir = os.path.join(args.out_dir[0], page_name)

    # ── Save patches + collect coordinates ────────────────────────────────────
    coords = []   # list of {patch_idx, x, y, patch_filename}
    count = 0
    for p in points:
        roi = get_image_patch(img, p[0], p[1], args)
        if roi is None:
            continue

        patch_fname = f'{page_name}_{count}.png'
        out_path = os.path.join(page_dir, patch_fname)
        cv2.imwrite(out_path, roi)

        coords.append({
            'patch_idx': count,
            'x': p[0],
            'y': p[1],
            'filename': patch_fname,
        })
        count += 1

    # ── Write coordinates JSON alongside patches ──────────────────────────────
    coords_path = os.path.join(page_dir, 'coordinates.json')
    with open(coords_path, 'w') as f:
        json.dump({
            'page': page_name,
            'source_image': os.path.basename(filename),
            'image_shape': list(img.shape[:2]),  # [height, width]
            'scale': scale,
            'win_size': args.win_size,
            'num_patches': count,
            'patches': coords,
        }, f)


if __name__ == "__main__":
    import argparse
    import pickle

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    parser = argparse.ArgumentParser(description="extract patches from images")
    parser.add_argument('--in_dir', metavar='in_dir', dest='in_dir', type=str, nargs=1,
                        help='input directory', required=True)
    parser.add_argument('--out_dir', metavar='out_dir', dest='out_dir', type=str, nargs=1,
                        help='output directory', required=True)
    parser.add_argument('--win_size', metavar='win_size', dest='win_size', type=int, nargs='?',
                        help='size of the patch',
                        default=32)
    parser.add_argument('--patches_per_page', metavar='patches_per_page', dest='patches_per_page', type=int, nargs='?',
                        help='maximal number of patches per page (-1 for no limit)',
                        default=-1)
    parser.add_argument('--scale', type=float, help='scale images up or down',
                        default=-1)
    parser.add_argument('--scale_csv', type=str, default=None,
                        help='csv with columns staged (image file name) and scale; overrides --scale')
    parser.add_argument('--sigma', type=float, help='blur factor for SIFT',
                        default=1.6)
    parser.add_argument('--edge_pixels', type=float, help='if more black_pixel_thresh percent of the pixels are black -> discard',
                        default=0.1)
    
    args = parser.parse_args()


    assert os.path.exists(args.in_dir[0]), 'in_dir {} does not exist'.format(args.in_dir[0])


    if not os.path.exists(args.out_dir[0]):
        logging.info('creating directory %s' % args.out_dir[0])
        os.mkdir(args.out_dir[0])

    assert len(os.listdir(args.out_dir[0])) == 0, 'out_dir is not empty'

    assert args.win_size % 2 == 0, 'win_size must be even'

    num_cores = int(multiprocessing.cpu_count() / 2)
    num_cores = 10
    path_to_centers = ''

    def chunks(xs, n):
        n = max(1, n)
        return (xs[i:i+n] for i in range(0, len(xs), n))

    files = [f for f in glob.glob(args.in_dir[0] + '/**/*.*', recursive=True) if os.path.isfile(f) and is_image_file(f)]

    file_lists = list(chunks(files, 5000))
    logging.info(f'{len(list(file_lists))} lists')

    for files in file_lists:
        assert len(files) > 0, 'no images found'
        logging.info('Found {} images'.format(len(files)))

        logging.info('calculating features for images in %s (number of cores:%d)' % (args.in_dir, num_cores))
        results = []

        results = Parallel(n_jobs=num_cores, verbose=0)(delayed(calc_features)(f, args) for f in files)

        logging.info('collecting descriptors')
        desc_list = []
        kp_list = []
        fn_list = []

        for r, f in tqdm(zip(results, files)):
            if len(r) == 0:
                logging.warning('no keypoints found in file {} '.format(f))
            for kp, desc in r:
                kp_list.append(kp)
                desc_list.append(desc)
                fn_list.append(f)

        logging.info('creating labels directories in output directory')
        for fn in set(fn_list):
            f_name = os.path.splitext(os.path.basename(fn))[0]
            if not os.path.exists(os.path.join(args.out_dir[0], str(f_name))):
                os.mkdir(os.path.join(args.out_dir[0], str(f_name)))

        results = Parallel(n_jobs=num_cores, verbose=0)(
            delayed(extract_patches)(filename, tup, args) for filename, tup in zip(files, results))

    config_out_path = os.path.join(args.out_dir[0], 'db-creation-parameters.json')
    logging.info('writing config parameters to {}')
    with open(config_out_path, 'w') as f:
        json.dump(vars(args), f)
    if args.scale_csv is not None:
        shutil.copy(args.scale_csv, os.path.join(args.out_dir[0], 'scales.csv'))

    out_files = [f for f in glob.glob(args.out_dir[0] + '/**/*.*', recursive=True) if os.path.isfile(f) and is_image_file(f)]
    logging.info(f'done - extracted {len(out_files)} patches')

