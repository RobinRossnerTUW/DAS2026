"""
prepare_patrick_data.py
───────────────────────
Prepare the MASLOTS Sanskrit hands set (Patrick-data) for patch extraction.

Reads data.json (concatenated JSON objects, each manuscript repeated), assigns
an integer writer id per hand, converts non-grayscale images to grayscale
(written to <root>/gray, originals untouched) and stages every image as a
symlink <writer>_<page>.jpg in <root>/staged, so the patch names carry the
labels. Writes <root>/labels.csv (staged name -> hand, manuscript, source).

Usage
─────
    python helpers/prepare_patrick_data.py --root /data/rrossner/datasets/Patrick-data
"""

import os, json, csv, argparse

import cv2
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def load_records(path):
    """data.json holds several JSON objects back to back; keep one per m-ms-id."""
    raw = open(path, encoding='utf-8').read()
    dec = json.JSONDecoder()
    records, pos = {}, 0
    while True:
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        if pos >= len(raw):
            break
        obj, pos = dec.raw_decode(raw, pos)
        records.setdefault(obj['m-ms-id'], obj)
    return list(records.values())


def resolve(root, rel):
    """Path of an image on disk; data.json has the wrong extension case for some."""
    path = os.path.normpath(os.path.join(root, rel))
    if os.path.exists(path):
        return path
    folder, name = os.path.split(path)
    match = [f for f in os.listdir(folder) if f.lower() == name.lower()]
    assert len(match) == 1, f'cannot resolve {rel}'
    return os.path.join(folder, match[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='stage Patrick-data for patch extraction')
    parser.add_argument('--root', type=str, required=True, help='folder with data.json and data/')
    args = parser.parse_args()

    gray_dir = os.path.join(args.root, 'gray')
    staged_dir = os.path.join(args.root, 'staged')
    os.makedirs(staged_dir, exist_ok=True)
    assert len(os.listdir(staged_dir)) == 0, 'staged dir is not empty'

    writer_ids = {}
    rows = []
    page = 0
    for rec in load_records(os.path.join(args.root, 'data.json')):
        for entry in rec['m-images']:
            assert len(entry['hands']) == 1, f"{entry['img']} has {len(entry['hands'])} hands"
            hand = entry['hands'][0]
            writer = writer_ids.setdefault(hand, len(writer_ids))
            rows.append({'staged': f'{writer}_{page:04d}.jpg', 'writer': writer, 'page': page,
                         'hand': hand, 'ms_id': rec['m-ms-id'],
                         'source': resolve(args.root, entry['img'])})
            page += 1

    n_converted = 0
    for row in tqdm(rows, 'Staging'):
        src = row['source']
        with Image.open(src) as im:
            is_gray = im.mode == 'L'
        if not is_gray:
            rel = os.path.splitext(os.path.relpath(src, os.path.join(args.root, 'data')))[0] + '.jpg'
            dst = os.path.join(gray_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                tmp = dst + '.tmp.jpg'
                cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                os.replace(tmp, dst)
            src = dst
            n_converted += 1
        row['gray'] = int(not is_gray)
        row['staged_target'] = src
        os.symlink(src, os.path.join(staged_dir, row['staged']))

    with open(os.path.join(args.root, 'labels.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['staged', 'writer', 'page', 'hand', 'ms_id', 'gray',
                                          'source', 'staged_target'])
        w.writeheader()
        w.writerows(rows)

    print(f'{len(rows)} images staged, {n_converted} converted to grayscale')
    for hand, writer in writer_ids.items():
        print(f'  writer {writer}: {hand} ({sum(r["writer"] == writer for r in rows)} images)')
