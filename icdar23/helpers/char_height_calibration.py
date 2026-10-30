"""Per-image scale for Patrick-data so that one character is about --target px high (SIFT win_size).

  survey  auto character height of every page (area-weighted median height of the ink components)
          -> survey.csv, survey_summary.csv
  rulers  per group, crops of 3 pages (small / median / large auto height) with a px ruler and reference
          boxes -> rulers/<group>/*.png and char_height.csv to fill in by hand
  scales  char_height.csv + survey.csv -> scales.csv (one scale per page)
"""
import argparse
import csv
import os
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

LABELS = '/data/rrossner/datasets/Patrick-data/labels.csv'
WIDTH_CLASSES = (2800, 4200, 5400, 7000, 10500)  # px, folio widths found in jinendra-psT-ms
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments', 'maslots_skt_char_height')


def read_pages():
    pages = []
    for r in csv.DictReader(open(LABELS)):
        group = r['ms_id']
        if group == 'psT-ms':
            group = r['hand'].replace('-hand', '')
        pages.append({'staged': r['staged'], 'group': group, 'source': r['source']})
    return pages


def components(gray):
    """Ink components (x, y, w, h, area) that can be a character: not specks, lines or stains."""
    ink = cv2.subtract(cv2.blur(gray, (151, 151)), gray)
    t, _ = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, _, st, _ = cv2.connectedComponentsWithStats((ink > t).astype(np.uint8), connectivity=8)
    st = st[1:]
    w, h, a = st[:, 2], st[:, 3], st[:, 4]
    return st[(a >= 40) & (w < 4 * h) & (h < 4 * w) & (h < gray.shape[0] / 3)]


def estimate(gray):
    """Auto character height: area-weighted quartiles of the component heights (specks carry no weight)."""
    st = components(gray)
    if len(st) < 20:
        return 0.0, 0.0, 0.0, st
    order = np.argsort(st[:, 3])
    h, cum = st[order, 3], np.cumsum(st[order, 4])
    q25, q50, q75 = (float(h[np.searchsorted(cum, f * cum[-1])]) for f in (0.25, 0.5, 0.75))
    return q25, q50, q75, st


def survey_one(page):
    gray = cv2.imread(page['source'], cv2.IMREAD_GRAYSCALE)
    q25, q50, q75, st = estimate(gray)
    if page['group'].startswith('psT'):
        # single-folio crops at mixed resolutions (the dpi tag does not follow it, the image width does)
        page = {**page, 'group': '%s_w%d' % (page['group'], min(WIDTH_CLASSES, key=lambda c: abs(np.log(c / gray.shape[1]))))}
    return {**page, 'width': gray.shape[1], 'height': gray.shape[0], 'auto_h': q50, 'auto_h_q25': q25,
            'auto_h_q75': q75, 'n_comp': len(st), 'auto_h_per_kwidth': round(1000 * q50 / gray.shape[1], 2)}


def survey(args):
    pages = read_pages()
    with Pool(args.workers) as pool:
        rows = pool.map(survey_one, pages, chunksize=4)
    write_csv(os.path.join(OUT, 'survey.csv'), rows)
    by_group = defaultdict(list)
    for r in rows:
        by_group[r['group']].append(r)
    summary = []
    for g, rs in sorted(by_group.items()):
        p = np.array([r['auto_h'] for r in rs if r['auto_h']])
        w = np.array([r['width'] for r in rs])
        pw = np.array([r['auto_h_per_kwidth'] for r in rs if r['auto_h']])
        summary.append({'group': g, 'pages': len(rs), 'failed': len(rs) - len(p),
                        'width_min': w.min(), 'width_med': int(np.median(w)), 'width_max': w.max(),
                        'auto_h_min': p.min(), 'auto_h_p10': np.percentile(p, 10), 'auto_h_med': np.median(p),
                        'auto_h_p90': np.percentile(p, 90), 'auto_h_max': p.max(),
                        'auto_h_cv': round(float(p.std() / p.mean()), 3),
                        'auto_h_per_kwidth_cv': round(float(pw.std() / pw.mean()), 3)})
    write_csv(os.path.join(OUT, 'survey_summary.csv'), summary)
    for s in summary:
        print(s)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def text(img, s, org, scale=0.5, color=(0, 0, 0), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def ruler_image(page, target):
    img = cv2.imread(page['source'], cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, auto_h, _, st = estimate(gray)
    h, w = gray.shape
    ch, cw = min(h, int(6 * auto_h)), min(w, int(11 * auto_h))
    # crop where most character-sized components are
    ok = st[np.abs(st[:, 3] - auto_h) <= 0.3 * auto_h]
    ny, nx = max(1, 2 * h // ch), max(1, 2 * w // cw)
    hist, ye, xe = np.histogram2d(ok[:, 1] + ok[:, 3] / 2, ok[:, 0] + ok[:, 2] / 2,
                                  bins=(ny, nx), range=((0, h), (0, w)))
    dens = hist.copy()  # crop spans about 2x2 cells
    dens[:-1] += hist[1:]
    dens[:, :-1] += hist[:, 1:]
    dens[:-1, :-1] += hist[1:, 1:]
    iy, ix = np.unravel_index(np.argmax(dens), dens.shape)
    x0 = int(np.clip(xe[ix], 0, w - cw))
    y0 = int(np.clip(ye[iy], 0, h - ch))
    crop = img[y0:y0 + ch, x0:x0 + cw]
    ch, cw = crop.shape[:2]
    z = max(1, int(round(1500 / cw)))
    big = cv2.resize(crop, (cw * z, ch * z), interpolation=cv2.INTER_CUBIC)

    # horizontal grid every 10 source px (strong every 50), blended so the ink stays readable
    grid = big.copy()
    for y in range(0, ch + 1, 10):
        strong = y % 50 == 0
        cv2.line(grid, (0, y * z), (cw * z, y * z), (0, 0, 255) if strong else (255, 160, 0), 1)
    big = cv2.addWeighted(grid, 0.45, big, 0.55, 0)

    margin, panel, head = 70, 330, 60
    canvas = np.full((head + ch * z + 20, margin + cw * z + panel, 3), 255, np.uint8)
    canvas[head:head + ch * z, margin:margin + cw * z] = big
    for y in range(0, ch + 1, 5 if z >= 3 else 10):
        yy = head + y * z
        length = 22 if y % 50 == 0 else 14 if y % 10 == 0 else 7
        cv2.line(canvas, (margin - length, yy), (margin, yy), (0, 0, 0), 2 if y % 50 == 0 else 1)
        if y % (10 if z >= 3 else 20) == 0:
            text(canvas, str(y), (4, yy + 5), 0.45, (0, 0, 255) if y % 50 == 0 else (0, 0, 0))
    text(canvas, '%s | %s (%s) | %dx%d px | auto char height %.0f px | crop at x=%d y=%d, zoom x%d'
         % (page['group'], os.path.basename(page['source']), page['staged'], w, h, auto_h, x0, y0, z),
         (8, 22), 0.55)
    text(canvas, 'ruler and grid in ORIGINAL px: orange every 10, red every 50', (8, 46), 0.5, (0, 0, 255))

    # reference boxes at the same zoom
    px0 = margin + cw * z + 20
    text(canvas, 'boxes (orig px -> scale)', (px0, head + 14), 0.5)
    y = head + 30
    sizes = sorted({int(round(f * auto_h / 4) * 4) for f in (0.5, 0.65, 0.8, 1.0, 1.2, 1.5, 1.8)})
    for s in sizes:
        if y + 20 + s * z > canvas.shape[0] - 10 or s * z > panel - 30:
            break
        text(canvas, '%d px -> %.3f' % (s, target / s), (px0, y + 14), 0.5, (0, 110, 0))
        cv2.rectangle(canvas, (px0, y + 20), (px0 + s * z, y + 20 + s * z), (0, 140, 0), 2)
        y += s * z + 34
    return canvas, auto_h


def rulers(args):
    rows = list(csv.DictReader(open(os.path.join(OUT, 'survey.csv'))))
    by_group = defaultdict(list)
    for r in rows:
        if float(r['auto_h']) > 0:
            by_group[r['group']].append(r)
    todo = []
    for g, rs in sorted(by_group.items()):
        rs.sort(key=lambda r: float(r['auto_h']))
        # 5th / 50th / 95th percentile auto height: shows the spread without picking estimator outliers
        seen = set()
        for tag, q in (('small', 0.05), ('median', 0.5), ('large', 0.95)):
            page = rs[int(round(q * (len(rs) - 1)))]
            if page['staged'] not in seen:
                seen.add(page['staged'])
                todo.append((g, tag, page))
    sheet = []
    for g, tag, page in todo:
        canvas, auto_h = ruler_image(page, args.target)
        name = '%s_%s_%s.png' % (g, tag, os.path.splitext(page['staged'])[0])
        path = os.path.join(OUT, 'rulers', g, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, canvas)
        sheet.append({'group': g, 'sample': tag, 'staged': page['staged'], 'ruler_png': name,
                      'auto_h': auto_h, 'char_height_px': ''})
        print(path)
    path = os.path.join(OUT, 'char_height.csv')
    if os.path.exists(path) and not args.force:
        print('kept existing %s (use --force to overwrite)' % path)
    else:
        write_csv(path, sheet)


def scales(args):
    survey_rows = list(csv.DictReader(open(os.path.join(OUT, 'survey.csv'))))
    ratio = defaultdict(list)  # hand-measured / auto character height, a property of the hand
    for r in csv.DictReader(open(os.path.join(OUT, 'char_height.csv'))):
        if not args.auto and r['char_height_px'].strip():
            ratio[r['group']].append(float(r['char_height_px']) / float(r['auto_h']))
    by_group = defaultdict(list)
    for r in survey_rows:
        by_group[r['group']].append(r)
    out = []
    for g, rs in sorted(by_group.items()):
        if args.auto:
            ratio[g] = [1.0]
        if not ratio[g]:
            print('no char height entered for %s, skipped' % g)
            continue
        k = float(np.median(ratio[g]))
        med = float(np.median([float(r['auto_h']) for r in rs if float(r['auto_h']) > 0]))
        for r in rs:
            auto_h, how = float(r['auto_h']), 'page'
            if not (args.per_page_groups and g.startswith(tuple(args.per_page_groups))) or auto_h == 0:
                auto_h, how = med, 'group_median'
            char_h = k * auto_h
            out.append({'staged': r['staged'], 'group': g, 'auto_h': auto_h, 'char_height': round(char_h, 1),
                        'scale': round(args.target / char_h, 4), 'auto_h_from': how})
        print('%s: measured/auto %.3f (%d samples, range %.3f), median scale %.3f'
              % (g, k, len(ratio[g]), float(np.ptp(ratio[g])), args.target / (k * med)))
    write_csv(os.path.join(OUT, 'scales.csv'), out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['survey', 'rulers', 'scales'])
    parser.add_argument('--target', type=int, default=32, help='wanted character height in px (SIFT win_size)')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--force', action='store_true', help='rulers: overwrite an existing char_height.csv')
    parser.add_argument('--auto', action='store_true',
                        help='scales: use the auto height as character height, no hand-measured correction')
    parser.add_argument('--per_page_groups', nargs='*', default=[],
                        help='scales: group prefixes that get one scale per page from its own auto height '
                             '(others get one scale per group)')
    args = parser.parse_args()
    {'survey': survey, 'rulers': rulers, 'scales': scales}[args.step](args)
