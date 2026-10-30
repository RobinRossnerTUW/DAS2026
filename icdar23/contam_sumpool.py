"""
contam_sumpool.py
─────────────────
Minimal contamination eval: SumPooling only, no HDF5 needed.

Patches of every test page are embedded once with the experiment's model and
kept in RAM as one float16 array (rows ordered by patch_idx, so they line up with
the coordinates.json line maps). Because SumPooling is sum -> l2, a contaminated
page descriptor is built from (float32) sums of patch rows instead of
materialising the mixed patch matrix. Foreign-page / patch / line selection is
the unchanged selection.py logic.

One process embeds once, then runs every requested (mode, gallery) combination;
(mode, trial) jobs are spread over --workers forked CPU processes that share the
embeddings copy-on-write.

The foreign source page is dropped from the own-writer ranking by default (it holds exact
copies of the query's foreign patches); --no_exclude_source restores the old behaviour.

--foreign_map additionally scores each contaminated query against the pages of
the writer its foreign content came from (results.calc_foreign_map, source page
excluded from relevance and ranking), i.e. the foreign-writer mAP of contam_main*.py.

Galleries: 'contaminated' (default) = queries and gallery are the same contaminated
set, so relevant gallery pages are contaminated too (foreign-writer mAP then peaks
around 50%, ~p(1-p)); 'clean' = contaminated queries vs the clean pages.

Outputs (in <out_dir or exp_dir/contamination/sumpool>/<mode>[_clean_gallery]/):
    trials.csv, summary.csv                   original writer labels
    foreign_trials.csv, foreign_summary.csv   foreign writer labels (--foreign_map)

Usage
─────
    python contam_sumpool.py --dataset cvl   --exp_dir <exp> --modes random lines --gpuid 0 --foreign_map
    python contam_sumpool.py --dataset icdar --exp_dir <exp> --modes random lines \
        --galleries contaminated clean --gpuid 0 --foreign_map --workers 20
"""

import os, re, csv, glob, pickle, argparse
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import torch
from torchvision import transforms
from sklearn.preprocessing import normalize
from tqdm import tqdm

import config          # parses --dataset from argv; selection.py / page_xml.py read their settings from here
import selection
import page_xml
from results import calc_foreign_map

from utils.utils import GPU, seed_everything, load_yaml
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader
from evaluators.retrieval import Retrieval
from page_encodings import SumPooling
from backbone import resnets
from backbone.model import Model

DATASETS = {
    'cvl': {
        # cvl_cluster_32 testset (model's own test patches)
        'patches_dir': '/data/rrossner/datasets/CVL/cvl_test_patches_bin',
        # coordinates.json per page for the patches above, from helpers/make_coords_for_patches.py
        'coords_dir':  '/data/rrossner/bot_dir/projects/DAS2026/cvl_test_patches_bin_coords',
        'page_re':     re.compile(r'(\d+)-(\d+)-cropped'),          # 0052-1-cropped
    },
    'icdar': {
        # icdar2017 testset; the coordinates.json files live next to the patches
        'patches_dir': '/data/rrossner/DAS2026/icdar/test_patches_coords',
        'coords_dir':  '/data/rrossner/DAS2026/icdar/test_patches_coords',
        'page_re':     re.compile(r'(\d+)-IMG_MAX_(\d+)'),          # 1467-IMG_MAX_1447101
    },
}
IDX_RE = re.compile(r'_(\d+)\.png$')                                # patch_idx suffix
KS = range(1, 10)

# Set in the parent before the worker pool is forked (shared copy-on-write).
G = {}


def recall_at_k(labels, distances, ks):
    recalls = {k: [] for k in ks}
    for i in range(len(labels)):
        idxs = np.argsort(distances[i])
        idxs = idxs[idxs != i]
        rel = labels[idxs] == labels[i]
        if rel.sum() == 0:
            continue
        for k in ks:
            recalls[k].append(rel[:k].sum() / rel.sum())
    return {f'R@{k}': float(np.mean(v)) for k, v in recalls.items()}


def embed_pages(model, patches_dir, page_re, batch_size, num_workers, keep=None, out_path=None):
    """
    Returns (feats, slices): feats is one (N, D) float16 array, rows ordered by
    page then patch_idx; slices maps (writer, page) -> (start, end) row range.
    """
    page_files = []
    for d in sorted(os.listdir(patches_dir)):
        m = page_re.fullmatch(d)
        if not m:
            continue
        if keep is not None and not keep((int(m.group(1)), int(m.group(2)))):
            continue
        fps = glob.glob(os.path.join(patches_dir, d, '*.png'))
        if not fps:
            continue
        fps.sort(key=lambda f: int(IDX_RE.search(f).group(1)))
        page_files.append(((int(m.group(1)), int(m.group(2))), fps))

    all_fps = [f for _, fps in page_files for f in fps]
    print(f'Embedding {len(all_fps)} patches from {len(page_files)} pages in {patches_dir}')
    ds = FilepathImageDataset(all_fps, pil_loader, transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    feats, pos = None, 0
    with torch.no_grad():
        for img in tqdm(loader, desc='Embedding patches', mininterval=30):
            f = torch.nn.functional.normalize(model(img.cuda())).cpu().numpy()
            if feats is None:
                feats = (np.lib.format.open_memmap(out_path, 'w+', np.float16, (len(all_fps), f.shape[1]))
                         if out_path else np.empty((len(all_fps), f.shape[1]), dtype=np.float16))
                print(f'Patch descriptor dim {f.shape[1]}, '
                      f'{feats.nbytes / 1e9:.1f} GB in RAM (float16)')
            feats[pos:pos + len(f)] = f
            pos += len(f)

    if out_path:
        feats.flush()
    slices, start = {}, 0
    for wp, fps in page_files:
        slices[wp] = (start, start + len(fps))
        start += len(fps)
    return feats, slices


def _rows(wp):
    s, e = G['slices'][wp]
    return G['feats'][s:e]


def page_sum_random(wp, fwp, sel):
    """Sum of workers._encode_page_job_random's `mixed` matrix."""
    own_sum = G['own_sums'][wp]
    if sel['n_replace'] == 0:
        return own_sum
    own, foreign = _rows(wp), _rows(fwp)
    dest, src = sel['dest_idx'], sel['src_idx']
    mask = (dest < len(own)) & (src < len(foreign))
    return (own_sum - own[dest[mask]].sum(0, dtype=np.float32)
            + foreign[src[mask]].sum(0, dtype=np.float32))


def page_sum_lines(wp, fwp, sel, line_maps, full_cover):
    """Sum of workers._encode_page_job_lines's `mixed` matrix."""
    own_sum = G['own_sums'][wp]
    if sel['n_replace'] == 0:
        return own_sum
    own, foreign = _rows(wp), _rows(fwp)
    line_map, f_line_map = line_maps[wp], line_maps[fwp]
    fgn_idx = [i for fli in sel['foreign_lines']
               for i in f_line_map.get(fli, []) if i < len(foreign)]
    if full_cover[wp]:
        # line map covers every patch exactly once -> kept = all - replaced lines
        rep_idx = [i for li in sel['lines_to_replace'] for i in line_map[li]]
        keep_sum = own_sum - own[rep_idx].sum(0, dtype=np.float32)
        keep_any = len(rep_idx) < len(own)
    else:
        keep_idx = [i for li in sorted(line_map) if li not in sel['lines_to_replace']
                    for i in line_map[li] if i < len(own)]
        keep_sum = own[keep_idx].sum(0, dtype=np.float32) if keep_idx else 0.0
        keep_any = bool(keep_idx)
    if not keep_any and not fgn_idx:
        return own_sum
    s = np.zeros_like(own_sum)
    if keep_any: s = s + keep_sum
    if fgn_idx:  s = s + foreign[fgn_idx].sum(0, dtype=np.float32)
    return s


def encode(sums):
    return normalize(G['pca'].transform(G['pooling'].normalize(np.stack(sums))), axis=1)


def run_job(job):
    """One (mode, trial): all levels x galleries. Runs in a forked worker."""
    mode, trial = job
    a, M = G['args'], G['modes'][mode]
    config.MODE = mode
    valid_wps, page_writers = M['valid_wps'], M['page_writers']

    foreign_map, sels = selection.precompute(
        a.seed + trial * 1009, valid_wps, M['valid_writers'], M['valid_writer_to_pages'],
        page_num_lines=M['page_num_lines'], page_line_map=M['page_line_map'],
        page_num_patches=G['npatch'])
    foreign_writers = np.array([foreign_map[wp][0] for wp in valid_wps])
    excluded = [{M['wp_to_idx'][foreign_map[wp]]} for wp in valid_wps]
    src_idx = np.array([M['wp_to_idx'][foreign_map[wp]] for wp in valid_wps])

    out = {g: ([], []) for g in a.galleries}
    for lvl in a.levels:
        if mode == 'lines':
            sums = [page_sum_lines(wp, foreign_map[wp], sels[wp][lvl], M['page_line_map'], M['full_cover'])
                    for wp in valid_wps]
        else:
            sums = [page_sum_random(wp, foreign_map[wp], sels[wp][lvl]) for wp in valid_wps]
        Q = encode(sums)

        for g in a.galleries:
            if g == 'clean':
                # query i vs clean pages; its own clean page gets the minimum distance so it
                # takes rank 0, which Retrieval treats as "the query itself" and skips
                # (calc_foreign_map ignores the diagonal as well)
                dist = 1 - Q @ M['clean'].T
                np.fill_diagonal(dist, -1.0)
            else:
                dist = 1 - Q @ Q.T

            # own-writer metrics: drop the foreign source page from the ranking (default). It holds
            # exact copies of the query's foreign patches; foreign mAP already excludes it.
            dist_own = dist
            if a.exclude_source:
                dist_own = dist.copy()
                dist_own[np.arange(len(valid_wps)), src_idx] = np.inf
            _, res = Retrieval.calc_map_from_distances(page_writers, dist_own)
            res.update(recall_at_k(page_writers, dist_own, KS))
            for k in KS:
                res[f'top-{k}'] = res[f'soft-{k}']
            meta = {'trial': trial, 'level': lvl, 'realised_pct': M['realised_pct'][lvl],
                    'mean_units_replaced': M['mean_units'][lvl]}
            out[g][0].append({**meta, **{k: float(v) for k, v in res.items()}})
            if a.foreign_map:
                out[g][1].append({**meta, **calc_foreign_map(
                    page_writers, foreign_writers, dist, excluded_per_query=excluded)})
    return mode, trial, out


def save_results(rows, levels, out_dir, prefix, label, extra, keep_metrics=None, write_trials=True):
    """
    Write <prefix>trials.csv (one row per trial x level) and <prefix>summary.csv (mean/std).
    keep_metrics: only these metric columns go into the summary; write_trials=False skips trials.csv.
    """
    rows = sorted(rows, key=lambda r: (r['trial'], levels.index(r['level'])))
    trials_path  = os.path.join(out_dir, f'{prefix}trials.csv')
    summary_path = os.path.join(out_dir, f'{prefix}summary.csv')
    if write_trials:
        with open(trials_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    metrics = [k for k in rows[0] if k not in ('trial', 'level', 'realised_pct', 'mean_units_replaced')]
    if keep_metrics:
        metrics = [k for k in metrics if k in keep_metrics]
    summary = []
    for lvl in levels:
        lr = [r for r in rows if r['level'] == lvl]
        s = {'level': lvl, 'realised_pct': lr[0]['realised_pct'],
             'mean_units_replaced': lr[0]['mean_units_replaced'], 'n_trials': len(lr), **extra}
        for m in metrics:
            vals = [r[m] for r in lr]
            s[f'{m}_mean'], s[f'{m}_std'] = float(np.mean(vals)), float(np.std(vals))
        summary.append(s)
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    print(f'\n=== {label} ===\nlevel  realised%   mAP (mean ± std)')
    for s in summary:
        print(f"{s['level']:>5}  {s['realised_pct']:>8.2f}   {s['map_mean']:.4f} ± {s['map_std']:.4f}")
    print((f'Saved: {trials_path}\n' if write_trials else '') + f'Saved: {summary_path}')


def prepare_mode(mode, all_wps):
    """Evaluable pages + metadata for one contamination mode (runs in the parent)."""
    config.MODE = mode
    page_line_map = page_num_lines = None
    full_cover = {}
    if mode == 'lines':
        page_line_map, page_num_lines, stats = page_xml.build_all_line_maps(all_wps)
        print(f'XML stats: {stats}')
        valid_wps = [wp for wp in all_wps if page_num_lines[wp] >= config.MIN_LINES]
        for wp in valid_wps:
            idx = sorted(i for lst in page_line_map[wp].values() for i in lst)
            full_cover[wp] = idx == list(range(G['npatch'][wp]))
        print(f'Line maps covering every patch exactly once: '
              f'{sum(full_cover.values())} / {len(valid_wps)} pages')
    else:
        valid_wps = list(all_wps)
    print(f'[{mode}] evaluable pages: {len(valid_wps)} / {len(all_wps)}')

    valid_writer_to_pages = defaultdict(list)
    for wp in valid_wps:
        valid_writer_to_pages[wp[0]].append(wp)
    realised_pct, mean_units, _ = selection.compute_realised_pct(
        valid_wps, page_num_lines=page_num_lines, page_num_patches=G['npatch'])
    return {
        'valid_wps': valid_wps, 'valid_writer_to_pages': valid_writer_to_pages,
        'valid_writers': list(valid_writer_to_pages.keys()),
        'page_writers': np.array([wp[0] for wp in valid_wps]),
        'wp_to_idx': {wp: i for i, wp in enumerate(valid_wps)},
        'page_line_map': page_line_map, 'page_num_lines': page_num_lines, 'full_cover': full_cover,
        'realised_pct': realised_pct, 'mean_units': mean_units,
        'clean': encode([G['own_sums'][wp] for wp in valid_wps]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS), default='cvl')
    parser.add_argument('--exp_dir', required=True, help='dir with config.yaml, model.pt, pca.pkl')
    parser.add_argument('--modes', choices=['random', 'lines'], nargs='+', default=None)
    parser.add_argument('--mode', choices=['random', 'lines'], default=None, help='single mode (same as --modes X)')
    parser.add_argument('--galleries', choices=['contaminated', 'clean'], nargs='+', default=['contaminated'])
    parser.add_argument('--clean_gallery', action='store_true', help='same as --galleries clean')
    parser.add_argument('--patches_dir', default=None, help='patch PNGs; default per --dataset')
    parser.add_argument('--coords_dir', default=None,
                        help='lines mode: <page>/coordinates.json matching patch_idx; default per --dataset')
    parser.add_argument('--levels', default=[0, 10, 20, 30, 40, 50], type=int, nargs='+')
    parser.add_argument('--trials', default=20, type=int)
    parser.add_argument('--seed', default=None, type=int, help='default: config.DATASET_CONFIGS[dataset].seed')
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--loader_workers', default=16, type=int)
    parser.add_argument('--workers', default=8, type=int, help='CPU processes for the trial loop')
    parser.add_argument('--gpuid', default='0', type=str)
    parser.add_argument('--foreign_map', action='store_true',
                        help='also evaluate against the foreign writer (source page excluded)')
    parser.add_argument('--no_exclude_source', dest='exclude_source', action='store_false',
                        help='keep the foreign source page in the own-writer ranking (old behaviour)')
    parser.add_argument('--out_dir', default=None,
                        help='base dir for <mode>[_clean_gallery]/ folders; default <exp_dir>/contamination/sumpool')
    args, _ = parser.parse_known_args()

    args.modes = args.modes or ([args.mode] if args.mode else None)
    assert args.modes, 'give --modes (or --mode)'
    if args.clean_gallery:
        args.galleries = ['clean']
    assert config.DATASET == args.dataset, (config.DATASET, args.dataset)   # config parsed the same argv
    ds = DATASETS[args.dataset]
    args.patches_dir = args.patches_dir or ds['patches_dir']
    args.coords_dir = args.coords_dir or ds['coords_dir']
    if args.seed is None:
        args.seed = config.DATASET_CONFIGS[args.dataset]['seed']

    # ── point the shared selection / page_xml modules at this run ─────────────
    config.PCT_LEVELS  = args.levels
    config.BIN_CENTERS = args.levels
    config.PATCHES_DIR = args.coords_dir
    out_base = args.out_dir or os.path.join(args.exp_dir, 'contamination', 'sumpool')
    print(f'dataset={args.dataset} modes={args.modes} galleries={args.galleries} levels={args.levels} '
          f'trials={args.trials} seed={args.seed} foreign_map={args.foreign_map}\nout: {out_base}')

    exp_cfg = load_yaml(os.path.join(args.exp_dir, 'config.yaml'))
    seed_everything(args.seed)
    GPU.set(args.gpuid, 400)

    # ── model + page-level PCA from the experiment ────────────────────────────
    backbone = getattr(resnets, exp_cfg['model']['name'])()
    random = exp_cfg['model'].get('encoding', None) == 'netrvlad'
    model = Model(backbone, dim=64, num_clusters=exp_cfg['model']['num_clusters'], random=random)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, 'model.pt'), map_location='cpu')['model_state_dict'])
    model = model.cuda().eval()

    feats, slices = embed_pages(model, args.patches_dir, ds['page_re'], args.batch_size, args.loader_workers)
    del model; torch.cuda.empty_cache()

    G.update(args=args, feats=feats, slices=slices,
             pca=pickle.load(open(os.path.join(args.exp_dir, 'pca.pkl'), 'rb')),
             pooling=SumPooling('l2', pn_alpha=0.4),
             npatch={wp: e - s for wp, (s, e) in slices.items()})
    G['own_sums'] = {wp: feats[s:e].sum(0, dtype=np.float32) for wp, (s, e) in slices.items()}
    all_wps = list(slices.keys())
    G['modes'] = {m: prepare_mode(m, all_wps) for m in args.modes}
    for m in args.modes:
        print(f"[{m}] clean baseline mAP (level 0 sanity check): "
              f"{Retrieval.calc_map_from_distances(G['modes'][m]['page_writers'], 1 - G['modes'][m]['clean'] @ G['modes'][m]['clean'].T)[1]['map']:.4f}")

    # ── trial loop, (mode, trial) jobs over forked workers ────────────────────
    jobs = [(m, t) for m in args.modes for t in range(args.trials)]
    collected = {(m, g): ([], []) for m in args.modes for g in args.galleries}
    with mp.get_context('fork').Pool(min(args.workers, len(jobs))) as pool:
        for mode, trial, out in tqdm(pool.imap_unordered(run_job, jobs), total=len(jobs), desc='(mode, trial) jobs'):
            for g, (rows, rows_f) in out.items():
                collected[(mode, g)][0].extend(rows)
                collected[(mode, g)][1].extend(rows_f)
            print(f"  done {mode} trial {trial + 1}: " + '  '.join(
                f"{r['level']}%={r['map']:.4f}" for r in out[args.galleries[0]][0]), flush=True)

    # ── write per (mode, gallery) folders ─────────────────────────────────────
    for (mode, g), (rows, rows_f) in collected.items():
        out_dir = os.path.join(out_base, mode + ('_clean_gallery' if g == 'clean' else ''))
        os.makedirs(out_dir, exist_ok=True)
        extra = {'n_pages': len(G['modes'][mode]['valid_wps']), 'patches_dir': args.patches_dir, 'gallery': g,
                 'source_page_excluded': args.exclude_source}
        save_results(rows, args.levels, out_dir, '', f'{mode} / {g} gallery: original writer labels', extra)
        if args.foreign_map:
            save_results(rows_f, args.levels, out_dir, 'foreign_',
                         f'{mode} / {g} gallery: foreign writer labels (source page excluded)', extra)


if __name__ == '__main__':
    main()
