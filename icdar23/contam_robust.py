"""
contam_robust.py
────────────────
Robust alternatives to plain SumPooling under contamination, on top of the same
machinery as contam_sumpool.py (patches embedded in RAM in patch_idx order,
unchanged selection.py logic + seeds, own + foreign-writer mAP, contaminated /
clean gallery). Runs in venvDAS26 (needs POT for topk_ot).

Method groups (--methods, any combination):
  basic    sumpool_ref    SumPooling reference under the same protocol (own folder, not contamination/sumpool)
           GeoMedian      Weiszfeld geometric median of the patches
           TrimmedMean20  drop the 20% of patches farthest from the page centroid
           KMeansReject   k-means (k=2), keep the majority cluster
           SumPool+SGR    SGR reranking of the SumPooling distance matrix
                          (config from <exp_dir>/eval/baseline_rerank.csv; contaminated gallery only)
  sgr_clean SumPool+SGR with a clean gallery (GPU): one graph per query = the clean pages with the
           query's own clean page replaced by its contaminated descriptor
  trim_eps Trimmed (partial Chamfer) distance on patch-level PCA descriptors, eps list (GPU)
  topk_ot  SumPooling top-k candidates reranked by exact partial OT (POT), m list (CPU pool)

All pooled descriptors use the SAME normalisation as the SumPooling baseline
(L2 only; the baseline's pn_alpha is inert) and the experiment's page PCA, so only
the aggregation step differs. trim_eps / topk_ot use a patch-level PCA
(<exp_dir>/pca_patch.pkl, 128-d whitened, fitted on 200k train patches if missing).

Outputs: <exp_dir>/contamination/<method>/<mode>[_clean_gallery]/
         trials.csv, summary.csv, foreign_trials.csv, foreign_summary.csv
"""

import os, re, csv, glob, pickle, argparse, time
# one BLAS thread per process, set before numpy loads OpenBLAS (config.py sets these too, but after numpy):
# otherwise every pool worker starts ~64 BLAS threads and the matmuls in robust_pooling thrash the machine
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
# keep freed blocks up to 32 MB in the heap instead of returning them to the OS: every topk_ot pair allocates
# ~200 MB of (n, n) cost / plan matrices, and re-faulting that memory in many workers at once is kernel-bound
try:
    import ctypes
    _libc = ctypes.CDLL('libc.so.6')
    _libc.mallopt(-3, 32 * 1024 * 1024)    # M_MMAP_THRESHOLD
    _libc.mallopt(-1, 4 * 1024 ** 3)       # M_TRIM_THRESHOLD
    _libc.mallopt(-2, 256 * 1024 ** 2)     # M_TOP_PAD
except OSError:
    pass
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm

import config
import selection
import contam_sumpool as cs
from contam_sumpool import G
from results import calc_foreign_map
from evaluators.retrieval import Retrieval
from page_encodings import SumPooling
from robust_pooling import GeometricMedianPooling, TrimmedMeanPooling, KMeansRejectionPooling
from reranking.sgr import sgr_reranking
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader
from utils.utils import GPU, seed_everything, load_yaml
from backbone import resnets
from backbone.model import Model

TRAIN_DIRS = {'cvl':   '/data/rrossner/datasets/CVL/cvl_train_patches_c32',
              'icdar': '/data/rrossner/DAS2026/icdar/train_patches'}

# pn_alpha=1.0 -> L2 only, identical to SumPooling('l2', pn_alpha=0.4).normalize (see findings)
POOLINGS = {
    'geomedian':          GeometricMedianPooling(pn_alpha=1.0),
    'trimmedmean20':      TrimmedMeanPooling(trim_frac=0.2, pn_alpha=1.0),
    'kmeans_reject_k2':   KMeansRejectionPooling(k=2, pn_alpha=1.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def mixed_rows(mode, wp, fwp, sel, M):
    """Global row indices (into G['feats']) of the contaminated page; mirrors workers.py."""
    s, e = G['slices'][wp]
    own = np.arange(s, e)
    if sel['n_replace'] == 0:
        return own
    fs, fe = G['slices'][fwp]
    n_for = fe - fs
    if mode == 'random':
        dest, src = sel['dest_idx'], sel['src_idx']
        mask = (dest < len(own)) & (src < n_for)
        rows = own.copy()
        rows[dest[mask]] = fs + src[mask]
        return rows
    lm, flm = M['page_line_map'][wp], M['page_line_map'][fwp]
    keep = [i for li in sorted(lm) if li not in sel['lines_to_replace'] for i in lm[li] if i < len(own)]
    fgn = [i for fli in sel['foreign_lines'] for i in flm.get(fli, []) if i < n_for]
    if not keep and not fgn:
        return own
    return np.concatenate([s + np.asarray(keep, dtype=np.int64), fs + np.asarray(fgn, dtype=np.int64)])


def trial_selection(mode, trial):
    M, a = G['modes'][mode], G['args']
    config.MODE = mode
    fmap, sels = selection.precompute(
        a.seed + trial * 1009, M['valid_wps'], M['valid_writers'], M['valid_writer_to_pages'],
        page_num_lines=M['page_num_lines'], page_line_map=M['page_line_map'], page_num_patches=G['npatch'])
    fw = np.array([fmap[wp][0] for wp in M['valid_wps']])
    excl = [{M['wp_to_idx'][fmap[wp]]} for wp in M['valid_wps']]
    return fmap, sels, fw, excl


def evaluate(mode, dist, fw, excl):
    """
    Own-writer metrics (as contam_sumpool) + foreign-writer mAP for one distance matrix.
    With --exclude_source (default) the foreign source page is also dropped from the ranking for
    the own-writer metrics: it holds exact copies of the query's foreign patches, which patch-level
    distances (trim_eps: source page ranked 1st for 97-100% of queries) turn into a near-zero distance.
    """
    pw = G['modes'][mode]['page_writers']
    if G['args'].exclude_source:
        dist = dist.copy()
        for i, ex in enumerate(excl):
            for j in ex:
                dist[i, j] = np.inf
    _, res = Retrieval.calc_map_from_distances(pw, dist)
    res.update(cs.recall_at_k(pw, dist, cs.KS))
    for k in cs.KS:
        res[f'top-{k}'] = res[f'soft-{k}']
    own = {k: float(v) for k, v in res.items()}
    fgn = calc_foreign_map(pw, fw, dist, excluded_per_query=excl) if G['args'].foreign_map else None
    return own, fgn


def _eval_task(t):
    mode, meta, key, dist, fw, excl = t
    own, fgn = evaluate(mode, dist, fw, excl)
    return key, meta, own, fgn


def meta_for(mode, trial, lvl):
    M = G['modes'][mode]
    return {'trial': trial, 'level': lvl, 'realised_pct': M['realised_pct'][lvl],
            'mean_units_replaced': M['mean_units'][lvl]}


def gallery_dist(Q, C, gallery):
    if gallery == 'clean':
        d = 1 - Q @ C.T
        np.fill_diagonal(d, -1.0)   # own clean page -> rank 0 (skipped by Retrieval / calc_foreign_map)
        return d
    return 1 - Q @ Q.T


# ══════════════════════════════════════════════════════════════════════════════
# basic: robust poolings + SGR (CPU pool over (mode, trial))
# ══════════════════════════════════════════════════════════════════════════════

def _clean_pool_page(wp):
    s, e = G['slices'][wp]
    arr = G['feats'][s:e].astype(np.float32)
    return wp, {m: POOLINGS[m].encode([arr])[0] for m in G['poolings']}


def sgr_numpy(X, k, layers, gamma):
    """
    numpy port of reranking/sgr.py (fixed version). Used inside forked workers:
    calling torch there after the parent initialised torch's OpenMP pool deadlocks.
    Checked against sgr_reranking in the parent before the pool starts (check_sgr_port).
    """
    X = X.astype(np.float32)
    S = X @ X.T
    A = np.exp(-(1.0 - S) ** 2 / gamma)            # build_adjacency_matrix: every column is in init_rank
    for _ in range(layers):
        A = (A + A.T) / 2
        r_k2 = np.argsort(-A, axis=1, kind='stable')[:, 1:k + 1]
        W = np.zeros_like(A)
        np.put_along_axis(W, r_k2, np.take_along_axis(S, r_k2, axis=1), axis=1)
        A = A + W @ A.T                             # gnn_propagate: A_[i] = A[i] + sum_j S[i,j] A[:, j]
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return 1.0 - A @ A.T


def check_sgr_port(X, pw, k, layers, gamma):
    _, d_t = sgr_reranking(torch.tensor(X, dtype=torch.float32), k=k, layer=layers, gamma=gamma)
    d_t, d_n = d_t.numpy(), sgr_numpy(X, k, layers, gamma)
    m_t = Retrieval.calc_map_from_distances(pw, d_t)[1]['map']
    m_n = Retrieval.calc_map_from_distances(pw, d_n)[1]['map']
    print(f'SGR port check: max|d_torch - d_numpy| = {np.abs(d_t - d_n).max():.2e}, mAP torch {m_t:.5f} numpy {m_n:.5f}')
    assert abs(m_t - m_n) < 1e-3, 'numpy SGR port disagrees with reranking/sgr.py'


_BUF = {}


def gather_rows(rows):
    """
    float32 view of G['feats'][rows] in a per-process buffer reused across pages. Allocating a fresh
    ~50 MB array per page made the 48 workers spend ~96% of their time in kernel page faults.
    """
    feats, n = G['feats'], len(rows)
    if 'f32' not in _BUF or len(_BUF['f32']) < n:
        cap = max(n, max(e - s for s, e in G['slices'].values()))
        _BUF['f16'] = np.empty((cap, feats.shape[1]), dtype=feats.dtype)
        _BUF['f32'] = np.empty((cap, feats.shape[1]), dtype=np.float32)
    f16, f32 = _BUF['f16'][:n], _BUF['f32'][:n]
    np.take(feats, rows, axis=0, out=f16)
    f32[...] = f16
    return f32


def _job_logger(mode, trial, lvl):
    """Per-job detail log <out_base>/_progress/basic_<mode>_t<trial>_l<lvl>.log (line-buffered, timestamped)."""
    d = os.path.join(G['out_base'], '_progress')
    os.makedirs(d, exist_ok=True)
    f = open(os.path.join(d, f'basic_{mode}_t{trial:02d}_l{lvl:02d}.log'), 'w', buffering=1)
    t0 = time.time()

    def log(msg):
        f.write(f'{time.strftime("%H:%M:%S")} +{time.time() - t0:7.1f}s pid {os.getpid()} | {msg}\n')
    return log, f


def basic_job(job):
    """One (mode, trial, level): robust poolings + SumPooling ref (+ SGR if the contaminated gallery is run)."""
    mode, trial, lvl = job
    a, M = G['args'], G['modes'][mode]
    t0 = time.time()
    log, logf = _job_logger(mode, trial, lvl)
    log(f'start {mode} trial {trial} level {lvl}')
    fmap, sels, fw, excl = trial_selection(mode, trial)
    log(f'trial_selection done ({time.time() - t0:.1f}s)')
    meta = meta_for(mode, trial, lvl)
    out = []
    if lvl == 0:
        Qs = {m: M[f'clean_{m}'] for m in G['poolings']}
        Qsum = M['clean']
    else:
        # one page at a time: materialising all mixed pages at once is tens of GB per worker
        pooled = {m: [] for m in G['poolings']}
        sums = []
        n = len(M['valid_wps'])
        tm = defaultdict(float)   # cumulative seconds per stage
        for i, wp in enumerate(M['valid_wps']):
            ts = time.perf_counter()
            arr = gather_rows(mixed_rows(mode, wp, fmap[wp], sels[wp][lvl], M))
            tm['gather'] += time.perf_counter() - ts
            for m in G['poolings']:
                ts = time.perf_counter()
                pooled[m].append(POOLINGS[m].encode([arr])[0])
                tm[m] += time.perf_counter() - ts
            sums.append(arr.sum(0))
            if (i + 1) % 50 == 0 or i == 0:
                el = time.time() - t0
                log(f'{i + 1}/{n} pages pooled (last page {arr.shape[0]} patches) | eta job {el / (i + 1) * (n - i - 1) / 60:.1f} min | '
                    + '  '.join(f'{k} {v / (i + 1):.3f}s/page' for k, v in tm.items()))
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f'    [{mode} t{trial} l{lvl}] {i + 1}/{n} pages pooled, {el / 60:.1f} min '
                      f'(eta {el / (i + 1) * (n - i - 1) / 60:.1f} min)', flush=True)
            del arr
        ts = time.time()
        Qs = {m: normalize(G['pca'].transform(np.stack(pooled[m])), axis=1) for m in G['poolings']}
        Qsum = cs.encode(sums)
        log(f'PCA + normalise done ({time.time() - ts:.1f}s)')
    ts = time.time()
    for m in G['poolings']:
        for g in a.galleries:
            own, fgn = evaluate(mode, gallery_dist(Qs[m], M[f'clean_{m}'], g), fw, excl)
            out.append((m, g, meta, own, fgn))
    for g in (a.galleries if a.sumpool_ref else []):   # SumPooling reference under the same protocol
        own, fgn = evaluate(mode, gallery_dist(Qsum, M['clean'], g), fw, excl)
        out.append(('sumpool_ref', g, meta, own, fgn))
    if 'contaminated' in a.galleries:   # SGR here is contaminated-gallery only (clean: --methods sgr_clean)
        own, fgn = evaluate(mode, sgr_numpy(Qsum, a.sgr_k, a.sgr_layers, a.sgr_gamma), fw, excl)
        out.append((G['sgr_name'], 'contaminated', meta, own, fgn))
    log(f'evaluation done ({time.time() - ts:.1f}s) | ' + '  '.join(f'{m}/{g} mAP {o["map"]:.4f}' for m, g, _, o, _ in out))
    log(f'job done ({(time.time() - t0) / 60:.1f} min)')
    logf.close()
    return mode, trial, lvl, out, time.time() - t0


def run_basic(collect):
    a = G['args']
    t0 = time.time()
    with mp.get_context('fork').Pool(a.workers) as pool:
        clean = dict(tqdm(pool.imap_unordered(_clean_pool_page, list(G['slices']), chunksize=8),
                          total=len(G['slices']), desc='clean robust descriptors', mininterval=30))
    print(f'clean robust descriptors: {(time.time() - t0) / 60:.1f} min', flush=True)
    for mode, M in G['modes'].items():
        for m in G['poolings']:
            M[f'clean_{m}'] = normalize(G['pca'].transform(
                np.stack([clean[wp][m] for wp in M['valid_wps']])), axis=1)
            base = Retrieval.calc_map_from_distances(M['page_writers'], 1 - M[f'clean_{m}'] @ M[f'clean_{m}'].T)[1]['map']
            print(f'[{mode}] clean {m} mAP: {base:.4f}  (SumPooling: '
                  f"{Retrieval.calc_map_from_distances(M['page_writers'], 1 - M['clean'] @ M['clean'].T)[1]['map']:.4f})")

    if 'contaminated' in a.galleries:
        M0 = G['modes'][a.modes[0]]
        check_sgr_port(M0['clean'], M0['page_writers'], a.sgr_k, a.sgr_layers, a.sgr_gamma)

    # one job per (mode, trial, level): (mode, trial) jobs left most cores idle with few trials
    jobs = [(m, t, l) for m in a.modes for t in range(a.trials) for l in a.levels]
    jobs.sort(key=lambda j: -j[2])   # heavy (contaminated) levels first, level 0 is cheap
    print(f'basic: {len(jobs)} (mode, trial, level) jobs on {min(a.workers, len(jobs))} workers', flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(min(a.workers, len(jobs))) as pool:
        for i, (mode, trial, lvl, out, dt) in enumerate(pool.imap_unordered(basic_job, jobs), 1):
            for method, g, meta, own, fgn in out:
                collect(method, mode, g, meta, own, fgn)
            maps = '  '.join(f'{method}={own["map"]:.4f}' for method, g, meta, own, fgn in out if g == a.galleries[0])
            el = time.time() - t0
            print(f'  basic done {i}/{len(jobs)}: {mode} trial {trial + 1} level {lvl} ({dt / 60:.1f} min) '
                  f'[{a.galleries[0]}] {maps} | elapsed {el / 60:.1f} min, eta {el / i * (len(jobs) - i) / 60:.1f} min',
                  flush=True)
            if i % 20 == 0:
                G['checkpoint']()


# ══════════════════════════════════════════════════════════════════════════════
# patch-level PCA (for trim_eps / topk_ot)
# ══════════════════════════════════════════════════════════════════════════════

def fit_patch_pca(model, path, n_samples=200_000, seed=0):
    a = G['args']
    files = sorted(glob.glob(os.path.join(TRAIN_DIRS[a.dataset], '**', '*.png'), recursive=True))
    sel = np.random.RandomState(seed).choice(len(files), min(n_samples, len(files)), replace=False)
    fps = [files[i] for i in sel]
    print(f'Fitting patch PCA on {len(fps)} / {len(files)} train patches from {TRAIN_DIRS[a.dataset]}')
    loader = torch.utils.data.DataLoader(FilepathImageDataset(fps, pil_loader, transforms.ToTensor()),
                                         batch_size=a.batch_size, num_workers=a.loader_workers)
    feats = []
    with torch.no_grad():
        for img in tqdm(loader, desc='train patches', mininterval=30):
            feats.append(torch.nn.functional.normalize(model(img.cuda())).cpu().numpy())
    pca = PCA(128, whiten=True).fit(np.concatenate(feats))
    tmp = f'{path}.tmp{os.getpid()}'
    pickle.dump(pca, open(tmp, 'wb'))
    os.replace(tmp, path)
    print(f'Saved patch PCA -> {path}')
    return pca


def build_patch_bank(pca_patch, chunk=200_000):
    feats = G['feats']
    bank = np.empty((len(feats), pca_patch.n_components_), dtype=np.float16)
    for s in range(0, len(feats), chunk):
        bank[s:s + chunk] = normalize(pca_patch.transform(feats[s:s + chunk].astype(np.float32)), axis=1)
    return bank


# ══════════════════════════════════════════════════════════════════════════════
# trim_eps (GPU)
# ══════════════════════════════════════════════════════════════════════════════

def _pad(rows_list, bank):
    lens = [len(r) for r in rows_list]
    N, max_n, D = len(rows_list), max(lens), bank.shape[1]
    pad = torch.zeros(N, max_n, D, dtype=bank.dtype, device=bank.device)
    mask = torch.zeros(N, max_n, dtype=torch.bool, device=bank.device)
    for i, r in enumerate(rows_list):
        pad[i, :lens[i]] = bank[torch.as_tensor(r, device=bank.device)]
        mask[i, :lens[i]] = True
    return pad.reshape(N * max_n, D), mask.reshape(N * max_n), N, max_n


def trimmed_dists(q_rows, g_rows, bank, eps_list, gal_cache=None):
    """d[i, j] = mean of the best (1-eps) nearest-neighbour distances of query page i's patches in gallery page j."""
    flat, fmask, Ng, max_n = gal_cache if gal_cache is not None else _pad(g_rows, bank)
    free = torch.cuda.mem_get_info()[0]
    j_chunk = max(1, min(Ng, int(free * 0.3 // (max(len(r) for r in q_rows) * max_n * 2 * 2))))
    out = {eps: np.zeros((len(q_rows), Ng), dtype=np.float32) for eps in eps_list}
    with torch.no_grad():
        for i, r in enumerate(q_rows):
            q = bank[torch.as_tensor(r, device=bank.device)]
            n_i = q.shape[0]
            best = torch.empty(n_i, Ng, dtype=torch.float32, device=bank.device)
            for j0 in range(0, Ng, j_chunk):
                j1 = min(j0 + j_chunk, Ng)
                sim = q @ flat[j0 * max_n:j1 * max_n].T
                sim.masked_fill_(~fmask[j0 * max_n:j1 * max_n].unsqueeze(0), -1.0)
                sb = sim.view(n_i, j1 - j0, max_n).amax(dim=2).float()
                best[:, j0:j1] = torch.sqrt(torch.clamp(2.0 - 2.0 * sb, min=0.0))
                del sim, sb
            srt = torch.sort(best, dim=0).values
            for eps in eps_list:
                k = max(1, int(np.floor((1.0 - eps) * n_i)))
                out[eps][i] = srt[:k].mean(dim=0).cpu().numpy()
    return out


def run_trim(collect, bank_np):
    a = G['args']
    dev = torch.device('cuda')
    bank = torch.from_numpy(bank_np).to(dev)
    names = {eps: f'trim_eps{eps:g}' for eps in a.trim_eps}
    with mp.get_context('fork').Pool(a.metric_workers) as pool:
        pending = []
        for mode in a.modes:
            M = G['modes'][mode]
            clean_rows = [np.arange(*G['slices'][wp]) for wp in M['valid_wps']]
            gal_clean = _pad(clean_rows, bank)
            level0 = None
            for trial in range(a.trials):
                fmap, sels, fw, excl = trial_selection(mode, trial)
                t0 = time.time()
                for lvl in a.levels:
                    meta = meta_for(mode, trial, lvl)
                    if lvl == 0:
                        if level0 is None:   # uncontaminated: identical for every trial and both galleries
                            level0 = trimmed_dists(clean_rows, None, bank, a.trim_eps, gal_cache=gal_clean)
                        dists = {'contaminated': level0, 'clean': level0}
                    else:
                        q_rows = [mixed_rows(mode, wp, fmap[wp], sels[wp][lvl], M) for wp in M['valid_wps']]
                        dists = {}
                        if 'contaminated' in a.galleries:
                            dists['contaminated'] = trimmed_dists(q_rows, q_rows, bank, a.trim_eps)
                        if 'clean' in a.galleries:
                            dists['clean'] = trimmed_dists(q_rows, None, bank, a.trim_eps, gal_cache=gal_clean)
                    for g in a.galleries:
                        for eps in a.trim_eps:
                            d = dists[g][eps].copy()
                            if g == 'clean':
                                np.fill_diagonal(d, -1.0)
                            pending.append(pool.apply_async(_eval_task, ((mode, meta, (names[eps], mode, g), d, fw, excl),)))
                print(f'  trim_eps done {mode} trial {trial + 1} ({time.time() - t0:.0f}s GPU)', flush=True)
                pending = _drain(pending, collect, block=False)
        _drain(pending, collect, block=True)


# ══════════════════════════════════════════════════════════════════════════════
# sgr_clean (GPU): SumPool+SGR with a clean gallery
# ══════════════════════════════════════════════════════════════════════════════

def _sgr_query_job(job):
    """Contaminated SumPooling descriptors of one (mode, trial): {level > 0: (N, D)}. Runs in a forked worker."""
    mode, trial = job
    a, M = G['args'], G['modes'][mode]
    fmap, sels, fw, excl = trial_selection(mode, trial)
    Qs = {}
    for lvl in a.levels:
        if lvl == 0:
            continue
        if mode == 'lines':
            sums = [cs.page_sum_lines(wp, fmap[wp], sels[wp][lvl], M['page_line_map'], M['full_cover'])
                    for wp in M['valid_wps']]
        else:
            sums = [cs.page_sum_random(wp, fmap[wp], sels[wp][lvl]) for wp in M['valid_wps']]
        Qs[lvl] = cs.encode(sums).astype(np.float32)
    return mode, trial, Qs, fw, excl


def sgr_clean_dists(Q, C, S_cc, k, layers, gamma, batch):
    """
    Row i = SGR distances of contaminated query i to the clean gallery.
    SGR is transductive (one graph over queries = gallery), so every query gets its own graph:
    the clean pages, with page i's clean descriptor replaced by its contaminated one. Neither
    another contaminated page nor the query's own clean page is part of that graph.
    Batched torch version of sgr_numpy; only row i of each graph's distance matrix is kept.
    """
    N = C.shape[0]
    out = torch.empty(N, N, device=C.device)
    with torch.no_grad():
        for s in range(0, N, batch):
            idx = torch.arange(s, min(s + batch, N), device=C.device)
            b = torch.arange(len(idx), device=C.device)
            q = Q[idx] @ C.T
            q[b, idx] = 1.0
            S = S_cc.repeat(len(idx), 1, 1)
            S[b, idx, :] = q
            S[b, :, idx] = q
            A = torch.exp(-(1.0 - S) ** 2 / gamma)
            for _ in range(layers):
                A = (A + A.transpose(1, 2)) / 2
                r_k2 = A.topk(k + 1, dim=-1).indices[..., 1:]
                W = torch.zeros_like(A).scatter_(2, r_k2, S.gather(2, r_k2))
                A = A + W @ A.transpose(1, 2)
                A = A / A.norm(dim=2, keepdim=True)
            out[idx] = 1.0 - (A[b, idx].unsqueeze(1) @ A.transpose(1, 2)).squeeze(1)
            del S, A, W
    return out.cpu().numpy()


def run_sgr_clean(collect):
    a = G['args']
    dev = torch.device('cuda')
    name = G['sgr_name']
    clean = {}
    for mode in a.modes:
        M = G['modes'][mode]
        C = torch.from_numpy(M['clean'].astype(np.float32)).to(dev)
        S_cc = C @ C.T
        # uncontaminated query = clean gallery page: every per-query graph is the plain clean graph
        d0 = sgr_clean_dists(C, C, S_cc, a.sgr_k, a.sgr_layers, a.sgr_gamma, a.sgr_batch)
        _, d_ref = sgr_reranking(torch.tensor(M['clean'], dtype=torch.float32), k=a.sgr_k, layer=a.sgr_layers,
                                 gamma=a.sgr_gamma)
        m_ref = Retrieval.calc_map_from_distances(M['page_writers'], d_ref.numpy())[1]['map']
        m_new = Retrieval.calc_map_from_distances(M['page_writers'], d0)[1]['map']
        print(f'[{mode}] SGR batched check: max|d_ref - d_batched| = {np.abs(d_ref.numpy() - d0).max():.2e}, '
              f'mAP reference {m_ref:.5f} batched {m_new:.5f}', flush=True)
        assert abs(m_ref - m_new) < 1e-3, 'batched SGR disagrees with reranking/sgr.py'
        np.fill_diagonal(d0, -1.0)
        clean[mode] = (C, S_cc, d0)

    jobs = [(m, t) for m in a.modes for t in range(a.trials)]
    with mp.get_context('fork').Pool(min(a.workers, len(jobs))) as q_pool, \
            mp.get_context('fork').Pool(a.metric_workers) as ev_pool:
        pending = []
        for mode, trial, Qs, fw, excl in q_pool.imap(_sgr_query_job, jobs):
            C, S_cc, d0 = clean[mode]
            t0 = time.time()
            for lvl in a.levels:
                if lvl == 0:
                    d = d0
                else:
                    d = sgr_clean_dists(torch.from_numpy(Qs[lvl]).to(dev), C, S_cc,
                                        a.sgr_k, a.sgr_layers, a.sgr_gamma, a.sgr_batch)
                    np.fill_diagonal(d, -1.0)   # own position -> rank 0, skipped (as gallery_dist)
                pending.append(ev_pool.apply_async(
                    _eval_task, ((mode, meta_for(mode, trial, lvl), (name, mode, 'clean'), d, fw, excl),)))
            print(f'  sgr_clean done {mode} trial {trial + 1} ({time.time() - t0:.0f}s GPU)', flush=True)
            pending = _drain(pending, collect, block=False)
            G['checkpoint']()
        _drain(pending, collect, block=True)


def _drain(pending, collect, block):
    rest = []
    for p in pending:
        if block or p.ready():
            (method, mode, g), meta, own, fgn = p.get()
            collect(method, mode, g, meta, own, fgn)
        else:
            rest.append(p)
    return rest


# ══════════════════════════════════════════════════════════════════════════════
# topk_ot (CPU pool of exact partial-OT solves)
# ══════════════════════════════════════════════════════════════════════════════

def _ot_pair(job):
    import ot
    from ot_distance import _DEFAULT_RETRY_SCHEDULE
    key, ra, rb, m_list = job
    Qa = G['pbank'][ra].astype(np.float64)
    Qb = G['pbank'][rb].astype(np.float64)
    Mc = ot.dist(Qa, Qb, metric='sqeuclidean')
    scale = float(Mc.max())
    Mn = Mc / scale if scale > 0 else Mc
    wa, wb = np.full(len(Qa), 1.0 / len(Qa)), np.full(len(Qb), 1.0 / len(Qb))
    out = {}
    for m in m_list:
        gamma = None
        for cfg in _DEFAULT_RETRY_SCHEDULE:
            try:
                gamma = ot.partial.partial_wasserstein(wa, wb, Mn, m=m, **cfg)
                break
            except ValueError:
                continue
        if gamma is None:
            raise RuntimeError(f'partial_wasserstein failed for {key}, m={m}')
        out[m] = float(np.sum(gamma * Mc))    # score with the unnormalised cost, as ot_distance.py
    return key, out


def topk_ot_dists(base, q_rows, g_rows, symmetric, pool, ks, m_list, clean, tag=''):
    """
    Rerank each query's top-k base candidates by partial OT; everything else keeps base order behind them.
    OT is solved once for the top-max(ks) candidates; smaller k reuse those distances (nested candidate sets).
    Returns {(k, m): (N, N) dist}.
    """
    N = base.shape[0]
    cand = {}
    for i in range(N):
        row = base[i].copy(); row[i] = np.inf
        cand[i] = np.argsort(row)[:max(ks)]
    pairs = {(min(i, j), max(i, j)) if symmetric else (i, j) for i in range(N) for j in cand[i]}
    jobs = [((i, j), q_rows[i], g_rows[j], m_list) for i, j in pairs]
    print(f'    {tag} partial OT: {len(jobs)} pairs (top-{max(ks)}, m={m_list}) on {pool._processes} workers', flush=True)
    t0, ot_out, step = time.time(), {}, max(1, len(jobs) // 10)
    for key, o in pool.imap_unordered(_ot_pair, jobs, chunksize=max(1, min(16, len(jobs) // (8 * pool._processes)))):
        ot_out[key] = o
        if len(ot_out) % step == 0 or len(ot_out) == len(jobs):
            el = time.time() - t0
            print(f'    {tag} OT {len(ot_out)}/{len(jobs)} pairs, {el / 60:.1f} min '
                  f'(eta {el / len(ot_out) * (len(jobs) - len(ot_out)) / 60:.1f} min)', flush=True)
    ot_max = max(v for o in ot_out.values() for v in o.values())
    offset = max(float(base.max()), ot_max) + 1.0
    mats = {}
    for k in ks:
        for m in m_list:
            d = base.astype(np.float32) + offset
            for i in range(N):
                for j in cand[i][:k]:
                    d[i, j] = ot_out[(min(i, j), max(i, j)) if symmetric else (i, j)][m]
            np.fill_diagonal(d, -1.0 if clean else 0.0)
            mats[(k, m)] = d
    return mats


def run_topk(collect):
    a = G['args']
    names = {(k, m): f'sumpool_top{k}_ot_m{m:g}' for k in a.topk_k for m in a.topk_m}
    with mp.get_context('fork').Pool(a.workers) as ot_pool, mp.get_context('fork').Pool(a.metric_workers) as ev_pool:
        pending = []
        for mode in a.modes:
            M = G['modes'][mode]
            clean_rows = [np.arange(*G['slices'][wp]) for wp in M['valid_wps']]
            level0 = None
            for trial in range(a.trials):
                fmap, sels, fw, excl = trial_selection(mode, trial)
                t0 = time.time()
                for lvl in a.levels:
                    meta = meta_for(mode, trial, lvl)
                    tag, tl = f'[{mode} t{trial + 1} l{lvl}]', time.time()
                    if lvl == 0:
                        if level0 is None:
                            print(f'  {tag} level 0 (clean pages, computed once and reused by every trial)', flush=True)
                            level0 = topk_ot_dists(1 - M['clean'] @ M['clean'].T, clean_rows, clean_rows, True,
                                                   ot_pool, a.topk_k, a.topk_m, clean=False, tag=tag)
                        dists = {}
                        for g in a.galleries:
                            dists[g] = {km: d.copy() for km, d in level0.items()}
                            if g == 'clean':
                                for d in dists[g].values(): np.fill_diagonal(d, -1.0)
                    else:
                        q_rows = [mixed_rows(mode, wp, fmap[wp], sels[wp][lvl], M) for wp in M['valid_wps']]
                        Q = cs.encode([G['feats'][r].sum(0, dtype=np.float32) for r in q_rows])
                        print(f'  {tag} SumPooling descriptors of the contaminated pages done ({time.time() - tl:.0f}s)', flush=True)
                        dists = {}
                        if 'contaminated' in a.galleries:
                            dists['contaminated'] = topk_ot_dists(1 - Q @ Q.T, q_rows, q_rows, True,
                                                                  ot_pool, a.topk_k, a.topk_m, clean=False, tag=tag + '[contaminated]')
                        if 'clean' in a.galleries:
                            b = 1 - Q @ M['clean'].T
                            np.fill_diagonal(b, -1.0)
                            dists['clean'] = topk_ot_dists(b, q_rows, clean_rows, False,
                                                           ot_pool, a.topk_k, a.topk_m, clean=True, tag=tag + '[clean]')
                    print(f'  {tag} level done ({(time.time() - tl) / 60:.1f} min), evaluation queued', flush=True)
                    for g in a.galleries:
                        for km in names:
                            pending.append(ev_pool.apply_async(
                                _eval_task, ((mode, meta, (names[km], mode, g), dists[g][km], fw, excl),)))
                print(f'  topk_ot done {mode} trial {trial + 1} ({time.time() - t0:.0f}s)', flush=True)
                pending = _drain(pending, collect, block=False)
                G['checkpoint']()
        _drain(pending, collect, block=True)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def read_best_sgr(exp_dir):
    p = os.path.join(exp_dir, 'eval', 'baseline_rerank.csv')
    if not os.path.isfile(p):
        return None
    r = next(csv.DictReader(open(p)))
    return int(r['sgr_k']), int(r['sgr_layers']), float(r['sgr_gamma'])


def start_memory_watchdog(limit_frac, interval=2.0):
    """
    Kill this run (parent + every pool / DataLoader worker) before the machine runs out of memory.
    Usage = (MemTotal - MemAvailable) / MemTotal over the WHOLE server, so it also reacts to other users.
    """
    import threading, signal

    def used_frac():
        info = dict(line.split(':')[:2] for line in open('/proc/meminfo'))
        total, avail = int(info['MemTotal'].split()[0]), int(info['MemAvailable'].split()[0])
        return (total - avail) / total

    def watch():
        while True:
            u = used_frac()
            if u >= limit_frac:
                print(f'\n[memory watchdog] server RAM usage {100 * u:.1f}% >= {100 * limit_frac:.0f}% limit '
                      f'-- killing this run and its workers.', flush=True)
                for c in mp.active_children():
                    try: os.kill(c.pid, signal.SIGKILL)
                    except OSError: pass
                os._exit(137)
            time.sleep(interval)

    print(f'[memory watchdog] active: abort at {100 * limit_frac:.0f}% server RAM (now {100 * used_frac():.1f}%)')
    threading.Thread(target=watch, daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(cs.DATASETS), default='cvl')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--methods', nargs='+', choices=['basic', 'sgr_clean', 'trim_eps', 'topk_ot'], required=True)
    parser.add_argument('--modes', choices=['random', 'lines'], nargs='+', default=['random', 'lines'])
    parser.add_argument('--galleries', choices=['contaminated', 'clean'], nargs='+', default=['contaminated', 'clean'])
    parser.add_argument('--levels', default=[0, 10, 20, 30, 40, 50], type=int, nargs='+')
    parser.add_argument('--trials', default=20, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--foreign_map', action='store_true')
    parser.add_argument('--no_exclude_source', dest='exclude_source', action='store_false',
                        help='keep the foreign source page in the own-writer ranking (default: excluded)')
    parser.add_argument('--sgr_k', type=int, default=None, help='default: <exp_dir>/eval/baseline_rerank.csv')
    parser.add_argument('--sgr_layers', type=int, default=None)
    parser.add_argument('--sgr_gamma', type=float, default=None)
    parser.add_argument('--sgr_batch', type=int, default=64, help='sgr_clean: per-query graphs per GPU batch')
    parser.add_argument('--trim_eps', type=float, nargs='+', default=list(config.TRIM_EPS))
    parser.add_argument('--topk_k', type=int, nargs='+', default=[config.TOPK_OT_K],
                        help='one or more k; OT is solved once for max(k) and reused for the smaller ones')
    parser.add_argument('--topk_m', type=float, nargs='+', default=list(config.TOPK_OT_M_LEVELS))
    parser.add_argument('--patch_pca', default=None, help='default: <exp_dir>/pca_patch.pkl (fitted if missing)')
    parser.add_argument('--poolings', nargs='+', choices=list(POOLINGS), default=list(POOLINGS), help='basic: robust poolings to run')
    parser.add_argument('--no_sumpool_ref', dest='sumpool_ref', action='store_false', help='basic: skip the SumPooling reference')
    parser.add_argument('--embed_cache', default=None,
                        help='<path>.npy on disk: read patch features from it (memory-mapped, no model / GPU), or write it with --embed_only')
    parser.add_argument('--embed_only', action='store_true',
                        help='embed straight into --embed_cache and exit, so the GPU is fully released before the CPU part')
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--loader_workers', default=16, type=int)
    parser.add_argument('--workers', default=20, type=int, help='basic: (mode, trial) jobs; topk_ot: OT solver processes')
    parser.add_argument('--metric_workers', default=12, type=int)
    parser.add_argument('--gpuid', default='0', type=str)
    parser.add_argument('--out_dir', default=None, help='default: <exp_dir>/contamination')
    parser.add_argument('--max_writers', default=None, type=int, help='smoke test: only the first N writers')
    parser.add_argument('--summary_only', action='store_true', help='write only summary.csv / foreign_summary.csv')
    parser.add_argument('--keep_metrics', nargs='+', default=None,
                        help='metric columns to keep in the summaries, e.g. map top1 soft-5 (default: all)')
    parser.add_argument('--mem_limit_frac', type=float, default=0.8,
                        help='abort (kill this run and all its workers) when system RAM in use reaches this fraction')
    a, _ = parser.parse_known_args()

    assert config.DATASET == a.dataset, (config.DATASET, a.dataset)
    ds = cs.DATASETS[a.dataset]
    a.patches_dir, a.coords_dir = ds['patches_dir'], ds['coords_dir']
    if a.seed is None:
        a.seed = config.DATASET_CONFIGS[a.dataset]['seed']
    use_sgr = bool({'basic', 'sgr_clean'} & set(a.methods))
    if use_sgr:
        best = read_best_sgr(a.exp_dir)
        a.sgr_k = a.sgr_k if a.sgr_k is not None else best[0]
        a.sgr_layers = a.sgr_layers if a.sgr_layers is not None else best[1]
        a.sgr_gamma = a.sgr_gamma if a.sgr_gamma is not None else best[2]
        G['sgr_name'] = f'sumpool_sgr_k{a.sgr_k}_l{a.sgr_layers}_g{a.sgr_gamma:g}'
    config.PCT_LEVELS = a.levels; config.BIN_CENTERS = a.levels; config.PATCHES_DIR = a.coords_dir
    out_base = a.out_dir or os.path.join(a.exp_dir, 'contamination')
    print(f'dataset={a.dataset} methods={a.methods} modes={a.modes} galleries={a.galleries} levels={a.levels} '
          f'trials={a.trials} seed={a.seed} foreign_map={a.foreign_map}\n'
          + (f'SGR k={a.sgr_k} layers={a.sgr_layers} gamma={a.sgr_gamma}\n' if use_sgr else '')
          + (f'trim eps={a.trim_eps}\n' if 'trim_eps' in a.methods else '')
          + (f'topk k={a.topk_k} m={a.topk_m}\n' if 'topk_ot' in a.methods else '') + f'out: {out_base}')

    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # `kill -USR1 <pid>` dumps that process's Python stack into the log
    G['out_base'] = out_base
    start_memory_watchdog(a.mem_limit_frac)
    exp_cfg = load_yaml(os.path.join(a.exp_dir, 'config.yaml'))
    seed_everything(a.seed)
    assert not a.embed_only or a.embed_cache, '--embed_only needs --embed_cache'
    from_cache = bool(a.embed_cache) and not a.embed_only
    # trim_eps / sgr_clean need the GPU; topk_ot is CPU-only once the patch PCA exists
    assert not (from_cache and {'trim_eps', 'sgr_clean'} & set(a.methods)), '--embed_cache: basic / topk_ot only'
    assert not (from_cache and 'topk_ot' in a.methods
                and not os.path.isfile(a.patch_pca or os.path.join(a.exp_dir, 'pca_patch.pkl'))), \
        '--embed_cache with topk_ot needs an existing patch PCA (fitting it needs the model)'
    model = None
    if not from_cache:
        GPU.set(a.gpuid, 400)
        backbone = getattr(resnets, exp_cfg['model']['name'])()
        model = Model(backbone, dim=64, num_clusters=exp_cfg['model']['num_clusters'],
                      random=exp_cfg['model'].get('encoding', None) == 'netrvlad')
        model.load_state_dict(torch.load(os.path.join(a.exp_dir, 'model.pt'), map_location='cpu')['model_state_dict'])
        model = model.cuda().eval()
    else:
        print(f'Reading patch features from {a.embed_cache} (memory-mapped): no model, no GPU in this process', flush=True)

    G.update(args=a, poolings=a.poolings)
    pca_patch = None
    if {'trim_eps', 'topk_ot'} & set(a.methods):
        a.patch_pca = a.patch_pca or os.path.join(a.exp_dir, 'pca_patch.pkl')
        if os.path.isfile(a.patch_pca):
            pca_patch = pickle.load(open(a.patch_pca, 'rb'))
            print(f'Loaded patch PCA {a.patch_pca}')
        else:
            pca_patch = fit_patch_pca(model, a.patch_pca, seed=a.seed)

    keep = None
    if a.max_writers:
        ws = sorted({int(ds['page_re'].fullmatch(d).group(1)) for d in os.listdir(a.patches_dir)
                     if ds['page_re'].fullmatch(d)})[:a.max_writers]
        keep = lambda wp, ws=set(ws): wp[0] in ws
    if from_cache:
        feats = np.load(a.embed_cache, mmap_mode='r')   # page cache, reclaimable: does not count against the RAM watchdogs
        slices = pickle.load(open(a.embed_cache + '.slices.pkl', 'rb'))
        print(f'Patch features {feats.shape} ({feats.nbytes / 1e9:.1f} GB on disk)', flush=True)
    else:
        feats, slices = cs.embed_pages(model, a.patches_dir, ds['page_re'], a.batch_size, a.loader_workers, keep=keep,
                                       out_path=a.embed_cache if a.embed_only else None)
        if a.embed_only:
            pickle.dump(slices, open(a.embed_cache + '.slices.pkl', 'wb'))
            print(f'Saved patch features -> {a.embed_cache}; exiting so the GPU is released', flush=True)
            return
        del model; torch.cuda.empty_cache()
    G.update(feats=feats, slices=slices, pca=pickle.load(open(os.path.join(a.exp_dir, 'pca.pkl'), 'rb')),
             pooling=SumPooling('l2', pn_alpha=0.4), npatch={wp: e - s for wp, (s, e) in slices.items()})
    G['own_sums'] = {wp: feats[s:e].sum(0, dtype=np.float32) for wp, (s, e) in slices.items()}
    G['modes'] = {m: cs.prepare_mode(m, list(slices)) for m in a.modes}
    if pca_patch is not None:
        t0 = time.time()
        G['pbank'] = build_patch_bank(pca_patch)
        print(f'patch bank (128-d PCA of all patches) built ({(time.time() - t0) / 60:.1f} min)', flush=True)

    # ── result collection + writing ───────────────────────────────────────────
    collected = defaultdict(lambda: ([], []))
    params = {}

    def collect(method, mode, g, meta, own, fgn):
        collected[(method, mode, g)][0].append({**meta, **own})
        if fgn is not None:
            collected[(method, mode, g)][1].append({**meta, **fgn})

    def write_all():
        for (method, mode, g), (rows, rows_f) in collected.items():
            if not rows:
                continue
            out_dir = os.path.join(out_base, method, mode + ('_clean_gallery' if g == 'clean' else ''))
            os.makedirs(out_dir, exist_ok=True)
            extra = {'n_pages': len(G['modes'][mode]['valid_wps']), 'patches_dir': a.patches_dir,
                     'gallery': g, 'method': method, 'source_page_excluded': a.exclude_source}
            lv = sorted({r['level'] for r in rows}, key=a.levels.index)
            kw = dict(keep_metrics=a.keep_metrics, write_trials=not a.summary_only)
            cs.save_results(rows, lv, out_dir, '', f'{method} / {mode} / {g}: original writer labels', extra, **kw)
            if rows_f:
                cs.save_results(rows_f, lv, out_dir, 'foreign_', f'{method} / {mode} / {g}: foreign writer labels', extra, **kw)

    G['checkpoint'] = write_all

    if 'basic' in a.methods:
        run_basic(collect)
        write_all()
    if 'sgr_clean' in a.methods:
        run_sgr_clean(collect)
        write_all()
    if 'trim_eps' in a.methods:
        run_trim(collect, G['pbank'])
        write_all()
    if 'topk_ot' in a.methods:
        run_topk(collect)
        write_all()
    print('\nDone.')


if __name__ == '__main__':
    main()
