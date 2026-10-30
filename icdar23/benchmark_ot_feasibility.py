"""
benchmark_ot_feasibility.py
───────────────────────────
Loads a real sample of pages straight from config.H5_PATH / config.INDEX_PATH
(no contamination mixing, no Pool, no selection.py) — just the same
_read_norm + _prep_patch_repr steps workers.py applies at 0% contamination,
which is identical to the unmodified page regardless of MODE. Enough to get
a realistic per-pair OT solve time before wiring anything into contam_main.py.

Run from the SAME directory as config.py / workers.py / trimming.py /
ot_distance.py (your repo dir, NOT the experiment_dir — config.py's paths
are absolute, so your CWD doesn't need to be the experiment_dir at all):

    python benchmark_ot_feasibility.py
    python benchmark_ot_feasibility.py --dataset cvl --mode random   # same flags config.py accepts
    python benchmark_ot_feasibility.py --n-sample 25 --n-jobs 32
"""

import argparse
import pickle
import random
import time

import h5py
import numpy as np

import config   # reads --dataset/--mode itself via parse_known_args, safe to import first
from ot_distance import partial_ot_distance_matrices


# ── extra CLI on top of what config.py already consumes ───────────────────────
_p = argparse.ArgumentParser(add_help=True)
_p.add_argument('--n-sample', type=int, default=25, help='pages to sample for the timing run')
_p.add_argument('--n-jobs',   type=int, default=None, help='override config.N_WORKERS for this run')
_p.add_argument('--m-list',   type=float, nargs='+', default=[0.5, 0.7, 0.9],
                help='partial-OT mass fractions to benchmark (use config.OT_M_LEVELS if you added it)')
_p.add_argument('--mem-limit-gb', type=float, default=None,
                help='total RAM budget across all workers (safety net, see ot_distance.py)')
_p.add_argument('--gpuid', type=str, default=None,
                help='CUDA_VISIBLE_DEVICES for this process (inert for the exact solver itself)')
_args, _ = _p.parse_known_args()

N_SAMPLE = _args.n_sample
N_JOBS   = _args.n_jobs or config.N_WORKERS
M_LIST   = getattr(config, 'OT_M_LEVELS', _args.m_list)
MEM_LIMIT_GB = _args.mem_limit_gb
GPUID    = _args.gpuid


def _read_norm(h5, key):
    arr   = h5[key][:].astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms < 1e-8, 1.0, norms)


def _prep_patch_repr(mixed, pca_patch):
    reduced = pca_patch.transform(mixed)
    norms   = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.where(norms < 1e-8, 1.0, norms)
    return reduced.astype(np.float32)


if __name__ == '__main__':
    print(f'Dataset={config.DATASET}  Mode={config.MODE}')
    print(f'H5_PATH        = {config.H5_PATH}')
    print(f'INDEX_PATH     = {config.INDEX_PATH}')
    print(f'PCA_PATCH_PATH = {config.PCA_PATCH_PATH}')

    with open(config.INDEX_PATH, 'rb') as f:
        index = pickle.load(f)
    pca_patch = pickle.load(open(config.PCA_PATCH_PATH, 'rb'))

    n_total_pages = len(index)
    print(f'\nTotal pages in index: {n_total_pages}')

    random.seed(config.SEED)
    sample_entries = random.sample(index, min(N_SAMPLE, n_total_pages))

    print(f'Reading {len(sample_entries)} real pages from {config.H5_PATH} ...')
    patch_arrays = []
    with h5py.File(config.H5_PATH, 'r') as h5:
        for e in sample_entries:
            own = _read_norm(h5, e['hdf5_key'])
            patch_arrays.append(_prep_patch_repr(own, pca_patch))

    sizes = [a.shape[0] for a in patch_arrays]
    print(f'Sample patch counts: min={min(sizes)} median={int(np.median(sizes))} '
          f'max={max(sizes)} mean={np.mean(sizes):.0f}')
    print(f'm_list = {M_LIST}   n_jobs = {N_JOBS}   '
          f'mem_limit_gb = {MEM_LIMIT_GB}   gpuid = {GPUID}')

    t0 = time.perf_counter()
    _ = partial_ot_distance_matrices(patch_arrays, M_LIST, n_jobs=N_JOBS,
                                      mem_limit_gb=MEM_LIMIT_GB, gpuid=GPUID)
    t1 = time.perf_counter()

    n_sample = len(patch_arrays)
    n_pairs_sample = n_sample * (n_sample - 1) // 2
    per_pair_all_m = (t1 - t0) / n_pairs_sample
    print(f'\n{n_pairs_sample} pairs (all {len(M_LIST)} m-values) solved in '
          f'{t1 - t0:.2f}s  ->  {per_pair_all_m * 1000:.1f} ms/pair (all m combined)')

    # ── extrapolate using the REAL page count from the index ─────────────────
    n_pairs_real = n_total_pages * (n_total_pages - 1) // 2
    total_cpu_seconds = n_pairs_real * per_pair_all_m * config.N_TRIALS * len(config.PCT_LEVELS)

    print(f'\nUsing real N={n_total_pages} pages, N_TRIALS={config.N_TRIALS}, '
          f'len(PCT_LEVELS)={len(config.PCT_LEVELS)}, N_WORKERS={config.N_WORKERS}:')
    print(f'  Estimated total CPU-seconds : {total_cpu_seconds:,.0f}s')
    print(f'  Estimated wall-clock        : '
          f'{total_cpu_seconds / config.N_WORKERS / 3600:.1f} hours (ideal, no overhead)')
    print('\nCaveat: real patch counts vary page-to-page far more than this small '
          'sample shows, and solve time scales ~cubically with patch count — a '
          'handful of unusually large pages will dominate wall-clock more than this '
          'mean-based estimate suggests. Treat this as a lower bound, and check the '
          'max() in the sample stats above, not just the mean.')
