"""
ot_distance.py
──────────────
Page-level partial optimal-transport distance for writer retrieval, using
POT's pre-implemented exact solver (ot.partial.partial_wasserstein) — no
reimplementation of the OT math, per the slide's own link to
https://pythonot.github.io/.

Why this looks different from trimming.py
──────────────────────────────────────────
trimming.py's min+trim reduces to one matmul + max → batches trivially on
GPU across all candidate pages at once. Partial OT's exact solver is a
network-simplex linear program per (query page, candidate page) pair, with
no GPU-native implementation (confirmed: POT's exact EMD/partial-EMD
backend is CPU-only, GPU wrapper just avoids host<->device copies, the
solve itself still runs on CPU). So parallelism here comes from
multiprocessing across page pairs, not GPU batching — reusing the same
Pool pattern as contam_main.py's trial loop.

Cost-matrix normalisation caveat
────────────────────────────────
The transport *plan* is solved on a normalised cost matrix (M / M.max())
for solver conditioning, matching the slide's snippet — but the reported
distance is computed against the ORIGINAL, unnormalised M. Scoring with
the normalised M would rescale each pair's distance by that pair's own
M.max(), which differs per candidate and would distort ranking.

Symmetry
────────
Unlike trimming's deliberately asymmetric one-sided distance, partial OT
cost is symmetric under swapping Q/X (same bipartite matching problem
either way) — so only the upper triangle is solved and mirrored, roughly
halving the number of LP solves versus a naive full N×N pass.

Zero-patch pages
─────────────────
A page with zero patches gets row/col = +inf (never a spurious top match),
except the diagonal, which is left at 0 (a page trivially "matches itself").

Fork-inherited patch arrays (avoids re-pickling per pair)
──────────────────────────────────────────────────────────
Earlier version passed Qi/Qj through the job tuple, meaning every one of
the ~N²/2 jobs re-pickled and piped two (n_patches, D) float arrays across
the multiprocessing IPC boundary — even though the Pool uses `fork`, which
already gives every worker the whole `patch_arrays` list for free via
copy-on-write. Fixed the same way workers.py does it: set a module global
BEFORE creating the Pool, so forked children inherit it directly; job
tuples now carry only (i, j, ...) — small ints, no arrays.

Public API
──────────
partial_ot_distance_matrices(patch_arrays, m_list, n_jobs=None, cost_metric='sqeuclidean')
    Returns { m: (N, N) np.float32 } for every m in m_list.
"""

import os
import resource
from itertools import combinations
from multiprocessing import get_context

import numpy as np
import ot


_PATCH_ARRAYS = None   # set by partial_ot_distance_matrices before Pool(); fork-inherited


def _worker_init(mem_limit_bytes):
    """
    Pool initializer: caps THIS worker's own address space. A per-process
    safety net, not a real constraint — measured peak per pair is well
    under 1GB even at ~2000 patches/page (see ot_distance's own benchmarks),
    so this should never actually bind. Splitting a total budget across
    n_jobs workers, rather than applying it once in the parent, so a
    runaway single pair fails that one job instead of taking down the
    whole shared machine.
    """
    if mem_limit_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))


_DEFAULT_RETRY_SCHEDULE = (
    dict(nb_dummies=1,  numItermax=100000),    # POT's own defaults
    dict(nb_dummies=1,  numItermax=1000000),   # same problem size, 10x iteration budget —
                                                # this is what the "numItermax reached" warning
                                                # actually asks for; try this BEFORE growing the LP
    dict(nb_dummies=10, numItermax=1000000),   # only grow the problem as a last resort
)


def _solve_pair(args):
    i, j, m_list, cost_metric, retry_schedule = args
    Qi = _PATCH_ARRAYS[i]
    Qj = _PATCH_ARRAYS[j]

    M = ot.dist(Qi, Qj, metric=cost_metric)          # (nq, nx), original scale
    scale = float(M.max())
    Mn = M / scale if scale > 0 else M               # normalised, for the solver only

    a = np.full(Qi.shape[0], 1.0 / Qi.shape[0])
    b = np.full(Qj.shape[0], 1.0 / Qj.shape[0])

    out = {}
    for m in m_list:
        gamma, last_err = None, None
        for cfg in retry_schedule:
            try:
                gamma = ot.partial.partial_wasserstein(a, b, Mn, m=m, **cfg)
                break
            except ValueError as e:
                # NOTE: partial_wasserstein raises this exact text
                # ("...try to increase the number of dummy points") for ANY
                # underlying solver warning, including "numItermax reached" —
                # the message doesn't tell you which. Escalating numItermax
                # BEFORE nb_dummies (see schedule above) targets the more
                # likely root cause without growing the LP unnecessarily.
                last_err = e
                continue
        if gamma is None:
            raise RuntimeError(
                f'partial_wasserstein failed for page pair (i={i}, j={j}), m={m} '
                f'even after retry schedule {retry_schedule}. Last error: {last_err}'
            )
        out[m] = float(np.sum(gamma * M))            # score with the UNNORMALISED M
    return i, j, out


def partial_ot_distance_matrices_topk(patch_arrays, base_dist, k, m_list, n_jobs=None,
                                       cost_metric='sqeuclidean', mem_limit_gb=None, gpuid=None,
                                       retry_schedule=_DEFAULT_RETRY_SCHEDULE):
    """
    Two-stage version of partial_ot_distance_matrices: instead of solving the
    exact partial-OT LP for every page pair, use an already-computed cheap
    distance matrix (base_dist, e.g. SumPooling cosine distance) to pick, for
    every query page, its top-k nearest candidates, and only solve OT for
    those (query, candidate) pairs. Turns the O(N^2) LP-solve cost into
    roughly O(k*N).

    base_dist : (N, N) np.ndarray, e.g. the SumPooling distance matrix for
                the same level/trial. Used both to pick candidates and, for
                every page pair NOT selected as a candidate, as the returned
                distance (i.e. this "reranks" only the top-k per query and
                leaves everything else at its stage-1 SumPooling rank).
    k         : candidates per query page.
    Other args: see partial_ot_distance_matrices.

    Returns { m: (N, N) np.float32 } where, for each query row i, the k
    candidate columns are ranked by exact partial-OT distance and guaranteed
    to sort ahead of every non-candidate column (which keeps its base_dist
    value, shifted by a constant offset so it can never outrank a candidate).
    """
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    global _PATCH_ARRAYS
    _PATCH_ARRAYS = patch_arrays

    N = len(patch_arrays)
    valid = [i for i, p in enumerate(patch_arrays) if p.shape[0] > 0]
    empty = [i for i, p in enumerate(patch_arrays) if p.shape[0] == 0]
    valid_set = set(valid)
    empty_mask = np.zeros(N, dtype=bool)
    empty_mask[empty] = True

    # top-k candidates per query, from the cheap base distance
    topk = {}
    for i in valid:
        row = base_dist[i].copy()
        row[i] = np.inf
        row[empty_mask] = np.inf
        order = np.argsort(row)
        topk[i] = [j for j in order[:k] if j in valid_set]

    directed = {(i, j) for i in valid for j in topk[i]}
    unordered_pairs = {(min(i, j), max(i, j)) for i, j in directed}
    jobs = [(i, j, m_list, cost_metric, retry_schedule) for i, j in unordered_pairs]

    ot_out = {}   # (i, j) with i < j -> {m: dist}
    if jobs:
        ctx = get_context('fork')
        n_workers = n_jobs or os.cpu_count() or 8
        chunksize = max(1, len(jobs) // (4 * n_workers))
        per_worker_bytes = int(mem_limit_gb * 1e9 / n_workers) if mem_limit_gb else None

        with ctx.Pool(processes=n_jobs, initializer=_worker_init,
                       initargs=(per_worker_bytes,)) as pool:
            for i, j, out in pool.imap_unordered(_solve_pair, jobs, chunksize=chunksize):
                ot_out[(i, j)] = out

    base_finite = base_dist[np.isfinite(base_dist)]
    base_max = float(base_finite.max()) if base_finite.size else 0.0
    ot_max = max((v for out in ot_out.values() for v in out.values()), default=0.0)
    offset = max(base_max, ot_max) + 1.0   # guarantees candidates always outrank non-candidates

    mats = {m: base_dist.astype(np.float32) + offset for m in m_list}
    for m in m_list:
        np.fill_diagonal(mats[m], 0.0)

    for i, j in directed:
        out = ot_out.get((min(i, j), max(i, j)))
        if out is None:
            continue
        for m in m_list:
            mats[m][i, j] = out[m]

    for i in empty:
        for m in m_list:
            mats[m][i, :] = np.inf
            mats[m][:, i] = np.inf
            mats[m][i, i] = 0.0

    _PATCH_ARRAYS = None
    return mats


def partial_ot_distance_matrices(patch_arrays, m_list, n_jobs=None, cost_metric='sqeuclidean',
                                  mem_limit_gb=None, gpuid=None, retry_schedule=_DEFAULT_RETRY_SCHEDULE):
    """
    patch_arrays : list[(n_i, D) np.float32], same PCA-reduced, unit-norm
                   local descriptors trimming.py consumes.
    m_list       : list[float], fraction of total mass to transport, e.g.
                   [0.5, 0.7, 0.9] (POT's `m` kwarg — same role as TRIM_EPS,
                   but as "mass kept" rather than "fraction discarded":
                   m=0.8 ~ eps=0.2 in spirit, not numerically identical).
    n_jobs       : worker processes; None = os.cpu_count() default (Pool's own default).
    cost_metric  : passed straight to ot.dist ('sqeuclidean', 'euclidean', 'cosine', ...).
    mem_limit_gb : float or None. Total RAM budget across ALL workers combined;
                   split evenly and applied as a hard per-worker RLIMIT_AS.
                   Safety net only — see _worker_init docstring. NOTE: this
                   caps virtual address space, which gets inflated by an
                   already-initialised CUDA context in the parent process
                   (e.g. from a prior trimming.py call on the same level) —
                   set to None if every worker fails identically at startup.
    gpuid        : str/int or None. Sets CUDA_VISIBLE_DEVICES for this process.
                   NOTE: ot.partial.partial_wasserstein (the exact solver used
                   here) has no GPU implementation — this has zero effect on
                   speed/memory for this function. Only useful to keep this
                   process off a GPU shared with other jobs on the same node.
    retry_schedule : tuple of dict, each a kwargs override for
                   ot.partial.partial_wasserstein (nb_dummies / numItermax).
                   partial_wasserstein raises the SAME generic ValueError
                   text for any underlying solver warning, so this escalates
                   numItermax first (targets "numItermax reached before
                   optimality" directly) and only grows nb_dummies (which
                   enlarges the LP, needing MORE iterations, not fewer) as a
                   last resort. Each pair tries these in order until one
                   solves without error; only escalates on failure, so the
                   common case pays no extra cost.

    Returns { m: (N, N) np.float32 dist matrix }, symmetric, diag = 0
    except zero-patch pages (row/col = inf, diag = 0).
    """
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    global _PATCH_ARRAYS
    _PATCH_ARRAYS = patch_arrays   # set BEFORE Pool() so fork children inherit it, no pickling

    N = len(patch_arrays)
    mats = {m: np.zeros((N, N), dtype=np.float32) for m in m_list}

    valid = [i for i, p in enumerate(patch_arrays) if p.shape[0] > 0]
    empty = [i for i, p in enumerate(patch_arrays) if p.shape[0] == 0]

    jobs = [(i, j, m_list, cost_metric, retry_schedule) for i, j in combinations(valid, 2)]

    if jobs:
        ctx = get_context('fork')
        n_workers = n_jobs or os.cpu_count() or 8
        chunksize = max(1, len(jobs) // (4 * n_workers))
        per_worker_bytes = int(mem_limit_gb * 1e9 / n_workers) if mem_limit_gb else None

        with ctx.Pool(processes=n_jobs, initializer=_worker_init,
                       initargs=(per_worker_bytes,)) as pool:
            for i, j, out in pool.imap_unordered(_solve_pair, jobs, chunksize=chunksize):
                for m, d in out.items():
                    mats[m][i, j] = d
                    mats[m][j, i] = d

    for i in empty:
        for m in m_list:
            mats[m][i, :] = np.inf
            mats[m][:, i] = np.inf
            mats[m][i, i] = 0.0

    _PATCH_ARRAYS = None   # release reference
    return mats
