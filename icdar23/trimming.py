"""
trimming.py
───────────
GPU-batched trimmed (one-sided partial Chamfer) distance for page-level
writer retrieval, as sketched in the "Trimming" slide:

    For each query patch q_i, find its best match in candidate page X:
        d_i = min_j || q_i - x_j ||
    Sort the d_i ascending, keep only the best (1 - eps) fraction, average:
        d(Q, X) = mean( d_(1), ..., d_(floor((1-eps) n)) )

Patches are assumed already L2-normalised (see workers._read_norm) and
PCA-reduced (see workers._prep_patch_repr / config.PCA_PATCH_PATH). Because
they are unit-norm, Euclidean distance is a monotonic function of cosine
similarity:

    ||a - b||^2 = 2 - 2 cos(a, b)      (for ||a|| = ||b|| = 1)

so nearest-neighbour search is done via a single matmul (cosine similarity)
rather than an explicit pairwise-Euclidean computation.

Directionality
──────────────
d(Q, X) != d(X, Q) in general — this is an intentional, cheaper choice
(computing one direction per pair rather than both and averaging). The
downstream retrieval code (retrieval.py) only ever reads distances[i, :]
for a fixed query row i, so an asymmetric matrix requires no changes there.
Self-distance dist[i, i] falls out naturally as 0 (a page's patch set
matches itself perfectly), matching the existing pooled-descriptor baseline.

Chunking
────────
The candidate axis is processed in chunks (`j_chunk` pages at a time) because
the intermediate similarity tensor has shape (n_i, j_chunk * max_n) — with
max_n set by the single largest page in the set, an unchunked pass can demand
tens of GB in one allocation. If j_chunk is left as None, it is auto-sized
from currently-free GPU memory (see _auto_j_chunk) so this doesn't need to be
hand-tuned per dataset; pass an explicit int to override.

Public API
──────────
trimmed_distance_matrices(patch_arrays, eps_list, device='cuda', j_chunk=None)
    Returns { eps: (N, N) np.float32 } for every eps in eps_list, sharing
    the expensive sort across eps values (only the trim fraction differs).
"""

import numpy as np
import torch


_CHUNK_CACHE = {}


def _auto_j_chunk(N, max_n, device, mem_fraction=0.25, min_chunk=1, fallback=50):
    """
    Pick a candidate-page chunk size so the transient (n_i, j_chunk*max_n)
    similarity tensor fits comfortably in currently-free GPU memory, using
    the worst case n_i = max_n. mem_fraction caps the fraction of *free*
    memory (not total) this single buffer is allowed to use, leaving room
    for the persistent padded tensor, best_dist, sorting, and anything else
    sharing the GPU. Cached per (N, max_n, device) so repeated calls across
    trials/levels with the same page-set shape don't re-query/re-print.
    """
    key = (N, max_n, device)
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]

    if device == 'cpu' or not torch.cuda.is_available():
        chunk = max(min_chunk, min(N, fallback))
    else:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            budget = free_bytes * mem_fraction
            bytes_per_cand_page = max_n * max_n * 4   # worst-case n_i=max_n, float32
            chunk = max(min_chunk, min(N, int(budget // max(1, bytes_per_cand_page))))
        except Exception:
            chunk = max(min_chunk, min(N, fallback))

    print(f'[trimming] auto j_chunk={chunk} (N={N}, max_n={max_n}, device={device})')
    _CHUNK_CACHE[key] = chunk
    return chunk


def _to_padded(patch_arrays, device):
    """
    patch_arrays : list of (n_i, D) np.float32 arrays, L2-normalised, same
                   order as page_writers / valid_wps.

    Returns padded (N, max_n, D), mask (N, max_n) bool, lens (list[int]).
    Pages with zero patches (shouldn't normally happen post-MIN_LINES/
    patch filtering) get an all-pad row; their output row is left as 0.
    """
    N = len(patch_arrays)
    lens = [int(p.shape[0]) for p in patch_arrays]
    max_n = max(1, max(lens))
    D = next(p.shape[1] for p in patch_arrays if p.shape[0] > 0)

    padded = torch.zeros(N, max_n, D, dtype=torch.float32, device=device)
    mask = torch.zeros(N, max_n, dtype=torch.bool, device=device)
    for i, p in enumerate(patch_arrays):
        n = lens[i]
        if n == 0:
            continue
        padded[i, :n] = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))
        mask[i, :n] = True
    return padded, mask, lens


def trimmed_distance_matrices(patch_arrays, eps_list, device='cuda', j_chunk=None):
    """
    patch_arrays : list[(n_i, D) np.float32], L2-normalised + PCA-reduced.
    eps_list     : list[float], e.g. [0.10, 0.25, 0.50].
    device       : 'cuda' or 'cpu'; falls back to 'cpu' if CUDA unavailable.
    j_chunk      : candidate pages per inner batch. None = all candidates
                   at once (fastest, most memory). Lower this if you hit
                   GPU OOM on large test sets (e.g. ICDAR).

    Returns { eps: (N, N) np.float32 dist matrix } for every eps.
    dist[i, j] = trimmed distance FROM query page i TO candidate page j.
    """
    device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    padded, mask, lens = _to_padded(patch_arrays, device)
    N, max_n, D = padded.shape

    flat = padded.reshape(N * max_n, D)        # (N*max_n, D)
    flat_mask = mask.reshape(N * max_n)         # (N*max_n,)

    out = {eps: np.zeros((N, N), dtype=np.float32) for eps in eps_list}
    if j_chunk is None:
        j_chunk = _auto_j_chunk(N, max_n, device)

    with torch.no_grad():
        for i in range(N):
            n_i = lens[i]
            if n_i == 0:
                continue
            q = padded[i, :n_i]                                   # (n_i, D)

            best_dist = torch.empty(n_i, N, device=device)
            for j0 in range(0, N, j_chunk):
                j1 = min(j0 + j_chunk, N)
                cand = flat[j0 * max_n: j1 * max_n]                # ((j1-j0)*max_n, D)
                cand_mask = flat_mask[j0 * max_n: j1 * max_n]

                sim = q @ cand.T                                   # (n_i, (j1-j0)*max_n)
                sim.masked_fill_(~cand_mask.unsqueeze(0), -1.0)    # in-place: no second full-size copy
                sim = sim.view(n_i, j1 - j0, max_n)
                sim_best, _ = sim.max(dim=2)                       # (n_i, j1-j0)
                best_dist[:, j0:j1] = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_best, min=0.0))
                del sim, sim_best, cand, cand_mask

            sorted_dist, _ = torch.sort(best_dist, dim=0)          # ascending over query patches
            for eps in eps_list:
                k = max(1, int(np.floor((1.0 - eps) * n_i)))
                out[eps][i, :] = sorted_dist[:k, :].mean(dim=0).detach().cpu().numpy()
            del best_dist, sorted_dist

    return out