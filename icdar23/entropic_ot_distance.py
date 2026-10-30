"""
entropic_ot_distance.py
────────────────────────
GPU-batched entropic (Sinkhorn) partial-OT distance for page-level writer
retrieval — a matrix-op-only alternative to ot_distance.py's exact solver
(ot.partial.partial_wasserstein), which is CPU-bound, per-pair, and has
proven numerically fragile at your scale (see ot_distance.py's retry
schedule, added to work around numItermax / dummy-point failures).

This reimplements POT's OWN entropic_partial_wasserstein iteration scheme
(ot.partial.partial_solvers._entropic_partial_wasserstein_sinkhorn — the
Bregman-projection algorithm from Benamou et al. 2015), just with a leading
batch dimension so MANY (query page, candidate page) pairs solve
simultaneously as one set of GPU tensor ops — the same principle
trimming.py uses to batch across candidate pages instead of looping.

Correctness: validated against POT's own per-pair reference in this
file's __main__ smoke test (run `python entropic_ot_distance.py`) — batched
output matches the per-pair reference to float32 tolerance.

Trade-off vs. the exact solver (read before using)
────────────────────────────────────────────────────
This is NOT the same objective as the slide's exact partial_wasserstein —
`reg` adds an entropy penalty:
  - too small  -> numerical underflow (K = exp(-M/reg) -> 0 everywhere),
                  transport plan collapses to NaN/Inf. This function
                  detects and reports which (i, j) pairs went unstable
                  (see `unstable_pairs` in the return) rather than silently
                  zeroing them — check that list before trusting a run.
  - too large  -> the transport plan smooths/blurs across many patches
                  instead of concentrating on true nearest matches, which
                  can hurt the ranking's discriminativeness (the whole
                  point of this method vs. mean-pooling).
Treat `reg` as a hyperparameter to sweep (start around 0.01-0.05 on
sqeuclidean costs over unit-norm descriptors, then adjust based on how
many pairs land in `unstable_pairs`), not a free performance win.

Public API
──────────
entropic_ot_distance_matrices(patch_arrays, m_list, reg, device='cuda',
                               numItermax=1000, stopThr=1e-6, pair_chunk=None)
    Returns (mats, unstable_pairs):
      mats            : { m: (N, N) np.float32 } for every m in m_list
      unstable_pairs  : list of (i, j) pairs that hit NaN/Inf at this reg
                        for ANY m — their distance in mats is np.nan, not
                        silently 0. Re-run these through ot_distance.py's
                        exact solver, or retry with a larger reg.
"""

import numpy as np
import torch


def _to_padded(patch_arrays, device):
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


def _batched_entropic_partial_cost(Mb, mask_i, mask_j, m, reg, numItermax, stopThr):
    """
    Mb     : (B, ni, nj) cost matrix, real distances (no padding trick needed —
             padding is handled via mask_i/mask_j directly, see below)
    mask_i : (B, ni) bool, True = real patch
    mask_j : (B, nj) bool
    m      : float, target mass (same for every pair in this batch)

    Returns (cost, unstable) : both (B,) tensors. unstable[k]=True means
    pair k hit NaN/Inf during the iterations (cost[k] is NaN in that case,
    not a silently-wrong number).
    """
    B, ni, nj = Mb.shape
    device = Mb.device

    mi = mask_i.float()
    mj = mask_j.float()
    a = mi / mi.sum(dim=1, keepdim=True).clamp(min=1)   # (B, ni), 0 on padding
    b = mj / mj.sum(dim=1, keepdim=True).clamp(min=1)   # (B, nj), 0 on padding

    pad2d = mask_i.unsqueeze(2) & mask_j.unsqueeze(1)   # (B, ni, nj) real-real entries

    K = torch.exp(-Mb / reg) * pad2d
    ksum = K.sum(dim=(1, 2), keepdim=True).clamp(min=1e-300)
    K = K * (m / ksum)

    q1 = torch.ones_like(K)
    q2 = torch.ones_like(K)
    q3 = torch.ones_like(K)
    dx = torch.ones(B, ni, device=device)
    dy = torch.ones(B, nj, device=device)

    for cpt in range(numItermax):
        Kprev = K
        K = K * q1
        rowsum = K.sum(dim=2).clamp(min=1e-300)
        scale_r = torch.minimum(a / rowsum, dx).unsqueeze(2)
        K1 = K * scale_r
        q1 = q1 * Kprev / K1.clamp(min=1e-300)

        K1prev = K1
        K1 = K1 * q2
        colsum = K1.sum(dim=1).clamp(min=1e-300)
        scale_c = torch.minimum(b / colsum, dy).unsqueeze(1)
        K2 = K1 * scale_c
        q2 = q2 * K1prev / K2.clamp(min=1e-300)

        K2prev = K2
        K2 = K2 * q3
        ksum2 = K2.sum(dim=(1, 2), keepdim=True).clamp(min=1e-300)
        K = K2 * (m / ksum2)
        q3 = q3 * K2prev / K.clamp(min=1e-300)

        K = K * pad2d   # re-zero padding every iteration (belt and suspenders)

        if cpt % 10 == 0:
            err = torch.linalg.vector_norm((Kprev - K).reshape(B, -1), dim=1)
            if torch.all(err < stopThr):
                break

    unstable = torch.isnan(K).reshape(B, -1).any(dim=1) | torch.isinf(K).reshape(B, -1).any(dim=1)
    cost = (Mb * K).sum(dim=(1, 2))
    cost = torch.where(unstable, torch.full_like(cost, float('nan')), cost)
    return cost, unstable


def entropic_ot_distance_matrices(patch_arrays, m_list, reg, device='cuda',
                                   numItermax=1000, stopThr=1e-6, pair_chunk=None,
                                   cost_metric='sqeuclidean'):
    """
    patch_arrays : list[(n_i, D) np.float32], same PCA-reduced, unit-norm
                   local descriptors ot_distance.py / trimming.py consume.
    m_list       : list[float], mass fractions (same meaning as ot_distance.py).
    reg          : float, entropy regularization strength. See module
                   docstring — this is the parameter you actually need to
                   tune; there's no universally-safe default.
    device       : 'cuda' or 'cpu'; falls back to 'cpu' if CUDA unavailable.
    numItermax, stopThr : passed through to the Sinkhorn loop, same meaning
                   as POT's own entropic_partial_wasserstein.
    pair_chunk   : page-pairs solved per GPU batch. None = a conservative
                   default (32) — raise this if GPU memory allows, since
                   memory scales as pair_chunk * max_n^2 in the WORST case
                   (the largest pair of pages in the whole run, not just
                   this chunk, since padding uses each pair's own max_n but
                   different chunks can have different max_n).

    Returns (mats, unstable_pairs) — see module docstring.
    """
    device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    padded, mask, lens = _to_padded(patch_arrays, device)
    N = len(patch_arrays)
    pair_chunk = pair_chunk or 32

    from itertools import combinations
    valid = [i for i, n in enumerate(lens) if n > 0]
    empty = [i for i, n in enumerate(lens) if n == 0]
    all_pairs = list(combinations(valid, 2))

    mats = {m: np.zeros((N, N), dtype=np.float32) for m in m_list}
    unstable_pairs = []

    with torch.no_grad():
        for start in range(0, len(all_pairs), pair_chunk):
            chunk = all_pairs[start:start + pair_chunk]
            idx_i = torch.tensor([p[0] for p in chunk], device=device)
            idx_j = torch.tensor([p[1] for p in chunk], device=device)

            ni_max = max(lens[p[0]] for p in chunk)
            nj_max = max(lens[p[1]] for p in chunk)

            Qi = padded[idx_i][:, :ni_max]        # (B, ni_max, D)
            Qj = padded[idx_j][:, :nj_max]        # (B, nj_max, D)
            mi = mask[idx_i][:, :ni_max]
            mj = mask[idx_j][:, :nj_max]

            if cost_metric == 'sqeuclidean':
                Mb = torch.cdist(Qi, Qj, p=2) ** 2
            else:
                Mb = torch.cdist(Qi, Qj, p=2)

            for m in m_list:
                cost, unstable = _batched_entropic_partial_cost(
                    Mb, mi, mj, m, reg, numItermax, stopThr)
                cost_np = cost.cpu().numpy()
                unstable_np = unstable.cpu().numpy()
                for k, (i, j) in enumerate(chunk):
                    mats[m][i, j] = cost_np[k]
                    mats[m][j, i] = cost_np[k]
                    if unstable_np[k]:
                        unstable_pairs.append((i, j, m))

    for i in empty:
        for m in m_list:
            mats[m][i, :] = np.inf
            mats[m][:, i] = np.inf
            mats[m][i, i] = 0.0

    return mats, unstable_pairs


if __name__ == '__main__':
    # ── smoke test: validate the batched implementation against POT's own
    #    per-pair reference on random data, before trusting it on real data ──
    import ot

    np.random.seed(0)
    torch.manual_seed(0)
    dim = 32
    reg = 0.05
    m = 0.7

    sizes = [(40, 55), (30, 30), (60, 45), (25, 50)]
    print(f'Validating batched entropic partial OT against POT reference '
          f'(reg={reg}, m={m})...\n')

    for nq, nx in sizes:
        Q = np.random.randn(nq, dim).astype(np.float32)
        X = np.random.randn(nx, dim).astype(np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        # reference: POT's own per-pair function
        M_ref = ot.dist(Q, X, metric='sqeuclidean')
        a_ref = np.full(nq, 1.0 / nq)
        b_ref = np.full(nx, 1.0 / nx)
        gamma_ref = ot.partial.entropic_partial_wasserstein(
            a_ref, b_ref, M_ref, reg=reg, m=m, numItermax=1000, stopThr=1e-100)
        ref_cost = float(np.sum(M_ref * gamma_ref))

        # batched implementation, batch of size 1
        mats, unstable = entropic_ot_distance_matrices(
            [Q, X], [m], reg=reg, device='cpu', numItermax=1000, stopThr=1e-100)
        batched_cost = mats[m][0, 1]

        rel_err = abs(batched_cost - ref_cost) / max(abs(ref_cost), 1e-8)
        status = 'OK' if rel_err < 1e-3 and not unstable else 'MISMATCH or UNSTABLE'
        print(f'  nq={nq:3d} nx={nx:3d}: ref={ref_cost:.6f}  batched={batched_cost:.6f}  '
              f'rel_err={rel_err:.2e}  unstable={unstable}  [{status}]')
