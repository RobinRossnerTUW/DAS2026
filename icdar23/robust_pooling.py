"""
robust_pooling.py
──────────────────
Robust alternatives to SumPooling for aggregating a page's patch embeddings
into a single descriptor, aimed at reducing the influence of contaminated
(foreign-writer) patches on the resulting page vector.

All three keep the exact same post-aggregation normalisation as the existing
baseline (`SumPooling('l2', pn_alpha=0.4)` → component-wise power-norm with
alpha, then L2-norm), so any difference in retrieval performance is
attributable to the aggregation step alone, not to a different final norm.

Interface (matches page_encodings.SumPooling / MaxPooling / LSEPooling):
    pooling = SomePooling(...)
    pooling.encode([patch_array_for_one_page])   ->  np.ndarray, shape (1, D)

Implemented
───────────
GeometricMedianPooling
    Weiszfeld's algorithm: the point minimising the sum of L2 distances to
    all patches. A minority of far-away (contaminated) patches has bounded
    influence on the geometric median, unlike the mean, which is dragged
    linearly by every outlier.

TrimmedMeanPooling
    Rank patches by distance to the page centroid, average only the closest
    (1 - trim_frac) fraction. Assumes contaminated patches sit farther from
    the genuine writer's centroid on average.

KMeansRejectionPooling
    k-means (k=2 by default) on the page's patches; keep only the majority
    cluster, discard the rest, then average. Assumes contamination forms its
    own cluster once it's a large enough fraction of the page.

A lightweight from-scratch k-means (numpy, k-means++ init, few iterations) is
used instead of sklearn's, since this can be called many times (once per
page per distinct n_replace per trial) and sklearn's per-call overhead adds
up. This keeps the extra cost close to the aggregation itself.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Shared post-aggregation normalisation (identical to the SumPooling baseline)
# ══════════════════════════════════════════════════════════════════════════════

def _power_norm_l2(x, alpha):
    """Component-wise signed power-norm, then L2-norm. x: (n, D)."""
    if alpha != 1.0:
        x = np.sign(x) * (np.abs(x) ** alpha)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-8, 1.0, n)


# ══════════════════════════════════════════════════════════════════════════════
# Geometric median (Weiszfeld's algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def _weiszfeld(patches, max_iter=30, tol=1e-5, eps=1e-6):
    """Geometric median of an (n, D) array of points."""
    if patches.shape[0] == 1:
        return patches[0].copy()
    y = patches.mean(axis=0)
    sq = np.einsum('ij,ij->i', patches, patches)   # ||x||^2, distances via matvec (no (n, D) temporaries)
    for _ in range(max_iter):
        dist  = np.sqrt(np.maximum(sq - 2.0 * (patches @ y) + y @ y, 0.0))
        dist  = np.where(dist < eps, eps, dist)
        w     = 1.0 / dist
        y_new = (w @ patches) / w.sum()
        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            break
        y = y_new
    return y


class GeometricMedianPooling:
    def __init__(self, pn_alpha=0.4, max_iter=30, tol=1e-5):
        self.pn_alpha = pn_alpha
        self.max_iter = max_iter
        self.tol      = tol

    def encode(self, patch_arrays):
        out = np.stack(
            [_weiszfeld(p, self.max_iter, self.tol) for p in patch_arrays], axis=0)
        return _power_norm_l2(out, self.pn_alpha)


# ══════════════════════════════════════════════════════════════════════════════
# Trimmed mean
# ══════════════════════════════════════════════════════════════════════════════

class TrimmedMeanPooling:
    def __init__(self, trim_frac=0.2, pn_alpha=0.4, min_keep=1):
        assert 0.0 <= trim_frac < 0.5, 'trim_frac must be in [0, 0.5)'
        self.trim_frac = trim_frac
        self.pn_alpha  = pn_alpha
        self.min_keep  = min_keep

    def _trimmed(self, patches):
        n = patches.shape[0]
        if n <= self.min_keep + 1:
            return patches.mean(axis=0)
        # no (n, D) temporaries: squared distances via matvec, kept mean via a 0/1 weight vector
        centroid = patches.mean(axis=0)
        sq       = np.einsum('ij,ij->i', patches, patches)
        d        = np.sqrt(np.maximum(sq - 2.0 * (patches @ centroid) + centroid @ centroid, 0.0))
        n_keep   = max(n - int(np.floor(n * self.trim_frac)), self.min_keep)
        keep_idx = np.argsort(d)[:n_keep]          # drop the farthest trim_frac
        w = np.zeros(n, dtype=patches.dtype); w[keep_idx] = 1.0
        return (w @ patches) / n_keep

    def encode(self, patch_arrays):
        out = np.stack([self._trimmed(p) for p in patch_arrays], axis=0)
        return _power_norm_l2(out, self.pn_alpha)


# ══════════════════════════════════════════════════════════════════════════════
# K-means rejection
# ══════════════════════════════════════════════════════════════════════════════

def _fast_kmeans_labels(X, k, n_iter=15, seed=0):
    """Minimal numpy k-means (k-means++ init). Returns (n,) int labels."""
    n   = X.shape[0]
    rng = np.random.RandomState(seed)

    centers    = np.empty((k, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.randint(n)]
    sq = np.einsum('ij,ij->i', X, X)
    sqd = lambda c: np.maximum(sq - 2.0 * (X @ c) + c @ c, 0.0)   # ||x - c||^2, no (n, D) temporary
    d2 = sqd(centers[0])
    for c in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 1e-12 else np.full(n, 1.0 / n)
        centers[c] = X[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, sqd(centers[c]))

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        # ||x - c||^2 up to the per-row constant ||x||^2 (no (n, k, D) temporary)
        dists  = (centers ** 2).sum(axis=1)[None, :] - 2.0 * (X @ centers.T)
        labels = dists.argmin(axis=1)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = (mask.astype(X.dtype) @ X) / mask.sum()   # cluster mean without copying X[mask]
    return labels


class KMeansRejectionPooling:
    def __init__(self, k=2, pn_alpha=0.4, min_patches=None,
                min_cluster_frac=0.15, seed=0):
        self.k                = k
        self.pn_alpha         = pn_alpha
        self.min_patches      = min_patches or (2 * k)
        self.min_cluster_frac = min_cluster_frac
        self.seed             = seed

    def _reject(self, patches):
        n = patches.shape[0]
        if n < self.min_patches:
            return patches.mean(axis=0)
        labels = _fast_kmeans_labels(patches, self.k, seed=self.seed)
        sizes  = np.bincount(labels, minlength=self.k)
        keep_c = int(np.argmax(sizes))
        mask   = labels == keep_c
        if mask.sum() < max(1, int(self.min_cluster_frac * n)):
            return patches.mean(axis=0)          # degenerate split, fall back
        return (mask.astype(patches.dtype) @ patches) / mask.sum()

    def encode(self, patch_arrays):
        out = np.stack([self._reject(p) for p in patch_arrays], axis=0)
        return _power_norm_l2(out, self.pn_alpha)
