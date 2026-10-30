"""
workers.py
──────────
Fork-safe multiprocessing workers for patch encoding.

Module-level globals are set by main.py BEFORE the Pool is created so that
forked child processes inherit them without pickling:

    import workers
    workers.WP_TO_KEY        = wp_to_key
    workers.PAGE_LINE_MAP    = page_line_map   # lines mode only
    workers.PAGE_NUM_PATCHES = page_num_patches
    workers.POOLING_CONFIGS  = pooling_configs
    workers.TRIM_PCA         = pca_patch       # None disables trim output entirely

Public API
──────────
worker_init(h5_path)   — Pool initializer; opens a per-process HDF5 handle.
encode_page_job(job)   — Pool map target; dispatches on config.MODE.
                         Returns (wp, { pct: (pooled_dict, patch_repr_or_None) }).
                         pooled_dict is unchanged: { pooling_name: vector }.
                         patch_repr is an (n_patches, TRIM_PCA_DIM) array, ready
                         for trimming.trimmed_distance_matrices, or None if
                         TRIM_PCA was not set.
"""

import h5py
import numpy as np

import config

# ── Fork-inherited globals (set by main.py before pool creation) ──────────────
WP_TO_KEY        = None
PAGE_LINE_MAP    = None    # lines mode: { wp -> {line_idx: [patch_idx, ...]} }
PAGE_NUM_PATCHES = None    # random mode: { wp -> int }
POOLING_CONFIGS  = None    # list of (name, pooling_obj)
TRIM_PCA         = None    # patch-level PCA (config.PCA_PATCH_PATH); None disables trim output
_WORKER_H5       = None    # per-process HDF5 handle


# ══════════════════════════════════════════════════════════════════════════════
# Pool initializer
# ══════════════════════════════════════════════════════════════════════════════

def worker_init(h5_path):
    global _WORKER_H5
    _WORKER_H5 = h5py.File(h5_path, 'r')


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read_norm(h5, key):
    arr   = h5[key][:].astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms < 1e-8, 1.0, norms)


def _encode(mixed):
    return {name: pooling.encode([mixed]) for name, pooling in POOLING_CONFIGS}


def _prep_patch_repr(mixed):
    """
    PCA-reduce + re-normalise the (already L2-normalised, contamination-mixed)
    raw patch array for use by trimming.trimmed_distance_matrices. Returns
    None if no patch-level PCA has been configured (trimming disabled).
    """
    if TRIM_PCA is None:
        return None
    reduced = TRIM_PCA.transform(mixed)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.where(norms < 1e-8, 1.0, norms)
    return reduced.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Lines-mode worker
# ══════════════════════════════════════════════════════════════════════════════

def _encode_page_job_lines(job):
    """
    job = (wp, fwp, per_level)

    per_level : { pct -> sel }
        sel['n_replace']        — int
        sel['lines_to_replace'] — set of own line indices
        sel['foreign_lines']    — list of line indices from fwp

    Selections with the same n_replace share one dict object; workers skip
    re-encoding via an id()-keyed cache (big win when bin centres collapse).
    fwp is guaranteed large enough for the hardest level (see selection.py).
    """
    wp, fwp, per_level = job
    h5           = _WORKER_H5
    line_map     = PAGE_LINE_MAP[wp]
    f_line_map   = PAGE_LINE_MAP[fwp]
    sorted_lines = sorted(line_map.keys())

    own     = _read_norm(h5, WP_TO_KEY[wp]);  n_own = own.shape[0]
    foreign = _read_norm(h5, WP_TO_KEY[fwp]); n_for = foreign.shape[0]

    cache, out = {}, {}
    for pct, sel in per_level.items():
        key = id(sel)
        if key in cache:
            out[pct] = cache[key]; continue

        if sel['n_replace'] == 0:
            mixed = own
        else:
            keep_idx = [i for li in sorted_lines
                        if li not in sel['lines_to_replace']
                        for i in line_map[li] if i < n_own]
            fgn_idx  = [i for fli in sel['foreign_lines']
                        for i in f_line_map.get(fli, []) if i < n_for]
            if keep_idx and fgn_idx:
                mixed = np.concatenate([own[keep_idx], foreign[fgn_idx]], axis=0)
            elif keep_idx:
                mixed = own[keep_idx]
            elif fgn_idx:
                mixed = foreign[fgn_idx]
            else:
                mixed = own

        result     = (_encode(mixed), _prep_patch_repr(mixed))
        cache[key] = result
        out[pct]   = result
    return wp, out


# ══════════════════════════════════════════════════════════════════════════════
# Random-mode worker
# ══════════════════════════════════════════════════════════════════════════════

def _encode_page_job_random(job):
    """
    job = (wp, fwp, per_level)

    per_level : { pct -> sel }
        sel['n_replace'] — int
        sel['dest_idx']  — np.ndarray  own patch positions to overwrite
        sel['src_idx']   — np.ndarray  source patch indices within fwp

    fwp is guaranteed large enough for the hardest level (see selection.py).
    """
    wp, fwp, per_level = job
    h5    = _WORKER_H5
    own   = _read_norm(h5, WP_TO_KEY[wp]); n_own = own.shape[0]

    needs_foreign = any(sel['n_replace'] > 0 for sel in per_level.values())
    if needs_foreign:
        foreign = _read_norm(h5, WP_TO_KEY[fwp]); n_for = foreign.shape[0]
    else:
        foreign = None; n_for = 0

    cache, out = {}, {}
    for pct, sel in per_level.items():
        key = id(sel)
        if key in cache:
            out[pct] = cache[key]; continue

        if sel['n_replace'] == 0:
            mixed = own
        else:
            dest = sel['dest_idx']; src = sel['src_idx']
            mask  = (dest < n_own) & (src < n_for)
            mixed = own.copy()
            if mask.any():
                mixed[dest[mask]] = foreign[src[mask]]

        result     = (_encode(mixed), _prep_patch_repr(mixed))
        cache[key] = result
        out[pct]   = result
    return wp, out


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher (MODE is fixed at import time; workers agree after fork)
# ══════════════════════════════════════════════════════════════════════════════

def encode_page_job(job):
    if config.MODE == 'lines':
        return _encode_page_job_lines(job)
    return _encode_page_job_random(job)
