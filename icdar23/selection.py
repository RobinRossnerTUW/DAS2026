"""
selection.py
────────────
Foreign-page selection and per-trial contamination precompute.

All functions are stateless: page metadata is passed in as arguments
rather than read from module-level globals, so this module has no
dependency on execution order.

Public API
──────────
closest_bin_n(num_lines, bin_c)
    Lines mode: return the n_replace in [0, num_lines//2] whose actual
    contamination % is closest to the target bin centre.

precompute(trial_seed, valid_wps, valid_writers, valid_writer_to_pages,
           page_num_lines, page_line_map, page_num_patches)
    Build (foreign_map, selections) for one trial.
    foreign_map  : { wp -> fwp }          — single source page per query
    selections   : { wp -> { pct -> sel } }

compute_realised_pct(valid_wps, page_num_lines, page_num_patches)
    Return (realised_pct_per_level, mean_units_replaced, pages_per_level)
    based on page metadata alone (no random sampling needed).
"""

import random as _random
import numpy as np

import config


# ══════════════════════════════════════════════════════════════════════════════
# Lines-mode helpers
# ══════════════════════════════════════════════════════════════════════════════

def closest_bin_n(num_lines, bin_c):
    """
    Return the n_replace in [0, num_lines//2] whose actual contamination
    percentage (100 * n / num_lines) is closest to bin_c.
    """
    return min(range(0, num_lines // 2 + 1),
               key=lambda n: abs(100.0 * n / num_lines - bin_c))


def _choose_foreign_page_lines(wp, fw_pages, rng, page_num_lines):
    """
    Pick a single foreign page with enough lines for the hardest bin centre.
    Falls back to the largest page if nothing qualifies.
    """
    n_max = closest_bin_n(page_num_lines[wp], max(config.BIN_CENTERS))
    cands = [p for p in fw_pages if page_num_lines.get(p, 0) >= n_max]
    if cands:
        return rng.choice(cands)
    return max(fw_pages, key=lambda p: page_num_lines.get(p, 0))


def _precompute_lines(trial_seed, valid_wps, valid_writers,
                      valid_writer_to_pages, page_num_lines, page_line_map):
    rng = _random.Random(trial_seed)

    # ── choose one foreign source page per query ──────────────────────────
    foreign_map = {}
    for wp in valid_wps:
        fw       = rng.choice([w for w in valid_writers if w != wp[0]])
        foreign_map[wp] = _choose_foreign_page_lines(
            wp, valid_writer_to_pages[fw], rng, page_num_lines)

    # ── build per-level selections ────────────────────────────────────────
    selections = {}
    for wp in valid_wps:
        num_lines      = page_num_lines[wp]
        sorted_lines   = sorted(page_line_map[wp].keys())
        fwp            = foreign_map[wp]
        f_sorted_lines = sorted(page_line_map[fwp].keys())

        per_level, cache = {}, {}
        for bin_c in config.BIN_CENTERS:
            n = closest_bin_n(num_lines, bin_c)
            if n in cache:
                per_level[bin_c] = cache[n]; continue

            if n == 0:
                sel = {'n_replace': 0}
            else:
                n_for = min(n, len(f_sorted_lines))   # fwp large enough: rarely binds
                sel = {
                    'n_replace':        n,
                    'lines_to_replace': set(rng.sample(sorted_lines, n)),
                    'foreign_lines':    rng.sample(f_sorted_lines, n_for) if n_for else [],
                }
            cache[n] = sel
            per_level[bin_c] = sel
        selections[wp] = per_level

    return foreign_map, selections


# ══════════════════════════════════════════════════════════════════════════════
# Random-mode helpers
# ══════════════════════════════════════════════════════════════════════════════

def _choose_foreign_page_random(wp, fw_pages, rng, page_num_patches):
    """
    Pick a single foreign page with enough patches for the hardest pct level.
    Falls back to the largest page if nothing qualifies.
    """
    n_max = int(round(max(config.PCT_LEVELS) * page_num_patches[wp] / 100.0))
    cands = [p for p in fw_pages if page_num_patches.get(p, 0) >= n_max]
    if cands:
        return rng.choice(cands)
    return max(fw_pages, key=lambda p: page_num_patches.get(p, 0))


def _precompute_random(trial_seed, valid_wps, valid_writers,
                       valid_writer_to_pages, page_num_patches):
    rng    = _random.Random(trial_seed)
    np_rng = np.random.RandomState(trial_seed)

    # ── choose one foreign source page per query ──────────────────────────
    foreign_map = {}
    for wp in valid_wps:
        fw = rng.choice([w for w in valid_writers if w != wp[0]])
        foreign_map[wp] = _choose_foreign_page_random(
            wp, valid_writer_to_pages[fw], rng, page_num_patches)

    # ── build per-level selections ────────────────────────────────────────
    selections = {}
    for wp in valid_wps:
        n_own = page_num_patches[wp]
        n_for = page_num_patches[foreign_map[wp]]

        per_level, cache = {}, {}
        for pct in config.PCT_LEVELS:
            n = min(int(round(pct * n_own / 100.0)), n_own, n_for)
            if n in cache:
                per_level[pct] = cache[n]; continue

            if n == 0:
                sel = {'n_replace': 0}
            else:
                sel = {
                    'n_replace': n,
                    'dest_idx':  np_rng.choice(n_own, n, replace=False),
                    'src_idx':   np_rng.choice(n_for, n, replace=False),
                }
            cache[n] = sel
            per_level[pct] = sel
        selections[wp] = per_level

    return foreign_map, selections


# ══════════════════════════════════════════════════════════════════════════════
# Public dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def precompute(trial_seed, valid_wps, valid_writers, valid_writer_to_pages,
               page_num_lines=None, page_line_map=None, page_num_patches=None):
    """
    Build foreign_map and selections for one trial.
    Pass the metadata dicts appropriate for the active MODE.
    """
    if config.MODE == 'lines':
        return _precompute_lines(trial_seed, valid_wps, valid_writers,
                                 valid_writer_to_pages, page_num_lines, page_line_map)
    return _precompute_random(trial_seed, valid_wps, valid_writers,
                              valid_writer_to_pages, page_num_patches)


# ══════════════════════════════════════════════════════════════════════════════
# Realised contamination summary (deterministic from page metadata)
# ══════════════════════════════════════════════════════════════════════════════

def compute_realised_pct(valid_wps, page_num_lines=None, page_num_patches=None):
    """
    Return
    ──────
    realised_pct_per_level : { lvl -> mean actual contamination % }
    mean_units_replaced    : { lvl -> mean n_replace per page }
    pages_per_level        : { lvl -> number of distinct n_replace values }
    """
    realised, mean_units, pages_per = {}, {}, {}

    if config.MODE == 'lines':
        for bin_c in config.BIN_CENTERS:
            pcts, ns = [], []
            for wp in valid_wps:
                n = closest_bin_n(page_num_lines[wp], bin_c)
                pcts.append(100.0 * n / page_num_lines[wp])
                ns.append(n)
            realised[bin_c]    = float(np.mean(pcts))
            mean_units[bin_c]  = float(np.mean(ns))
            pages_per[bin_c]   = len(set(ns))
    else:
        n_patches_arr = np.array([page_num_patches[wp] for wp in valid_wps])
        for pct in config.PCT_LEVELS:
            ns = np.array([int(round(pct * page_num_patches[wp] / 100.0))
                           for wp in valid_wps])
            realised[pct]   = float(np.mean(100.0 * ns / n_patches_arr))
            mean_units[pct] = float(np.mean(ns))
            pages_per[pct]  = len(set(ns.tolist()))

    return realised, mean_units, pages_per
