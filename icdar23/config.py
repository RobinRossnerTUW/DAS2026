"""
config.py
─────────
All configuration constants, CLI overrides, and resolved paths.
Every other module imports from here. No heavy logic lives here.
"""

import os, argparse, random
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Top-level toggles  (edit these or override via CLI)
# ══════════════════════════════════════════════════════════════════════════════
DATASET = 'cvl'   # 'cvl' | 'icdar'
MODE    = 'random'   # 'lines' | 'random'

# PCT_LEVELS       = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
PCT_LEVELS       = [0, 25, 50]
MIN_LINES        = 2          # lines mode: pages with fewer lines are dropped
N_TRIALS         = 5
USE_WANDB        = True
WANDB_RUN        = None       # wandb run name; None = auto
EVAL_FOREIGN_MAP = False       # compute foreign-writer mAP alongside original

# ── Trimmed (partial Chamfer) distance, evaluated alongside SumPooling ────────
USE_TRIMMING  = False
TRIM_EPS      = [0.10, 0.25, 0.50]   # fraction of worst query-patch matches discarded
TRIM_PCA_DIM  = 128                  # dim of the *patch-level* PCA (separate from PCA_PATH)
TRIM_DEVICE   = 'cuda'               # falls back to 'cpu' automatically if unavailable
TRIM_J_CHUNK  = 200                 # candidate pages per inner batch; set an int if you hit GPU OOM

# ── Optimal Transport distance, evaluated alongside SumPooling ───────────────
USE_OT          = False
OT_M_LEVELS     = [0.5, 0.7, 0.9]   # fraction of total mass transported (POT's `m`)
OT_MEM_LIMIT_GB = None                # total RAM budget across all OT workers (safety net)
OT_GPUID        = '4'                # inert for the exact OT solver itself — see ot_distance.py
CHECKPOINT_EVERY_HOURS = 12

# ── TopK-OT: SumPooling retrieval, exact partial OT rerank on top-k only ────
USE_TOPK_OT      = True
TOPK_OT_K        = 2
TOPK_OT_M_LEVELS = [0.5, 0.7, 0.9]   # same m semantics as OT_M_LEVELS

# --- Entropic OT (GPU) distance, evaluated alongside SumPooling ---
USE_ENTROPIC_OT        = False
ENTROPIC_OT_REG        = 0.03     # start here; watch the unstable-pairs count printed at runtime
ENTROPIC_OT_M_LEVELS    = [0.5, 0.7, 0.9]
ENTROPIC_OT_DEVICE      = 'cuda'
ENTROPIC_OT_PAIR_CHUNK  = 128     # GPU memory is not your constraint here — raise this and see
ENTROPIC_OT_NUMITERMAX  = 1000
ENTROPIC_OT_STOPTHR     = 1e-6

# ══════════════════════════════════════════════════════════════════════════════
# Per-dataset paths and metadata
# ══════════════════════════════════════════════════════════════════════════════
DATASET_CONFIGS = {
    'cvl': {
        'experiment_dir': '/data/rrossner/outlierDetection/icdar23/experiments/cvl_cluster_285k-2026-08-27-09-43',
        'patches_dir':    '/data/rrossner/DAS2026/cvl/test_patches_coords',
        'xml_dir':        '/data/rrossner/datasets/CVL/xml',
        'page_ns':        '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}',
        'seed':           40,
        'n_workers':      32,
    },
    'icdar': {
        'experiment_dir': '/data/rrossner/DAS2026/icdar23/experiments/test_repository-2026-05-08-12-23',
        'patches_dir':    '/data/rrossner/DAS2026/icdar/test_patches_coords',
        'xml_dirs': [
            '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-1/page',
            '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-2/page',
            '/data/rrossner/datasets/ICDAR17/ScriptNet-HistoricalWI-2017-color-3/page',
        ],
        'page_ns':        '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}',
        'seed':           43,
        'n_workers':      64,
    },
}

WANDB_PROJECTS = {
    ('cvl',   'lines'):  'cvl-contamination-lines',
    ('cvl',   'random'): 'cvl-contamination-random-patches',
    ('icdar', 'lines'):  'icdar23-contamination-lines',
    ('icdar', 'random'): 'icdar23-contamination-random-patches',
}

# ══════════════════════════════════════════════════════════════════════════════
# CLI overrides
# ══════════════════════════════════════════════════════════════════════════════
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--dataset',  choices=['cvl', 'icdar'], default=None)
_parser.add_argument('--mode',     choices=['lines', 'random'], default=None)
_parser.add_argument('--no-wandb', action='store_true')
_args, _ = _parser.parse_known_args()
if _args.dataset  is not None: DATASET   = _args.dataset
if _args.mode     is not None: MODE      = _args.mode
if _args.no_wandb:             USE_WANDB = False

assert DATASET in DATASET_CONFIGS, f'unknown dataset {DATASET!r}'
assert MODE    in ('lines', 'random'), f'unknown mode {MODE!r}'

# ══════════════════════════════════════════════════════════════════════════════
# Resolved paths and derived constants
# ══════════════════════════════════════════════════════════════════════════════
CFG            = DATASET_CONFIGS[DATASET]
EXPERIMENT_DIR = CFG['experiment_dir']
H5_PATH        = os.path.join(EXPERIMENT_DIR, 'raw_patches_test.h5')
INDEX_PATH     = os.path.join(EXPERIMENT_DIR, 'patch_index_test.pkl')
PCA_PATH       = os.path.join(EXPERIMENT_DIR, 'pca.pkl')
PCA_PATCH_PATH = os.path.join(EXPERIMENT_DIR, 'pca_patch.pkl')   # patch-level PCA, for trimming
PATCHES_DIR    = CFG['patches_dir']
PAGE_NS        = CFG['page_ns']
SEED           = CFG['seed']
N_WORKERS      = CFG['n_workers']

# Normalise to list — ICDAR has three colour sub-dirs, CVL has one.
XML_DIRS = CFG.get('xml_dirs', [CFG.get('xml_dir')])

MODE_NAME     = 'lines' if MODE == 'lines' else 'random_patches'
OUT_DIR       = os.path.join(EXPERIMENT_DIR, f'contamination_results_{MODE_NAME}')
FILE_STEM     = f'contamination_results_{DATASET}_{MODE_NAME}'
WANDB_PROJECT = WANDB_PROJECTS[(DATASET, MODE)]
BIN_CENTERS   = PCT_LEVELS   # alias used in lines-mode code paths

# ══════════════════════════════════════════════════════════════════════════════
# Environment, seeds, output directory
# ══════════════════════════════════════════════════════════════════════════════
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
