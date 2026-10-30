import argparse, os, glob, json, sys
from datetime import datetime

import torch
import numpy as np
from sklearn.preprocessing import normalize

# ── DATASET SWITCH ────────────────────────────────────────────────────────────
# Set DATASET to "icdar" or "cvl"

DATASET = "icdar"

# ─────────────────────────────────────────────────────────────────────────────

if DATASET == "icdar":
    REPO        = "/data/rrossner/DAS2026/icdar23"
    EXPERIMENT  = "test_repository-2026-05-08-12-23"
    BASE_DIR    = "/data/rrossner/DAS2026/icdar23/experiments/test_repository-2026-05-08-12-23"
    EMBED_PATH  = os.path.join(BASE_DIR, "embeddings_SumPooling-PN0p4_512.npy")
    SAVE_PATH   = os.path.join(BASE_DIR, "reranked")
    RUN_NAME    = "icdar2017"

elif DATASET == "cvl":
    REPO        = "/data/rrossner/DAS2026/icdar23"   # reranking modules live here
    BASE_DIR    = "/data/rrossner/datasets/CVL/7663856"
    EMBED_PATH  = os.path.join(BASE_DIR, "embeddings_SumPooling-PN0p4_512.npy")
    SAVE_PATH   = os.path.join(BASE_DIR, "reranked")
    RUN_NAME    = "cvl"

else:
    raise ValueError(f"Unknown DATASET: {DATASET!r}  — choose 'icdar' or 'cvl'")

# ── ALGORITHM CONFIG ──────────────────────────────────────────────────────────
ALGORITHM   = "sgr"   # sgr | gnn | krnn
K           = 2
K1          = 1       # gnn only
K2          = 2       # gnn only
LAYERS      = 2
GAMMA       = 0.15    # sgr only

# ── GRID SEARCH CONFIG ────────────────────────────────────────────────────────
GRID_GAMMAS  = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
GRID_KS      = [1, 2, 3, 4]
GRID_LAYERS  = [1, 2]
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, REPO)
from reranking import gnn, krnn, sgr
from evaluators.retrieval import Retrieval


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(embed_path):
    writer_path = embed_path.replace("embeddings_", "writers_")
    assert os.path.isfile(embed_path),  f"Embeddings not found: {embed_path}"
    assert os.path.isfile(writer_path), f"Writers file not found: {writer_path}"
    writers = np.load(writer_path)
    X_np    = np.load(embed_path)
    log(f"embeddings {X_np.shape}  writers {writers.shape}  "
        f"unique writers {len(set(writers.tolist()))}")
    return X_np, writers


def print_debug(X_np, writers, k, gamma):
    norms  = np.linalg.norm(X_np, axis=1)
    print(f"[debug] embedding norms  min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}")
    print(f"[debug] embeddings       min={X_np.min():.4f}  max={X_np.max():.4f}  mean={X_np.mean():.4f}")

    X_norm = normalize(X_np, axis=1)
    sim    = X_norm @ X_norm.T
    np.fill_diagonal(sim, 0)
    print(f"[debug] cosine sim (off-diag)  min={sim.min():.4f}  max={sim.max():.4f}  mean={sim.mean():.4f}")

    correct_at_k = []
    for i in range(len(writers)):
        row    = sim[i].copy()
        row[i] = -1
        top_k  = np.argsort(row)[::-1][:k]
        correct_at_k.append(sum(writers[j] == writers[i] for j in top_k) / k)
    print(f"[debug] top-{k} neighbor precision (same writer): {np.mean(correct_at_k):.4f}")
    print(f"[debug] → if this is low, reranking will amplify errors")

    sim_t = torch.tensor(X_norm, dtype=torch.float32)
    S     = sim_t @ sim_t.T
    A     = torch.exp(-((1 - S) ** 2) / gamma)
    A_np  = A.numpy()
    np.fill_diagonal(A_np, 0)
    print(f"[debug] adjacency A (gamma={gamma})  min={A_np.min():.4f}  max={A_np.max():.4f}  mean={A_np.mean():.4f}")
    print(f"[debug] → if max is near 1 everywhere, gamma is too large")


def run_reranking(X_np, algorithm, k, k1, k2, layers, gamma):
    X = torch.tensor(X_np)
    if algorithm == "sgr":
        X_reranked, dist = sgr.sgr_reranking(X, k, layer=layers, gamma=gamma)
    elif algorithm == "krnn":
        X_reranked = krnn.kRNN(X, k)
        dist       = 1 - torch.mm(X_reranked, X_reranked.t())
    elif algorithm == "gnn":
        X_reranked, dist = gnn.gnn_reranking(X, k1, k2, layer=layers)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return X_reranked, dist


def evaluate(X_np, writers, dist_matrix):
    X_base = normalize(X_np, axis=1)
    D      = dist_matrix.numpy() if hasattr(dist_matrix, "numpy") else dist_matrix
    np.fill_diagonal(D, 0.0)

    _eval             = Retrieval(options={"distance": "cosine"})
    res_base, csv_base = _eval.eval(X_base, writers)
    res_rr,   csv_rr   = Retrieval.calc_map_from_distances(writers, D)
    return res_base, csv_base, res_rr, csv_rr


def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    log(f"dataset:    {DATASET}")
    log(f"embeddings: {args.embedding_path}")

    X_np, writers = load_data(args.embedding_path)

    # ── Optional baseline reference ───────────────────────────────────────────
    baseline_json = os.path.join(BASE_DIR, "metrics_baseline.json")
    if os.path.exists(baseline_json):
        ref = json.load(open(baseline_json))
        log(f"[ref] baseline  mAP: {ref['map']:.4f}   Top1: {ref['top1']:.4f}  ({ref.get('pooling','')})")

    if args.debug:
        print_debug(X_np, writers, args.k, args.gamma)

    # ── Grid search mode ──────────────────────────────────────────────────────
    if args.grid:
        log("=== GRID SEARCH (SGR) ===")
        X_norm      = normalize(X_np, axis=1)
        best_map    = -1
        best_params = {}
        results     = []

        total = len(GRID_GAMMAS) * len(GRID_KS) * len(GRID_LAYERS)
        done  = 0
        for gamma in GRID_GAMMAS:
            for k in GRID_KS:
                for layers in GRID_LAYERS:
                    done += 1
                    X            = torch.tensor(X_norm)
                    X_rr, dist   = sgr.sgr_reranking(X, k, layer=layers, gamma=gamma)
                    D            = dist.numpy()
                    np.fill_diagonal(D, 0.0)
                    res, csv     = Retrieval.calc_map_from_distances(writers, D)
                    entry        = dict(gamma=gamma, k=k, layers=layers,
                                        map=float(res["map"]), top1=float(res["top1"]))
                    results.append(entry)
                    print(f"  [{done:>2}/{total}]  gamma={gamma}  k={k}  layers={layers}"
                          f"  mAP={res['map']:.4f}  Top1={res['top1']:.4f}")
                    if res["map"] > best_map:
                        best_map    = res["map"]
                        best_params = entry.copy()

        log(f"=== BEST ===  mAP={best_map:.4f}  Top1={best_params['top1']:.4f}"
            f"  gamma={best_params['gamma']}  k={best_params['k']}  layers={best_params['layers']}")

        out = os.path.join(args.save_path, f"{args.run_name}_grid_search.json")
        json.dump({"dataset": DATASET, "best": best_params, "grid": results},
                  open(out, "w"), indent=2)
        log(f"Grid results saved → {out}")

        # Overwrite args with best params so we fall through to save the best run
        args.gamma  = best_params["gamma"]
        args.k      = best_params["k"]
        args.layers = best_params["layers"]
        log(f"Running final reranking with best params …")

    # ── Single reranking run ───────────────────────────────────────────────────
    log(f"=== RERANKING ({args.algorithm})  k={args.k}  layers={args.layers}  gamma={args.gamma} ===")
    X_reranked, dist = run_reranking(X_np, args.algorithm,
                                     args.k, args.k1, args.k2,
                                     args.layers, args.gamma)

    tag = f"{args.run_name}_{args.algorithm}"
    np.save(os.path.join(args.save_path, f"{tag}.npy"),      X_reranked.numpy())
    np.save(os.path.join(args.save_path, f"{tag}_dist.npy"), dist.numpy())

    # ── Evaluation ────────────────────────────────────────────────────────────
    log("=== EVALUATION ===")
    res_base, csv_base, res_rr, csv_rr = evaluate(X_np, writers, dist)

    print(f"Baseline  mAP: {res_base['map']:.4f}   Top1: {res_base['top1']:.4f}")
    print(f"Reranked  mAP: {res_rr['map']:.4f}   Top1: {res_rr['top1']:.4f}")
    print(f"Delta     mAP: {res_rr['map'] - res_base['map']:+.4f}"
          f"   Top1: {res_rr['top1'] - res_base['top1']:+.4f}")

    metrics = {
        "dataset":   DATASET,
        "algorithm": args.algorithm,
        "k":  args.k, "k1": args.k1, "k2": args.k2,
        "layers": args.layers, "gamma": args.gamma,
        "baseline": {k: float(v) for k, v in csv_base.items()},
        "reranked":  {k: float(v) for k, v in csv_rr.items()},
        "delta":     {k: float(csv_rr[k] - csv_base[k]) for k in csv_rr},
    }
    out = os.path.join(args.save_path, "metrics_reranked.json")
    json.dump(metrics, open(out, "w"), indent=2)
    log(f"Saved metrics → {out}")
    log(f"ALL DONE — reranked embeddings at {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", default=EMBED_PATH)
    parser.add_argument("--save-path",      default=SAVE_PATH)
    parser.add_argument("--algorithm",      default=ALGORITHM)
    parser.add_argument("--k",              type=int,   default=K)
    parser.add_argument("--k1",             type=int,   default=K1)
    parser.add_argument("--k2",             type=int,   default=K2)
    parser.add_argument("--layers",         type=int,   default=LAYERS)
    parser.add_argument("--gamma",          type=float, default=GAMMA)
    parser.add_argument("--run-name",       default=RUN_NAME)
    parser.add_argument("--grid",           action="store_true",
                        help="Run SGR grid search over gamma/k/layers, then save best")
    parser.add_argument("--debug",          action="store_true",
                        help="Print embedding/similarity/adjacency diagnostics")
    args = parser.parse_args()
    main(args)