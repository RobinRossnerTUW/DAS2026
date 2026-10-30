"""
compare_results.py
──────────────────
Reads the standardised output files from both pipelines and prints
three comparison tables:

  1. Baseline accuracy          (from metrics_baseline.json)
  2. Reranking impact           (from reranked/metrics_reranked.json)
  3. Contamination robustness   (from contamination/contamination_results*.csv)

Also saves a combined CSV suitable for a paper results table.

Usage:
    python compare_results.py \
        --icdar_exp  /path/to/historical-wi/experiments/7434382 \
        --cvl_exp    /path/to/cvl-database/experiments/7536496 \
        --out_dir    ./comparison

The script finds artifact subdirs automatically (same logic as the sbatch files).
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
from pathlib import Path

METRICS = ['map', 'top1', 'hard-1', 'hard-5', 'soft-1', 'soft-5']
SEP = '-' * 80


def find_artifact_dir(exp_dir: Path, pattern: str) -> Path:
    candidates = sorted(exp_dir.glob(f'experiments/{pattern}'))
    if not candidates:
        raise FileNotFoundError(
            f'No dir matching experiments/{pattern} under {exp_dir}')
    return candidates[-1]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Expected file not found: {path}')
    return json.loads(path.read_text())


def load_baseline(exp_dir: Path, artifact_pattern: str) -> dict:
    art = find_artifact_dir(exp_dir, artifact_pattern)
    return load_json(art / 'metrics_baseline.json')


def load_reranked(exp_dir: Path) -> dict:
    path = exp_dir / 'reranked' / 'metrics_reranked.json'
    return load_json(path)


def load_contamination(exp_dir: Path, csv_name: str, pkl_name: str = None) -> pd.DataFrame:
    path = exp_dir / 'contamination' / csv_name
    if not path.exists():
        raise FileNotFoundError(f'Contamination CSV not found: {path}')
    df = pd.read_csv(path)

    if 'contamination_pct' not in df.columns and 'lines_replaced' in df.columns:
        # recover avg_lines_per_page from the companion pkl
        pkl_path = exp_dir / 'contamination' / (pkl_name or csv_name.replace('.csv', '.pkl'))
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Need avg_lines_per_page but pkl not found: {pkl_path}")
        import pickle
        meta = pickle.load(open(pkl_path, 'rb'))
        avg_lines = meta['avg_lines_per_page']
        df['contamination_pct'] = df['lines_replaced'] / avg_lines * 100
        print(f'  CVL: avg_lines_per_page={avg_lines:.1f}, '
              f'pct range: {df["contamination_pct"].min():.1f}–{df["contamination_pct"].max():.1f}%')

    return df


def fmt(v, pct=True):
    if pct:
        return f'{v * 100:.2f}%'
    return f'{v:.4f}'


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--icdar_exp', required=True,
                    help='ICDAR experiment root dir (contains experiments/, reranked/, contamination/)')
parser.add_argument('--cvl_exp', required=True,
                    help='CVL experiment root dir')
parser.add_argument('--out_dir', default='./comparison',
                    help='where to save combined CSVs')
parser.add_argument('--icdar_artifact', default='test_repository-*',
                    help='glob pattern for ICDAR artifact subdir (default: test_repository-*)')
parser.add_argument('--cvl_artifact', default='cvl_baseline*',
                    help='glob pattern for CVL artifact subdir (default: cvl_baseline*)')
args = parser.parse_args()

icdar_dir = Path(args.icdar_exp)
cvl_dir   = Path(args.cvl_exp)
out_dir   = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — BASELINE ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('TABLE 1 — BASELINE ACCURACY')
print(SEP)

b_icdar = load_baseline(icdar_dir, args.icdar_artifact)
b_cvl   = load_baseline(cvl_dir,   args.cvl_artifact)

header  = f'{"Metric":<12}  {"ICDAR2017":>10}  {"CVL":>10}  {"Δ (CVL-ICDAR)":>14}'
print(header)
print('-' * len(header))

baseline_rows = []
for m in METRICS:
    vi = b_icdar.get(m, float('nan'))
    vc = b_cvl.get(m, float('nan'))
    delta = vc - vi
    print(f'{m:<12}  {fmt(vi):>10}  {fmt(vc):>10}  {delta*100:>+13.2f}%')
    baseline_rows.append({'metric': m, 'icdar2017': vi, 'cvl': vc, 'delta': delta})

pd.DataFrame(baseline_rows).to_csv(out_dir / 'table1_baseline.csv', index=False)
print(f'\n→ saved: {out_dir}/table1_baseline.csv')

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — RERANKING IMPACT
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('TABLE 2 — RERANKING IMPACT')
print(SEP)

r_icdar = load_reranked(icdar_dir)
r_cvl   = load_reranked(cvl_dir)

print(f'  Algorithm: ICDAR={r_icdar["algorithm"]}  CVL={r_cvl["algorithm"]}')
print()

col_w = 12
header2 = (f'{"Metric":<{col_w}}  '
           f'{"ICDAR base":>11}  {"ICDAR rr":>9}  {"ICDAR Δ":>9}  '
           f'{"CVL base":>9}  {"CVL rr":>8}  {"CVL Δ":>8}')
print(header2)
print('-' * len(header2))

rerank_rows = []
for m in METRICS:
    bi = r_icdar['baseline'].get(m, float('nan'))
    ri = r_icdar['reranked'].get(m, float('nan'))
    di = r_icdar['delta'].get(m, float('nan'))
    bc = r_cvl['baseline'].get(m, float('nan'))
    rc = r_cvl['reranked'].get(m, float('nan'))
    dc = r_cvl['delta'].get(m, float('nan'))
    print(f'{m:<{col_w}}  '
          f'{fmt(bi):>11}  {fmt(ri):>9}  {di*100:>+8.2f}%  '
          f'{fmt(bc):>9}  {fmt(rc):>8}  {dc*100:>+7.2f}%')
    rerank_rows.append({
        'metric': m,
        'icdar_baseline': bi, 'icdar_reranked': ri, 'icdar_delta': di,
        'cvl_baseline': bc,   'cvl_reranked': rc,   'cvl_delta': dc,
    })

pd.DataFrame(rerank_rows).to_csv(out_dir / 'table2_reranking.csv', index=False)
print(f'\n→ saved: {out_dir}/table2_reranking.csv')

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — CONTAMINATION ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('TABLE 3 — CONTAMINATION ROBUSTNESS  (SumPooling-PN0p4, mAP)')
print(SEP)

df_icdar = load_contamination(icdar_dir, 'contamination_results.csv')
df_cvl   = load_contamination(cvl_dir,   'contamination_results_cvl.csv',
                               pkl_name='contamination_results_cvl.pkl')

POOLING = 'SumPooling-PN0p4'
MAP_COL = f'{POOLING}_map_mean'
STD_COL = f'{POOLING}_map_std'

def bucket(df):
    df = df.copy()
    df['pct_bucket'] = (df['contamination_pct'] / 5).round() * 5
    return df.groupby('pct_bucket').first().reset_index()

bi = bucket(df_icdar)
bc = bucket(df_cvl)

merged = pd.merge(
    bi[['pct_bucket', MAP_COL, STD_COL]].rename(columns={MAP_COL: 'icdar_map', STD_COL: 'icdar_std'}),
    bc[['pct_bucket', MAP_COL, STD_COL]].rename(columns={MAP_COL: 'cvl_map',   STD_COL: 'cvl_std'}),
    on='pct_bucket', how='outer'
).sort_values('pct_bucket')

# ── TABLE 3a — bucketed (original) ───────────────────────────────────────────
print('\n[3a] Bucketed to nearest 5%')
header3 = f'{"Contam%":>8}  {"ICDAR mAP":>12}  {"CVL mAP":>10}  {"ICDAR drop":>11}  {"CVL drop":>10}'
print(header3)
print('-' * len(header3))

icdar_0    = merged.loc[merged['pct_bucket'] == 0, 'icdar_map'].values
cvl_0      = merged.loc[merged['pct_bucket'] == 0, 'cvl_map'].values
icdar_base = icdar_0[0] if len(icdar_0) else float('nan')
cvl_base   = cvl_0[0]   if len(cvl_0)   else float('nan')

contam_rows = []
for _, row in merged.iterrows():
    pct  = row['pct_bucket']
    im   = row.get('icdar_map', float('nan'))
    is_  = row.get('icdar_std', float('nan'))
    cm   = row.get('cvl_map',   float('nan'))
    cs   = row.get('cvl_std',   float('nan'))
    id_  = (im - icdar_base) * 100
    cd   = (cm - cvl_base)   * 100
    im_s = f'{im*100:.2f}±{is_*100:.2f}%' if not np.isnan(im) else 'n/a'
    cm_s = f'{cm*100:.2f}±{cs*100:.2f}%'  if not np.isnan(cm) else 'n/a'
    id_s = f'{id_:+.2f}pp' if not np.isnan(id_) else 'n/a'
    cd_s = f'{cd:+.2f}pp'  if not np.isnan(cd)  else 'n/a'
    print(f'{pct:>7.0f}%  {im_s:>12}  {cm_s:>10}  {id_s:>11}  {cd_s:>10}')
    contam_rows.append({'contamination_pct': pct,
                        'icdar_map': im, 'icdar_std': is_, 'icdar_drop_pp': id_,
                        'cvl_map':   cm, 'cvl_std':   cs,  'cvl_drop_pp':  cd})

pd.DataFrame(contam_rows).to_csv(out_dir / 'table3_contamination.csv', index=False)
print(f'\n→ saved: {out_dir}/table3_contamination.csv')

# ── TABLE 3b — raw per-dataset rows, all natural pct points ──────────────────
print('\n[3b] Detailed — natural contamination % per dataset')

# build raw detail from un-bucketed CSVs
def detail_rows(df, label):
    rows = []
    base = df.loc[df['contamination_pct'].idxmin(), MAP_COL]
    for _, r in df.iterrows():
        rows.append({
            'dataset':          label,
            'contamination_pct': round(r['contamination_pct'], 1),
            'mAP':              round(r[MAP_COL] * 100, 3),
            'mAP_std':          round(r[STD_COL] * 100, 3),
            'drop_pp':          round((r[MAP_COL] - base) * 100, 3),
        })
    return rows

detail_icdar = detail_rows(df_icdar, 'ICDAR2017')
detail_cvl   = detail_rows(df_cvl,   'CVL')
all_detail   = detail_icdar + detail_cvl
df_detail    = pd.DataFrame(all_detail)

# print as a pivot-style table
pivot = df_detail.pivot_table(
    index='contamination_pct', columns='dataset',
    values=['mAP', 'mAP_std', 'drop_pp'], aggfunc='first'
).round(3)
print(pivot.to_string())

detail_path = out_dir / 'table3b_contamination_detail.csv'
df_detail.to_csv(detail_path, index=False)
print(f'\n→ saved: {detail_path}')

# ── WANDB — line graph ────────────────────────────────────────────────────────
try:
    import wandb

    run = wandb.init(
        project='compare_icdar_cvl',
        name='contamination_robustness',
        config={
            'icdar_exp': str(icdar_dir),
            'cvl_exp':   str(cvl_dir),
            'pooling':   POOLING,
        },
        reinit=True,
    )

    # log each data point as a custom x-axis step
    for row in detail_icdar:
        wandb.log({
            'contamination_pct':  row['contamination_pct'],
            'ICDAR2017/mAP':      row['mAP'],
            'ICDAR2017/mAP_std':  row['mAP_std'],
            'ICDAR2017/drop_pp':  row['drop_pp'],
        }, step=int(row['contamination_pct'] * 10))   # unique int step

    for row in detail_cvl:
        wandb.log({
            'contamination_pct': row['contamination_pct'],
            'CVL/mAP':           row['mAP'],
            'CVL/mAP_std':       row['mAP_std'],
            'CVL/drop_pp':       row['drop_pp'],
        }, step=int(row['contamination_pct'] * 10))

    # also log a wandb Table so you can make custom charts in the UI
    wb_table = wandb.Table(dataframe=df_detail)
    wandb.log({'contamination_detail': wb_table})

    # custom line plot: mAP vs contamination_pct, one line per dataset
    wandb.log({
        'mAP_vs_contamination': wandb.plot.line(
            wb_table,
            x='contamination_pct',
            y='mAP',
            stroke='dataset',
            title='mAP vs Contamination % (SumPooling-PN0p4)',
        )
    })

    run.finish()
    print(f'\n→ wandb run logged to project: compare_icdar_cvl')

except Exception as e:
    print(f'\n[wandb] Skipped — {e}')

# ══════════════════════════════════════════════════════════════════════════════
# COMBINED PAPER TABLE — one row per dataset, all key numbers
# ══════════════════════════════════════════════════════════════════════════════
def get(d, key):
    return float(d.get(key, float('nan')))

paper_rows = [
    {
        'dataset':              'ICDAR2017',
        'baseline_mAP':         get(b_icdar, 'map'),
        'baseline_top1':        get(b_icdar, 'top1'),
        'baseline_hard1':       get(b_icdar, 'hard-1'),
        'baseline_soft1':       get(b_icdar, 'soft-1'),
        'reranked_mAP':         get(r_icdar['reranked'], 'map'),
        'reranked_top1':        get(r_icdar['reranked'], 'top1'),
        'rerank_delta_mAP':     get(r_icdar['delta'], 'map'),
        'contam_50pct_mAP':     merged.loc[merged['pct_bucket'] == 50, 'icdar_map'].values[0]
                                if 50 in merged['pct_bucket'].values else float('nan'),
    },
    {
        'dataset':              'CVL',
        'baseline_mAP':         get(b_cvl, 'map'),
        'baseline_top1':        get(b_cvl, 'top1'),
        'baseline_hard1':       get(b_cvl, 'hard-1'),
        'baseline_soft1':       get(b_cvl, 'soft-1'),
        'reranked_mAP':         get(r_cvl['reranked'], 'map'),
        'reranked_top1':        get(r_cvl['reranked'], 'top1'),
        'rerank_delta_mAP':     get(r_cvl['delta'], 'map'),
        'contam_50pct_mAP':     merged.loc[merged['pct_bucket'] == 50, 'cvl_map'].values[0]
                                if 50 in merged['pct_bucket'].values else float('nan'),
    },
]
pd.DataFrame(paper_rows).to_csv(out_dir / 'table_combined.csv', index=False)

print(f'\n{SEP}')
print('COMBINED PAPER TABLE')
print(SEP)
df_paper = pd.DataFrame(paper_rows)
print(df_paper.to_string(index=False, float_format='{:.4f}'.format))
print(f'\n→ saved: {out_dir}/table_combined.csv')
print(f'\nAll outputs written to: {out_dir}/')
