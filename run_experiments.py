# run_experiments.py
# End-to-end experiment driver for CS5510 HW1 reconstruction attack.
# Produces per-trial results, summaries, and plots for three defenses.

import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

from ps2_starter import (
    data, pub, target,
    make_random_predicate,
    execute_subsetsums_exact,
    execute_subsetsums_round,
    execute_subsetsums_noise,
    execute_subsetsums_sample,
    reconstruction_attack
)

# -------------------- Configuration --------------------

# Reproducibility
import random
GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# Experiment grid
N = len(data)                      # dataset size
NUM_QUERIES_FACTOR = 2             # use 2n random queries per trial
K = NUM_QUERIES_FACTOR * N
PARAMS = list(range(1, N + 1))     # parameter values: 1..n
RUNS_PER_PARAM = 10                # number of trials per parameter

# Output directory
OUTDIR = "."

# File names
RESULTS_FN = {
    "round":  os.path.join(OUTDIR, "results_round.csv"),
    "noise":  os.path.join(OUTDIR, "results_noise.csv"),
    "sample": os.path.join(OUTDIR, "results_sample.csv"),
}
SUMMARY_FN = {
    "round":  os.path.join(OUTDIR, "summary_round.csv"),
    "noise":  os.path.join(OUTDIR, "summary_noise.csv"),
    "sample": os.path.join(OUTDIR, "summary_sample.csv"),
}
COMBINED_SUMMARY_FN = os.path.join(OUTDIR, "reconstruction_defense_summary_full.csv")

# Plot names
PLOTS = {
    "round_rmse":   os.path.join(OUTDIR, "rmse_vs_param_round.png"),
    "round_succ":   os.path.join(OUTDIR, "success_vs_param_round.png"),
    "round_trade":  os.path.join(OUTDIR, "tradeoff_round.png"),
    "noise_rmse":   os.path.join(OUTDIR, "rmse_vs_param_noise.png"),
    "noise_succ":   os.path.join(OUTDIR, "success_vs_param_noise.png"),
    "noise_trade":  os.path.join(OUTDIR, "tradeoff_noise.png"),
    "sample_rmse":  os.path.join(OUTDIR, "rmse_vs_param_sample.png"),
    "sample_succ":  os.path.join(OUTDIR, "success_vs_param_sample.png"),
    "sample_trade": os.path.join(OUTDIR, "tradeoff_sample.png"),
}

# -------------------- Helpers --------------------

def rmse(approx, exact):
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return float(np.sqrt(np.mean((approx - exact) ** 2)))

def reconstruction_success(pred, truth):
    pred = np.asarray(pred).astype(int)
    truth = np.asarray(truth).astype(int)
    return float(np.mean(pred == truth))

def majority_baseline(y):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y)
    return counts.max() / len(y)

def find_transition(summary_df, majority_frac, eps=1e-6):
    """Return the smallest param_value whose avg_success <= majority_frac + eps."""
    tmp = summary_df.sort_values("param_value")
    for _, row in tmp.iterrows():
        if row["avg_success"] <= majority_frac + eps:
            return int(row["param_value"]), float(row["avg_success"])
    # If never drops to baseline, return the last point
    last = tmp.iloc[-1]
    return int(last["param_value"]), float(last["avg_success"])

def ensure_parent_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -------------------- Core experiment loop --------------------

def run_all(defense_kind, params, runs_per_param, k):
    """Runs all trials for one defense. Returns (per_trial_df, summary_df)."""
    rows = []

    iterator = params
    if _HAS_TQDM:
        iterator = tqdm(params, desc=f"defense={defense_kind}")

    for p in iterator:
        for trial in range(runs_per_param):
            # New random predicates each trial
            preds = [make_random_predicate() for _ in range(k)]
            exact = execute_subsetsums_exact(preds)

            if defense_kind == "round":
                approx = execute_subsetsums_round(p, preds)
            elif defense_kind == "noise":
                approx = execute_subsetsums_noise(p, preds)
            elif defense_kind == "sample":
                approx = execute_subsetsums_sample(p, preds)
            else:
                raise ValueError("Unknown defense kind")

            recon = reconstruction_attack(data[pub], preds, approx)
            row = {
                "defense": defense_kind,
                "param_value": int(p),
                "trial": int(trial),
                "rmse": rmse(approx, exact),
                "success": reconstruction_success(recon, data[target].values),
            }
            rows.append(row)

    df = pd.DataFrame(rows, columns=["defense","param_value","trial","rmse","success"])
    ensure_parent_dir(RESULTS_FN[defense_kind])
    df.to_csv(RESULTS_FN[defense_kind], index=False)

    summary = df.groupby(["defense","param_value"], as_index=False).agg(
        avg_rmse=("rmse","mean"), std_rmse=("rmse","std"),
        avg_success=("success","mean"), std_success=("success","std")
    )
    ensure_parent_dir(SUMMARY_FN[defense_kind])
    summary.to_csv(SUMMARY_FN[defense_kind], index=False)

    return df, summary

# -------------------- Plotting --------------------

def plot_with_errorbars(summary_df, xlab, ylab, title, outpath,
                        xcol="param_value", ycol="avg_rmse", yerrcol="std_rmse",
                        baseline=None, transition_param=None):
    x = summary_df[xcol].values
    y = summary_df[ycol].values
    yerr = summary_df[yerrcol].values

    plt.figure(figsize=(9, 5))
    plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3)
    if baseline is not None and "success" in ylab.lower():
        # draw horizontal majority baseline
        plt.axhline(baseline, linestyle="--")
    if transition_param is not None:
        plt.axvline(transition_param, linestyle="--")
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title); plt.grid(True)
    ensure_parent_dir(outpath)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

def plot_tradeoff(summary_df, title, outpath):
    # Scatter: x = avg_rmse, y = avg_success
    x = summary_df["avg_rmse"].values
    y = summary_df["avg_success"].values
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x, y)
    plt.xlabel("RMSE")
    plt.ylabel("Reconstruction success")
    plt.title(title)
    plt.grid(True)
    ensure_parent_dir(outpath)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

# -------------------- Main --------------------

def main():
    print(f"Dataset size n = {N}")
    print(f"Queries per trial k = {K}")
    print(f"Params 1..n  ({len(PARAMS)} values)")
    print(f"Runs per param = {RUNS_PER_PARAM}\n")

    t0 = time.time()

    # Run all defenses
    df_round, summary_round = run_all("round", PARAMS, RUNS_PER_PARAM, K)
    df_noise, summary_noise = run_all("noise", PARAMS, RUNS_PER_PARAM, K)
    df_sample, summary_sample = run_all("sample", PARAMS, RUNS_PER_PARAM, K)

    # Combined summary
    combined = pd.concat([summary_round, summary_noise, summary_sample], ignore_index=True)
    ensure_parent_dir(COMBINED_SUMMARY_FN)
    combined.to_csv(COMBINED_SUMMARY_FN, index=False)

    # Baseline and transitions
    maj = majority_baseline(data[target].values)
    r_transition = find_transition(summary_round, maj)
    n_transition = find_transition(summary_noise, maj)
    s_transition = find_transition(summary_sample, maj)

    print("\nMajority baseline (success fraction):", maj)
    print("Transition ~ baseline (param, avg_success):")
    print("  R (rounding):", r_transition)
    print("  sigma (noise):", n_transition)
    print("  t (sampling):", s_transition)

    # Plots (with baseline and transition lines where applicable)
    plot_with_errorbars(summary_round, xlab="R (round)", ylab="RMSE",
                        title="RMSE vs R (rounding)", outpath=PLOTS["round_rmse"],
                        baseline=None, transition_param=r_transition[0])
    plot_with_errorbars(summary_round, xlab="R (round)", ylab="Success",
                        title="Reconstruction success vs R (rounding)", outpath=PLOTS["round_succ"],
                        baseline=maj, transition_param=r_transition[0])
    plot_tradeoff(summary_round, "Trade-off (rounding)", PLOTS["round_trade"])

    plot_with_errorbars(summary_noise, xlab="sigma (noise)", ylab="RMSE",
                        title="RMSE vs sigma (Gaussian noise)", outpath=PLOTS["noise_rmse"],
                        baseline=None, transition_param=n_transition[0])
    plot_with_errorbars(summary_noise, xlab="sigma (noise)", ylab="Success",
                        title="Reconstruction success vs sigma (Gaussian noise)", outpath=PLOTS["noise_succ"],
                        baseline=maj, transition_param=n_transition[0])
    plot_tradeoff(summary_noise, "Trade-off (Gaussian noise)", PLOTS["noise_trade"])

    plot_with_errorbars(summary_sample, xlab="t (subsample size)", ylab="RMSE",
                        title="RMSE vs t (subsampling)", outpath=PLOTS["sample_rmse"],
                        baseline=None, transition_param=s_transition[0])
    plot_with_errorbars(summary_sample, xlab="t (subsample size)", ylab="Success",
                        title="Reconstruction success vs t (subsampling)", outpath=PLOTS["sample_succ"],
                        baseline=maj, transition_param=s_transition[0])
    plot_tradeoff(summary_sample, "Trade-off (subsampling)", PLOTS["sample_trade"])

    t1 = time.time()
    print(f"\nWrote:")
    for k, v in RESULTS_FN.items():  print(" -", v)
    for k, v in SUMMARY_FN.items():  print(" -", v)
    print(" -", COMBINED_SUMMARY_FN)
    for k, v in PLOTS.items():       print(" -", v)
    print(f"\nTotal runtime: {t1 - t0:.2f}s")

if __name__ == "__main__":
    main()
