"""
tests/analyse.py
----------------
Visualisation and analysis for all pipeline test stages.

Usage
-----
    cd full_pipeline

    # Navigation accuracy
    python -m tests.analyse nav

    # Gripper IO (Stage 1)
    python -m tests.analyse grasp

    # YOLO Verify classifier (Stage 2)
    python -m tests.analyse verify

    # Transit slip detection (Stage 3)
    python -m tests.analyse transit

    # Recovery success
    python -m tests.analyse recovery

    # All stages at once
    python -m tests.analyse all

Outputs
-------
Each command opens matplotlib figures and saves PNG copies to
tests/results/figures/<stage>_*.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def _load(stage: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{stage}_results.csv"
    if not path.exists():
        print(f"[!] No results file for stage '{stage}': {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[+] Loaded {len(df)} rows from {path}")
    return df


def _save(fig, name: str):
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"    Saved → {p}")


def _style():
    plt.rcParams.update({
        "font.family":    "sans-serif",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "figure.dpi":         120,
    })


# ═════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
def analyse_nav():
    _style()
    df = _load("nav")
    df = df.dropna(subset=["actual_x", "actual_y", "error_mm"])

    modes  = df["detection_mode"].unique()
    colors = {"aruco": "#2196F3", "yolo_only": "#FF9800", "unknown": "#9E9E9E"}

    # ── Fig 1: XY scatter — estimated hover vs measured ground truth ──────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for mode in modes:
        m = df[df["detection_mode"] == mode]
        c = colors.get(mode, "#9E9E9E")
        ax.scatter(m["hover_x"], m["hover_y"],
                   label=f"Hover TCP ({mode})", marker="x", s=80, c=c, zorder=3)
        ax.scatter(m["actual_x"], m["actual_y"],
                   label=f"Ground truth ({mode})", marker="o", s=60,
                   facecolors="none", edgecolors=c, zorder=3)
        # error lines
        for _, r in m.iterrows():
            ax.plot([r.hover_x, r.actual_x], [r.hover_y, r.actual_y],
                    c=c, alpha=0.4, lw=1)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Navigation: Estimated hover vs. Ground-truth tag XY")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    _save(fig, "nav_scatter.png")
    plt.show(block=False)

    # ── Fig 2: Error distribution (box + strip) ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Navigation XY Error (mm)", fontsize=13)

    # Box-plot by mode
    ax = axes[0]
    data_by_mode = [df[df["detection_mode"] == m]["error_mm"].dropna().values
                    for m in modes]
    bplot = ax.boxplot(data_by_mode, patch_artist=True, notch=False,
                       medianprops=dict(color="white", lw=2))
    for patch, mode in zip(bplot["boxes"], modes):
        patch.set_facecolor(colors.get(mode, "#9E9E9E"))
    ax.set_xticklabels(modes, rotation=10)
    ax.set_ylabel("XY error (mm)")
    ax.set_title("By detection mode")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # CDF overall
    ax = axes[1]
    errors = np.sort(df["error_mm"].dropna().values)
    cdf    = np.arange(1, len(errors) + 1) / len(errors)
    ax.step(errors, cdf, where="post", color="#4CAF50", lw=2)
    # percentile lines
    for pct, ls in [(50, "--"), (90, ":")]:
        v = np.percentile(errors, pct)
        ax.axvline(v, color="gray", ls=ls, lw=1)
        ax.text(v + 0.3, pct / 100 - 0.05, f"P{pct}={v:.1f}mm",
                fontsize=8, color="gray")
    ax.set_xlabel("XY error (mm)"); ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF — all trials")

    plt.tight_layout()
    _save(fig, "nav_error_dist.png")
    plt.show(block=False)

    # ── Fig 3: Error vs. trial index (drift / warm-up effect) ────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for mode in modes:
        m = df[df["detection_mode"] == mode]
        c = colors.get(mode, "#9E9E9E")
        ax.plot(m["trial_id"], m["error_mm"], "o-", color=c, label=mode,
                alpha=0.8, lw=1.5, markersize=5)
    ax.set_xlabel("Trial"); ax.set_ylabel("XY error (mm)")
    ax.set_title("Navigation error over trials")
    ax.legend(fontsize=8)
    _save(fig, "nav_error_vs_trial.png")
    plt.show(block=False)

    # ── Stats summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  NAVIGATION ACCURACY — SUMMARY")
    print("=" * 50)
    for mode in modes:
        e = df[df["detection_mode"] == mode]["error_mm"].dropna()
        print(f"\n  Mode: {mode}  (n={len(e)})")
        print(f"    Mean  : {e.mean():.2f} mm")
        print(f"    Median: {e.median():.2f} mm")
        print(f"    Std   : {e.std():.2f} mm")
        print(f"    P90   : {e.quantile(0.90):.2f} mm")
        print(f"    Max   : {e.max():.2f} mm")
    print()

    if len(modes) == 2 and "aruco" in modes and "yolo_only" in modes:
        e_a = df[df["detection_mode"] == "aruco"]["error_mm"].dropna()
        e_y = df[df["detection_mode"] == "yolo_only"]["error_mm"].dropna()
        if len(e_a) >= 5 and len(e_y) >= 5:
            _, p = stats.mannwhitneyu(e_a, e_y, alternative="two-sided")
            print(f"  Mann-Whitney U test (ArUco vs YOLO-only):  p={p:.4f}")
            sig = "Significant" if p < 0.05 else "Not significant"
            print(f"  → {sig} difference (α=0.05)")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# GRASP (Stage 1 — physical IO)
# ═════════════════════════════════════════════════════════════════════════════
def analyse_grasp():
    _style()
    df = _load("grasp")

    # Expected columns: trial_id, object_desc, true_label, width_mm,
    #                   force_detected, predicted_label, correct

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Stage 1 — Gripper IO (width + force)", fontsize=13)

    # Width distribution by true label
    ax = axes[0]
    for label, c in [("holding", "#4CAF50"), ("missed", "#F44336")]:
        sub = df[df["true_label"] == label]["width_mm"].dropna()
        ax.hist(sub, bins=12, color=c, alpha=0.7, label=label)
    ax.axvline(11.0, color="black", ls="--", lw=1.5, label="threshold 11 mm")
    ax.set_xlabel("Calibrated jaw width (mm)"); ax.set_ylabel("Count")
    ax.set_title("Width distribution by ground truth")
    ax.legend()

    # Confusion matrix
    ax = axes[1]
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    labels = ["holding", "missed"]
    cm = confusion_matrix(df["true_label"], df["predicted_label"],
                          labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix (width check)")

    plt.tight_layout()
    _save(fig, "grasp_io.png")
    plt.show(block=False)

    acc = (df["true_label"] == df["predicted_label"]).mean()
    print(f"\n  Stage 1 accuracy: {acc*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# VERIFY (Stage 2 — YOLO classifier)
# ═════════════════════════════════════════════════════════════════════════════
def analyse_verify():
    _style()
    df = _load("verify")
    # Expected columns: true_label, yolo_conf, result

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Stage 2 — YOLO Verify Classifier", fontsize=13)

    ax = axes[0]
    for label, c in [("holding", "#4CAF50"), ("empty", "#F44336")]:
        sub = df[df["true_label"] == label]["yolo_conf"].dropna()
        ax.hist(sub, bins=15, color=c, alpha=0.7, label=label)
    ax.axvline(0.90, color="black", ls="--", lw=1.5, label="threshold 0.90")
    ax.set_xlabel("p(holding)"); ax.set_ylabel("Count")
    ax.set_title("Confidence distribution by ground truth")
    ax.legend()

    ax = axes[1]
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    labels = ["holding", "empty"]
    cm = confusion_matrix(df["true_label"], df["result"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title("Confusion matrix")

    plt.tight_layout()
    _save(fig, "verify_cls.png")
    plt.show(block=False)

    acc = (df["true_label"] == df["result"]).mean()
    print(f"\n  Stage 2 accuracy: {acc*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# TRANSIT (Stage 3 — slip detection)
# ═════════════════════════════════════════════════════════════════════════════
def analyse_transit():
    _style()
    df = _load("transit")
    # Expected columns: scenario, true_outcome, detected_outcome,
    #                   frames_to_detect, transit_speed_ms

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Stage 3 — Transit Slip Detection", fontsize=13)

    ax = axes[0]
    scenarios = df["scenario"].unique()
    colors    = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))
    for sc, c in zip(scenarios, colors):
        sub = df[df["scenario"] == sc]["frames_to_detect"].dropna()
        ax.hist(sub, bins=8, color=c, alpha=0.7, label=sc)
    ax.set_xlabel("Frames to detect"); ax.set_ylabel("Count")
    ax.set_title("Detection latency (frames)")
    ax.legend()

    ax = axes[1]
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    labels = ["arrived", "slip", "empty"]
    labels = [l for l in labels if l in df["true_outcome"].values]
    cm = confusion_matrix(df["true_outcome"], df["detected_outcome"],
                          labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Oranges")
    ax.set_title("Confusion matrix")

    plt.tight_layout()
    _save(fig, "transit_slip.png")
    plt.show(block=False)


# ═════════════════════════════════════════════════════════════════════════════
# RECOVERY
# ═════════════════════════════════════════════════════════════════════════════
def analyse_recovery():
    _style()
    df = _load("recovery")
    # Expected columns: failure_type, recovered, time_to_recover_s, notes

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Recovery — Success Rate & Time", fontsize=13)

    ax = axes[0]
    summary = df.groupby("failure_type")["recovered"].mean() * 100
    bars = ax.bar(summary.index, summary.values,
                  color=["#4CAF50" if v >= 70 else "#F44336" for v in summary.values])
    ax.set_ylabel("Recovery success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Success rate by failure type")
    for bar, v in zip(bars, summary.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    for ft in df["failure_type"].unique():
        sub = df[(df["failure_type"] == ft) & (df["recovered"] == 1)]["time_to_recover_s"]
        if not sub.empty:
            ax.scatter([ft] * len(sub), sub, alpha=0.6, zorder=3)
    ax.set_ylabel("Time to recover (s)")
    ax.set_title("Recovery time by failure type")

    plt.tight_layout()
    _save(fig, "recovery.png")
    plt.show(block=False)

    print("\n  Recovery summary:")
    print(summary.to_string())


# ═════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═════════════════════════════════════════════════════════════════════════════
STAGES = {
    "nav":      analyse_nav,
    "grasp":    analyse_grasp,
    "verify":   analyse_verify,
    "transit":  analyse_transit,
    "recovery": analyse_recovery,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in list(STAGES) + ["all"]:
        print(f"Usage:  python -m tests.analyse <stage>")
        print(f"Stages: {', '.join(list(STAGES) + ['all'])}")
        sys.exit(1)

    choice = sys.argv[1]
    if choice == "all":
        for fn in STAGES.values():
            try:
                fn()
            except SystemExit:
                pass
    else:
        STAGES[choice]()

    plt.show()
