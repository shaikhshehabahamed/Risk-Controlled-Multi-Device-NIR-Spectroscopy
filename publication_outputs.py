"""
publication_outputs.py
======================
Central module for generating publication-quality figures and tables for the
NIR plastic-sorting paper.  Every function writes PDF + 300-dpi PNG figures
and CSV / LaTeX table artefacts.  Nothing is printed to stdout.

Output layout under any configured root directory
--------------------------------------------------
<output_dir>/
    figures/    – <stem>.pdf  (vector, editable text)  +  <stem>.png  (300 dpi)
    tables/     – <stem>.csv
    latex/      – <stem>.tex  (ready-to-include LaTeX tabular snippets)

Style
-----
IEEE / Elsevier single-column (3.5 in) or double-column (7.0 in) width.
Serif font, no grid, top/right spines removed, colorblind-safe palette.
"""

from __future__ import annotations
import logging

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── Publication RC ────────────────────────────────────────────────────────────

_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "patch.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
}

_W1 = 3.5   # single-column width (inches)
_W2 = 7.16  # double-column width (inches)

# Colorblind-safe, print-friendly palette
_C = {
    "blue":   "#2166AC",
    "red":    "#D6604D",
    "green":  "#4DAC26",
    "orange": "#F4A582",
    "grey":   "#878787",
    "purple": "#762A83",
}
_BAR_COLORS = [_C["blue"], _C["red"], _C["green"], _C["orange"], _C["purple"], _C["grey"]]


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _ctx():
    return matplotlib.rc_context(_RC)


def setup_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create figures/, tables/, latex/ subdirectories and return their paths."""
    root = Path(output_dir)
    dirs = {name: (root / name) for name in ("figures", "tables", "latex")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_fig(fig: plt.Figure, dirs: dict[str, Path], stem: str) -> dict[str, str]:
    pdf = dirs["figures"] / f"{stem}.pdf"
    png = dirs["figures"] / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _save_csv(df: pd.DataFrame, dirs: dict[str, Path], stem: str, *, index: bool = False) -> str:
    path = dirs["tables"] / f"{stem}.csv"
    df.to_csv(path, index=index)
    return str(path)


def _save_latex(df: pd.DataFrame, dirs: dict[str, Path], stem: str,
                *, float_format: str = "%.3f", index: bool = False) -> str:
    path = dirs["latex"] / f"{stem}.tex"
    tex = df.to_latex(index=index, float_format=float_format,
                      column_format="l" + "r" * (len(df.columns) - (0 if index else 1)))
    path.write_text(tex, encoding="utf-8")
    return str(path)


# ── Figure 1 – Preprocessing method comparison ───────────────────────────────

def figure_preprocessing_comparison(
    results_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Grouped bar chart comparing preprocessing methods on accuracy,
    balanced accuracy, and macro F1.

    Input:  results_df from group_split_baseline.run_group_split_baseline()
    Output: figures/preprocessing_comparison.{pdf,png}
            tables/preprocessing_comparison.csv
            latex/preprocessing_comparison.tex
    """
    dirs = setup_output_dirs(output_dir)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    labels  = ["Accuracy", "Bal. Accuracy", "Macro F1"]

    df = results_df[["method"] + [m for m in metrics if m in results_df.columns]].copy()
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    methods = df["method"].tolist()
    n_m = len(methods)
    n_g = len([m for m in metrics if m in df.columns])
    x = np.arange(n_m)
    w = 0.22

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1 if n_m <= 5 else _W2, 2.6))
        for gi, (metric, label) in enumerate(zip(metrics, labels)):
            if metric not in df.columns:
                continue
            vals = df[metric].to_numpy()
            offset = (gi - (n_g - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w, label=label, color=_BAR_COLORS[gi],
                          edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods], rotation=25, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.12)
        ax.legend(loc="upper right", ncol=3)
        ax.set_title("Preprocessing Method Comparison (Group-Safe Split)")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "preprocessing_comparison")
    artefacts["csv"]   = _save_csv(df, dirs, "preprocessing_comparison")
    artefacts["latex"] = _save_latex(df, dirs, "preprocessing_comparison")
    return artefacts


# ── Figure 2 – PCA component selection ───────────────────────────────────────

def figure_pca_selection(
    results_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Line chart of validation F1 / balanced accuracy vs n_components_used.

    Input:  results_df from pca_baseline.run_pca_experiment()
    Output: figures/pca_selection.{pdf,png}
            tables/pca_selection.csv
            latex/pca_selection.tex
    """
    dirs = setup_output_dirs(output_dir)

    val_f1_col  = "validation_macro_f1"   if "validation_macro_f1"   in results_df.columns else None
    val_ba_col  = "validation_balanced_accuracy" if "validation_balanced_accuracy" in results_df.columns else None
    comp_col    = "n_components_used"     if "n_components_used"     in results_df.columns else "n_components_requested"

    df = results_df.copy()
    df["_x"] = pd.to_numeric(df[comp_col], errors="coerce")
    df = df.dropna(subset=["_x"]).sort_values("_x").reset_index(drop=True)

    best_idx = int(df[val_f1_col].idxmax()) if val_f1_col else 0

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1, 2.5))

        if val_f1_col:
            ax.plot(df["_x"], df[val_f1_col], "o-", color=_C["blue"],
                    label="Val. Macro F1")
            ax.axvline(df.loc[best_idx, "_x"], color=_C["blue"],
                       linestyle="--", linewidth=0.9, alpha=0.6)
        if val_ba_col:
            ax.plot(df["_x"], df[val_ba_col], "s--", color=_C["red"],
                    label="Val. Balanced Acc.")

        ax.set_xlabel("PCA Components")
        ax.set_ylabel("Validation Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.set_title("PCA Component Selection")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "pca_selection")
    artefacts["csv"]   = _save_csv(df.drop(columns=["_x"]), dirs, "pca_selection")
    artefacts["latex"] = _save_latex(
        df[[comp_col] + ([val_f1_col] if val_f1_col else []) + ([val_ba_col] if val_ba_col else [])],
        dirs, "pca_selection"
    )
    return artefacts


# ── Figure 3 – PCA scree plot ─────────────────────────────────────────────────

def figure_pca_scree(
    pca_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    champion_n: Optional[int] = None,
) -> dict:
    """
    Scree plot: per-component explained variance and cumulative curve.

    Input:  pca_df from freeze_champion_model (columns: component,
            explained_variance_ratio, cumulative_explained_variance)
    Output: figures/pca_scree.{pdf,png}
            tables/pca_scree.csv
            latex/pca_scree.tex
    """
    dirs = setup_output_dirs(output_dir)

    with _ctx():
        fig, ax1 = plt.subplots(figsize=(_W1, 2.6))
        ax2 = ax1.twinx()
        ax2.spines["top"].set_visible(False)

        x = pca_df["component"].to_numpy()
        ax1.bar(x, pca_df["explained_variance_ratio"], color=_C["blue"],
                alpha=0.75, label="Per-component EVR")
        ax2.plot(x, pca_df["cumulative_explained_variance"] * 100,
                 color=_C["red"], marker=".", markersize=3,
                 linewidth=1.2, label="Cumulative EVR (%)")

        if champion_n is not None:
            ax1.axvline(champion_n, color=_C["grey"], linestyle="--",
                        linewidth=1.0, label=f"Champion (n={champion_n})")

        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax2.set_ylabel("Cumulative EVR (%)")
        ax2.set_ylim(0, 105)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right",
                   fontsize=7)
        ax1.set_title("PCA Explained Variance")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "pca_scree")
    artefacts["csv"]   = _save_csv(pca_df, dirs, "pca_scree")
    artefacts["latex"] = _save_latex(pca_df, dirs, "pca_scree")
    return artefacts


# ── Figure 4 – Normalised confusion matrix ────────────────────────────────────

def figure_confusion_matrix(
    conf_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    title: str = "Confusion Matrix (row-normalised, %)",
) -> dict:
    """
    Row-normalised confusion matrix heatmap with per-cell percentage annotation.

    Input:  conf_df – square DataFrame (index = true labels, columns = predicted)
    Output: figures/confusion_matrix.{pdf,png}
    """
    dirs = setup_output_dirs(output_dir)
    labels = list(conf_df.index)
    n = len(labels)

    raw = conf_df.to_numpy(dtype=float)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm = raw / row_sums * 100.0

    # Dynamic figure size: scale with class count
    side = max(_W1, min(_W2, 0.35 * n + 0.8))
    fig_w = side
    fig_h = side * 0.82 + 0.5

    with _ctx():
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Recall (%)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # Annotate cells
        thresh = 50.0
        fs = max(5, 9 - n // 5)
        for i in range(n):
            for j in range(n):
                v = norm[i, j]
                if v > 0.5:
                    color = "white" if v > thresh else "black"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=fs, color=color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(6, 8 - n // 8))
        ax.set_yticklabels(labels, fontsize=max(6, 8 - n // 8))
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "confusion_matrix")

    return artefacts


# ── Figure 5 – Per-class classification report ───────────────────────────────

def figure_classification_report(
    report_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Horizontal grouped bar chart of per-class precision, recall, and F1-score.

    Input:  report_df from sklearn classification_report (output_dict=True)
            columns include: precision, recall, f1-score; index = class labels
    Output: figures/classification_report.{pdf,png}
            tables/classification_report.csv
            latex/classification_report.tex
    """
    dirs = setup_output_dirs(output_dir)

    # Keep only per-class rows (drop averages and support)
    summary_rows = {"accuracy", "macro avg", "weighted avg"}
    df = report_df[~report_df.index.isin(summary_rows)].copy()
    df = df[["precision", "recall", "f1-score"]].dropna()
    df = df.sort_values("f1-score", ascending=True)

    n = len(df)
    fig_h = max(2.5, 0.28 * n + 0.6)
    fig_w = _W2 if n > 12 else _W1

    y = np.arange(n)
    w = 0.24

    with _ctx():
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        metrics_plot = [("f1-score", "Macro F1", _C["blue"]),
                        ("precision", "Precision", _C["red"]),
                        ("recall", "Recall", _C["green"])]
        for gi, (col, lbl, clr) in enumerate(metrics_plot):
            offset = (gi - 1) * w
            ax.barh(y + offset, df[col].to_numpy(), w, label=lbl,
                    color=clr, alpha=0.85, edgecolor="white", linewidth=0.3)

        ax.set_yticks(y)
        ax.set_yticklabels(df.index.tolist(), fontsize=max(6, 9 - n // 8))
        ax.set_xlabel("Score")
        ax.set_xlim(0, 1.12)
        ax.axvline(1.0, color=_C["grey"], linewidth=0.7, linestyle="--")
        ax.legend(loc="lower right", ncol=3)
        ax.set_title("Per-class Classification Performance")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "classification_report")
    artefacts["csv"]   = _save_csv(df.sort_values("f1-score", ascending=False), dirs, "classification_report", index=True)
    artefacts["latex"] = _save_latex(df.sort_values("f1-score", ascending=False), dirs, "classification_report", index=True)
    return artefacts


# ── Figure 6 – Device-to-device transfer heatmap ─────────────────────────────

def figure_device_transfer_heatmap(
    metrics_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    metric: str = "macro_f1",
) -> dict:
    """
    Square heatmap of macro F1 for every (train_device, test_device) pair,
    with a separate panel for Leave-One-Device-Out (LODO) results.

    Input:  metrics_df from device_domain_experiments.run_device_domain_experiments()
    Output: figures/device_transfer_heatmap.{pdf,png}
            tables/device_transfer_matrix.csv
    """
    dirs = setup_output_dirs(output_dir)

    single = metrics_df[metrics_df["experiment"] == "single_source_device_transfer"].copy()
    lodo   = metrics_df[metrics_df["experiment"] == "leave_one_device_out"].copy()

    devices = sorted(set(single["train_devices"].tolist() + single["test_device"].tolist()))
    n_d = len(devices)

    # Build pivot matrix
    mat = pd.DataFrame(np.nan, index=devices, columns=devices)
    for _, row in single.iterrows():
        mat.loc[row["train_devices"], row["test_device"]] = row[metric]

    with _ctx():
        n_panels = 1 + (1 if not lodo.empty else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(_W2 if n_panels == 2 else _W1, 2.8))
        if n_panels == 1:
            axes = [axes]

        # Panel 1: single-source transfer matrix
        data = mat.to_numpy(dtype=float)
        im1 = axes[0].imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        for i, td in enumerate(devices):
            for j, test_d in enumerate(devices):
                v = mat.loc[td, test_d]
                if not np.isnan(v):
                    axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                                 fontsize=7, color="black" if v > 0.3 else "white")
        axes[0].set_xticks(range(n_d))
        axes[0].set_yticks(range(n_d))
        axes[0].set_xticklabels(devices, rotation=30, ha="right", fontsize=8)
        axes[0].set_yticklabels(devices, fontsize=8)
        axes[0].set_xlabel("Test Device")
        axes[0].set_ylabel("Train Device")
        axes[0].set_title("Single-Source Transfer")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label=metric.replace("_", " ").title())

        # Panel 2: LODO bar chart
        if not lodo.empty:
            td = lodo["test_device"].tolist()
            vals = lodo[metric].tolist()
            x = np.arange(len(td))
            axes[1].bar(x, vals, color=_C["blue"], edgecolor="white", linewidth=0.4)
            for xi, v in zip(x, vals):
                axes[1].text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(td, rotation=25, ha="right", fontsize=8)
            axes[1].set_ylabel(metric.replace("_", " ").title())
            axes[1].set_ylim(0, 1.12)
            axes[1].set_title("Leave-One-Device-Out")

        fig.suptitle("Cross-Device Transfer Performance", y=1.01)
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "device_transfer_heatmap")
    artefacts["csv"] = _save_csv(mat.reset_index().rename(columns={"index": "train_device"}),
                                  dirs, "device_transfer_matrix")
    artefacts["latex"] = _save_latex(mat, dirs, "device_transfer_matrix", index=True)
    return artefacts


# ── Figure 7 – Robustness analysis bar chart ─────────────────────────────────

def figure_robustness_analysis(
    robustness_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Clustered bar chart showing purity and coverage under spectral perturbations.
    The 'clean' baseline is highlighted with a reference line.

    Input:  robustness_df from robustness_analysis.evaluate_robustness_suite()
    Output: figures/robustness_analysis.{pdf,png}
            tables/robustness_analysis.csv
            latex/robustness_analysis.tex
    """
    dirs = setup_output_dirs(output_dir)

    metrics = [m for m in ["purity", "coverage", "accepted_macro_f1"] if m in robustness_df.columns]
    labels  = [m.replace("_", " ").title() for m in metrics]

    df = robustness_df.copy()
    df = df.sort_values("perturbation", key=lambda s: s.map({"clean": "0_clean"}).fillna(s))
    perts = df["perturbation"].tolist()
    n_p = len(perts)
    n_m = len(metrics)
    x = np.arange(n_p)
    w = 0.22

    # Baseline values from 'clean' row
    clean_row = df[df["perturbation"] == "clean"]

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W2, 2.8))
        for gi, (metric, label) in enumerate(zip(metrics, labels)):
            offset = (gi - (n_m - 1) / 2) * w
            vals = df[metric].to_numpy()
            bars = ax.bar(x + offset, vals, w, label=label, color=_BAR_COLORS[gi],
                          edgecolor="white", linewidth=0.3, alpha=0.88)
            ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=2)
            # Draw clean baseline line
            if not clean_row.empty:
                base = float(clean_row[metric].iloc[0])
                ax.axhline(base, color=_BAR_COLORS[gi], linestyle=":",
                           linewidth=0.9, alpha=0.6)

        # Shade the 'clean' column
        clean_x = next((i for i, p in enumerate(perts) if p == "clean"), None)
        if clean_x is not None:
            ax.axvspan(clean_x - 0.45, clean_x + 0.45, alpha=0.07,
                       color=_C["grey"], zorder=0)
            ax.text(clean_x, 1.02, "clean", ha="center", va="bottom",
                    fontsize=7, color=_C["grey"])

        ax.set_xticks(x)
        xticklabels = [p.replace("_", "\n") for p in perts]
        ax.set_xticklabels(xticklabels, fontsize=7, ha="center")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.12)
        ax.legend(loc="upper right", ncol=3)
        ax.set_title("Robustness Under Spectral Perturbations")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "robustness_analysis")
    artefacts["csv"]   = _save_csv(robustness_df, dirs, "robustness_analysis")
    artefacts["latex"] = _save_latex(
        robustness_df[["perturbation"] + metrics].round(4),
        dirs, "robustness_analysis"
    )
    return artefacts


# ── Figure 8 – RVEA Pareto front ─────────────────────────────────────────────

def figure_pareto_front(
    pareto_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    selected_row: Optional[pd.Series] = None,
) -> dict:
    """
    Scatter plot of the Pareto front in coverage–purity space,
    with the selected operating point highlighted.

    Input:  pareto_df from run_breakthrough_pipeline — must contain the columns
            produced by optimize_abstention_policy:
            objective_purity_loss, objective_coverage_loss,
            objective_worst_domain_loss
    Output: figures/pareto_front.{pdf,png}
            tables/pareto_front.csv
            latex/pareto_front.tex
    """
    dirs = setup_output_dirs(output_dir)

    # Fix #21: use column names produced by optimize_abstention_policy.
    purity_col   = "objective_purity_loss"
    coverage_col = "objective_coverage_loss"
    wdp_col      = "objective_worst_domain_loss"

    df = pareto_df.copy()
    missing = [c for c in [purity_col, coverage_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"figure_pareto_front: required columns missing: {missing}. "
            "Expected 'objective_purity_loss', 'objective_coverage_loss', "
            "'objective_worst_domain_loss' as produced by optimize_abstention_policy."
        )

    df["_purity"]   = 1.0 - df[purity_col].clip(0, 1)
    df["_coverage"] = 1.0 - df[coverage_col].clip(0, 1)
    wdp_vals = (1.0 - df[wdp_col].clip(0, 1)).to_numpy() if wdp_col in df.columns else None

    out_df = pd.DataFrame({"purity": df["_purity"], "coverage": df["_coverage"]})
    if wdp_vals is not None:
        out_df["worst_domain_performance"] = wdp_vals
    decision_cols = [
        c for c in pareto_df.columns
        if c.startswith("threshold::") or c.startswith("threshold_pair::")
        or c.startswith("max_entropy_pair::")
    ]
    for c in decision_cols:
        out_df[c] = pareto_df[c].values
    if "min_margin" in pareto_df.columns:
        out_df["min_margin"] = pareto_df["min_margin"].values
    if "max_entropy" in pareto_df.columns:
        out_df["max_entropy"] = pareto_df["max_entropy"].values

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1, 2.6))
        scatter_kw = dict(s=18, linewidths=0.4, edgecolors="white", alpha=0.85, zorder=3)
        if wdp_vals is not None:
            sc = ax.scatter(df["_coverage"], df["_purity"], c=wdp_vals,
                            cmap="RdYlGn", vmin=0, vmax=1, **scatter_kw)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Worst-Domain Perf.", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        else:
            ax.scatter(df["_coverage"], df["_purity"], color=_C["blue"], **scatter_kw)
        if selected_row is not None:
            sel_purity   = 1.0 - float(selected_row[purity_col])
            sel_coverage = 1.0 - float(selected_row[coverage_col])
            ax.scatter([sel_coverage], [sel_purity], color=_C["red"],
                       s=60, marker="*", zorder=5, label="Selected policy")
            ax.legend(loc="lower left", fontsize=7)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Purity")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Policy Search Pareto Front\n(Coverage vs. Purity)")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "pareto_front")  # Fix #19: inside ctx

    artefacts["csv"]   = _save_csv(out_df, dirs, "pareto_front")
    artefacts["latex"] = _save_latex(
        out_df[["purity", "coverage"] + (["worst_domain_performance"] if wdp_vals is not None else [])].round(4),
        dirs, "pareto_front"
    )
    return artefacts


# ── Figure 9 – Policy metrics summary (calibration vs test) ──────────────────

def figure_policy_metrics(
    policy_metrics_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Grouped bar chart of purity, coverage, accepted accuracy, and
    realised circularity proxy across calibration and test splits.

    Input:  policy_metrics_df – 2-row DataFrame (split = calibration / test)
    Output: figures/policy_metrics.{pdf,png}
            tables/policy_metrics.csv
            latex/policy_metrics.tex
    """
    dirs = setup_output_dirs(output_dir)

    metrics = [m for m in ["purity", "coverage", "accepted_macro_f1",
                            "realized_circularity_proxy"] if m in policy_metrics_df.columns]
    labels  = [m.replace("_", " ").replace("macro f1", "Macro F1").title() for m in metrics]
    splits  = policy_metrics_df["split"].tolist()

    n_s = len(splits)
    n_m = len(metrics)
    x = np.arange(n_m)
    w = 0.30 / max(1, n_s)

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1, 2.5))
        split_colors = [_C["blue"], _C["red"], _C["green"]]
        for si, split in enumerate(splits):
            row = policy_metrics_df[policy_metrics_df["split"] == split]
            if row.empty:
                continue
            vals = [float(row[m].iloc[0]) for m in metrics]
            offset = (si - (n_s - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w, label=split.replace("_", " ").title()[:30],
                          color=split_colors[si % len(split_colors)],
                          edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.20)
        ax.legend(loc="upper right")
        ax.set_title("Abstention Policy Performance")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "policy_metrics")
    artefacts["csv"]   = _save_csv(policy_metrics_df, dirs, "policy_metrics")
    artefacts["latex"] = _save_latex(policy_metrics_df[["split"] + metrics].round(4),
                                      dirs, "policy_metrics")
    return artefacts


# ── Figure 10 – Circularity proxy bar chart ──────────────────────────────────

def figure_circularity_proxy(
    proxy_table: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Horizontal bar chart of recoverability_proxy per composition class,
    coloured by dominant component.

    Input:  proxy_table from circularity_proxy.build_proxy_table()
    Output: figures/circularity_proxy.{pdf,png}
            tables/circularity_proxy.csv
            latex/circularity_proxy.tex
    """
    dirs = setup_output_dirs(output_dir)

    df = proxy_table.copy()
    if "recoverability_proxy" not in df.columns:
        return {}
    df = df.sort_values("recoverability_proxy", ascending=True).reset_index(drop=True)

    n = len(df)
    fig_h = max(2.5, 0.26 * n + 0.6)

    # Assign colors by dominant component
    components = df["dominant_component"].unique().tolist() if "dominant_component" in df.columns else []
    comp_color = {c: _BAR_COLORS[i % len(_BAR_COLORS)] for i, c in enumerate(sorted(components))}

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1, fig_h))
        y = np.arange(n)
        bar_colors = [comp_color.get(c, _C["grey"]) for c in df.get("dominant_component", [""] * n)]
        bars = ax.barh(y, df["recoverability_proxy"].to_numpy(), color=bar_colors,
                       edgecolor="white", linewidth=0.3, alpha=0.88)
        for bar, v in zip(bars, df["recoverability_proxy"]):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=6.5)

        ax.set_yticks(y)
        comp_col = "canonical_code" if "canonical_code" in df.columns else "composition_code"
        ax.set_yticklabels(df[comp_col].tolist(), fontsize=max(6, 8 - n // 10))
        ax.set_xlabel("Recoverability Proxy (Herfindahl Index)")
        ax.set_xlim(0, 1.15)
        ax.axvline(1.0, color=_C["grey"], linestyle="--", linewidth=0.7)

        # Legend for dominant components
        if components:
            legend_patches = [mpatches.Patch(color=comp_color[c], label=c.upper())
                              for c in sorted(components)]
            ax.legend(handles=legend_patches, loc="lower right", fontsize=7,
                      title="Dom. component", title_fontsize=7)
        ax.set_title("Recoverability Proxy by Composition Class")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "circularity_proxy")
    artefacts["csv"]   = _save_csv(df, dirs, "circularity_proxy")
    artefacts["latex"] = _save_latex(
        df[[comp_col, "recoverability_proxy", "n_nonzero_components"]].round(4),
        dirs, "circularity_proxy"
    )
    return artefacts


# ── Figure 11 – RVEA optimisation convergence ────────────────────────────────

def figure_rvea_convergence(
    history_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Line chart of Pareto front size and hypervolume indicator across generations.

    Input:  history_df – rvea_history.csv (columns: generation,
            pareto_front_size, and optionally hypervolume)
    Output: figures/rvea_convergence.{pdf,png}
            tables/rvea_convergence.csv
    """
    dirs = setup_output_dirs(output_dir)

    if "generation" not in history_df.columns:
        return {}

    # Fix #22: history_df has columns generation/candidate_idx/selection_score.
    # Summarise best and mean selection_score per generation for a useful plot.
    if "selection_score" not in history_df.columns:
        return {}

    gen_stats = (
        history_df.groupby("generation")["selection_score"]
        .agg(best_score="max", mean_score="mean")
        .reset_index()
    )

    with _ctx():
        fig, ax1 = plt.subplots(figsize=(_W1, 2.4))
        ax1.plot(gen_stats["generation"], gen_stats["best_score"],
                 color=_C["blue"], marker=".", markersize=3, label="Best")
        ax1.plot(gen_stats["generation"], gen_stats["mean_score"],
                 color=_C["grey"], linestyle="--", marker=".", markersize=3,
                 label="Mean")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Selection Score")
        ax1.legend(loc="lower right", fontsize=7)
        ax1.set_title("Policy Search Convergence per Generation")
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "rvea_convergence")

    artefacts["csv"] = _save_csv(gen_stats, dirs, "rvea_convergence")
    return artefacts


# ── Figure 12 – Sample mean spectra per instrument ───────────────────────────

def figure_mean_spectra(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Mean NIR spectrum for each instrument type, overlaid on a single plot.

    Input:  df – loaded NIR DataFrame from nir_loader.load_nir_spectra()
            must contain columns: instrument, wavelength_axis, intensity_axis
    Output: figures/mean_spectra.{pdf,png}
    """
    dirs = setup_output_dirs(output_dir)

    instruments = sorted(df["instrument"].dropna().unique().tolist())
    inst_colors = [_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(len(instruments))]

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W2, 2.6))
        for inst, clr in zip(instruments, inst_colors):
            sub = df[df["instrument"] == inst]
            # Stack all intensity vectors for this instrument
            arrays = np.vstack(sub["intensity_axis"].apply(
                lambda v: np.asarray(v, dtype=float)).tolist())
            wl = np.asarray(sub["wavelength_axis"].iloc[0], dtype=float)
            mean = arrays.mean(axis=0)
            std  = arrays.std(axis=0)
            ax.plot(wl, mean, color=clr, linewidth=1.2, label=inst)
            ax.fill_between(wl, mean - std, mean + std, color=clr, alpha=0.12)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.legend(loc="upper right")
        ax.set_title("Mean NIR Spectra per Instrument (±1 SD)")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "mean_spectra")
    return artefacts



# ── Figure X – Open-set metrics ─────────────────────────────────────────────

def figure_open_set_metrics(
    metrics_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    dirs = setup_output_dirs(output_dir)
    if metrics_df.empty:
        raise ValueError("metrics_df must not be empty.")
    row = metrics_df.iloc[0]
    metric_map = {
        "Unknown AUROC": float(row.get("unknown_auroc", np.nan)),
        "Unknown AUPRC": float(row.get("unknown_auprc", np.nan)),
        "Known Accept": float(row.get("known_acceptance_rate", np.nan)),
        "Unknown Reject": float(row.get("unknown_rejection_rate", np.nan)),
        "Open-set Macro F1": float(row.get("combined_open_set_macro_f1", np.nan)),
    }
    df = pd.DataFrame({"metric": list(metric_map.keys()), "value": list(metric_map.values())})
    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1, 2.6))
        bars = ax.bar(np.arange(len(df)), df["value"].to_numpy(dtype=float), color=_BAR_COLORS[: len(df)])
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df["metric"], rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Open-set Evaluation Metrics")
        for bar, val in zip(bars, df["value"].to_numpy(dtype=float)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "open_set_metrics")
    artefacts["csv"] = _save_csv(df, dirs, "open_set_metrics")
    artefacts["latex"] = _save_latex(df, dirs, "open_set_metrics")
    return artefacts


# ── Figure X+1 – Open-set unknown-score distributions ──────────────────────

def figure_open_set_score_distribution(
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    dirs = setup_output_dirs(output_dir)
    required = {"unknown_score", "is_unknown_true"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df is missing required columns: {sorted(missing)}")
    df = predictions_df.copy()
    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1, 2.5))
        known_scores = df.loc[df["is_unknown_true"].astype(int) == 0, "unknown_score"].to_numpy(dtype=float)
        unknown_scores = df.loc[df["is_unknown_true"].astype(int) == 1, "unknown_score"].to_numpy(dtype=float)
        bins = np.linspace(float(np.nanmin(df["unknown_score"])), float(np.nanmax(df["unknown_score"])), 30)
        if len(known_scores) > 0:
            ax.hist(known_scores, bins=bins, alpha=0.65, color=_C["blue"], label="Known", density=True)
        if len(unknown_scores) > 0:
            ax.hist(unknown_scores, bins=bins, alpha=0.65, color=_C["red"], label="Unknown", density=True)
        ax.set_xlabel("Unknown score")
        ax.set_ylabel("Density")
        ax.set_title("Known vs Unknown Score Distributions")
        ax.legend(loc="upper right")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "open_set_score_distribution")
    artefacts["csv"] = _save_csv(df[[c for c in ["unknown_score", "is_unknown_true", "predicted_open_label", "true_open_label"] if c in df.columns]], dirs, "open_set_score_distribution")
    return artefacts


# ── Figure X+2 – Flat vs composition-aware comparison ──────────────────────

def figure_composition_model_comparison(
    comparison_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    dirs = setup_output_dirs(output_dir)
    if comparison_df.empty:
        raise ValueError("comparison_df must not be empty.")
    score_cols = [c for c in ["label_accuracy", "label_balanced_accuracy", "label_macro_f1"] if c in comparison_df.columns]
    df = comparison_df[["model"] + score_cols].copy()
    melted = df.melt(id_vars="model", var_name="metric", value_name="value")
    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1, 2.6))
        metrics = list(dict.fromkeys(melted["metric"].tolist()))
        models = list(dict.fromkeys(melted["model"].tolist()))
        x = np.arange(len(metrics))
        w = 0.35 if len(models) <= 2 else 0.25
        for mi, model in enumerate(models):
            sub = melted[melted["model"] == model].set_index("metric").reindex(metrics).reset_index()
            vals = sub["value"].to_numpy(dtype=float)
            offset = (mi - (len(models) - 1) / 2.0) * w
            ax.bar(x + offset, vals, width=w, label=model, color=_BAR_COLORS[mi % len(_BAR_COLORS)])
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("label_", "").replace("_", " ").title() for m in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Flat vs Composition-aware Model Comparison")
        ax.legend(loc="lower right")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "composition_model_comparison")
    artefacts["csv"] = _save_csv(comparison_df, dirs, "composition_model_comparison")
    artefacts["latex"] = _save_latex(comparison_df, dirs, "composition_model_comparison")
    return artefacts


# ── Figure X+3 – Composition-aware structured metrics ──────────────────────

def figure_composition_structured_metrics(
    metrics_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    dirs = setup_output_dirs(output_dir)
    if metrics_df.empty:
        raise ValueError("metrics_df must not be empty.")
    row = metrics_df.iloc[0]
    metric_map = {
        "Dominant Acc": float(row.get("dominant_accuracy", np.nan)),
        "Mono-material Acc": float(row.get("mono_material_accuracy", np.nan)),
        "Fraction MAE": float(row.get("fraction_mae", np.nan)),
        "Recoverability MAE": float(row.get("recoverability_mae", np.nan)),
        "Difficulty MAE": float(row.get("sorting_difficulty_mae", np.nan)),
    }
    df = pd.DataFrame({"metric": list(metric_map.keys()), "value": list(metric_map.values())})
    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1, 2.7))
        bars = ax.bar(np.arange(len(df)), df["value"].to_numpy(dtype=float), color=_BAR_COLORS[: len(df)])
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df["metric"], rotation=25, ha="right")
        ymax = max(1.05, float(np.nanmax(df["value"])) * 1.15) if np.isfinite(df["value"]).any() else 1.05
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Value")
        ax.set_title("Composition-aware Structured-target Performance")
        for bar, val in zip(bars, df["value"].to_numpy(dtype=float)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * ymax, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "composition_structured_metrics")
    artefacts["csv"] = _save_csv(df, dirs, "composition_structured_metrics")
    artefacts["latex"] = _save_latex(df, dirs, "composition_structured_metrics")
    return artefacts

# ── Figure 13 – Focus / dangerous pair error rates ───────────────────────────

def figure_focus_pairs_error_rates(
    focus_pairs_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    title: str = "High-Confusion Class Pairs (Policy Focus Pairs)",
) -> dict:
    """
    Horizontal bar chart of per-pair error rates flagged by the RVEA
    abstention-policy optimiser.

    Input:  focus_pairs_df from rvea_optimizer.save_policy_optimization_outputs()
            (focus_pair_summary.csv or dangerous_pairs.csv)
            required columns: top2_pair, error_rate, n_errors
            optional columns: domain, mean_entropy
    Output: figures/focus_pairs_error_rates.{pdf,png}
            tables/focus_pairs_error_rates.csv
            latex/focus_pairs_error_rates.tex
    """
    dirs = setup_output_dirs(output_dir)

    df = focus_pairs_df.copy()
    if df.empty or "error_rate" not in df.columns:
        return {}

    pair_col = "top2_pair" if "top2_pair" in df.columns else df.columns[0]
    domain_col = "domain" if "domain" in df.columns else None

    df = df.sort_values("error_rate", ascending=True).reset_index(drop=True)
    n = len(df)
    fig_h = max(2.2, 0.35 * n + 0.6)

    # Build y-axis labels that include domain when available
    if domain_col:
        y_labels = [f"{row[pair_col]}  [{row[domain_col]}]" for _, row in df.iterrows()]
    else:
        y_labels = df[pair_col].astype(str).tolist()

    y = np.arange(n)
    colors = [_C["red"] if v >= 0.5 else _C["orange"] if v >= 0.25 else _C["blue"]
              for v in df["error_rate"].to_numpy()]

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W2, fig_h))
        bars = ax.barh(y, df["error_rate"].to_numpy(), color=colors,
                       edgecolor="white", linewidth=0.3, alpha=0.88)
        for bar, (_, row) in zip(bars, df.iterrows()):
            n_err = int(row.get("n_errors", 0))
            er = float(row["error_rate"])
            ax.text(er + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{er:.3f}  (n={n_err})", va="center", fontsize=7)

        ax.set_yticks(y)
        ax.set_yticklabels(y_labels, fontsize=max(6, 9 - n // 6))
        ax.set_xlabel("Error Rate")
        ax.set_xlim(0, 1.18)
        ax.axvline(0.5, color=_C["grey"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(0.5, n - 0.1, "50%", ha="center", va="top", fontsize=7,
                color=_C["grey"])

        # Legend patches for colour coding
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(color=_C["red"],    label="≥ 50 % error"),
            Patch(color=_C["orange"], label="25–50 % error"),
            Patch(color=_C["blue"],   label="< 25 % error"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7)
        ax.set_title(title)
        fig.tight_layout()
        stem = "focus_pairs_error_rates"
        artefacts = _save_fig(fig, dirs, stem)
    keep_cols = [c for c in [pair_col, domain_col, "n_errors", "error_rate", "mean_entropy"]
                 if c and c in df.columns]
    artefacts["csv"]   = _save_csv(df[keep_cols], dirs, stem)
    artefacts["latex"] = _save_latex(df[keep_cols].round(4), dirs, stem)
    return artefacts


# ── Figure 14 – Per-device abstention thresholds ─────────────────────────────

def figure_policy_thresholds(
    policy: dict,
    output_dir: str | Path,
) -> dict:
    """
    Bar chart of per-device confidence thresholds selected by risk-controlled
    policy optimisation, with reference lines for min_margin and max_entropy.

    Input:  policy dict (policy.json from run_breakthrough_pipeline)
    Output: figures/policy_thresholds.{pdf,png}
            tables/policy_thresholds.csv
            latex/policy_thresholds.tex
    """
    dirs = setup_output_dirs(output_dir)

    thresholds = policy.get("thresholds_by_domain", {})
    if not thresholds:
        return {}

    devices = sorted(thresholds.keys())
    values  = [float(thresholds[d]) for d in devices]
    min_margin  = float(policy.get("min_margin") or 0.0)
    # Fix #35: policy.get() returns None when key exists but value is None (null in JSON).
    # float(None) raises TypeError, so guard explicitly.
    _raw_entropy = policy.get("max_entropy")
    max_entropy = float(_raw_entropy) if _raw_entropy is not None else 1.0

    x = np.arange(len(devices))

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W1 if len(devices) <= 4 else _W2, 2.5))
        bars = ax.bar(x, values, color=_C["blue"], edgecolor="white",
                      linewidth=0.4, alpha=0.88, label="Confidence threshold")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(min_margin,  color=_C["red"],    linestyle="--", linewidth=1.0,
                   label=f"Min. margin = {min_margin:.3f}")
        ax.axhline(max_entropy, color=_C["orange"], linestyle=":",  linewidth=1.0,
                   label=f"Max. entropy = {max_entropy:.3f}")

        ax.set_xticks(x)
        ax.set_xticklabels(devices, rotation=20, ha="right")
        ax.set_ylabel("Threshold / Margin Value")
        ax.set_ylim(0, 1.12)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Risk-Controlled Abstention Policy Thresholds")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "policy_thresholds")  # Fix #19: inside ctx

    rows = [{"device": d, "threshold": v} for d, v in zip(devices, values)]
    rows.append({"device": "min_margin",  "threshold": min_margin})
    rows.append({"device": "max_entropy", "threshold": max_entropy})
    tbl = pd.DataFrame(rows)
    artefacts["csv"]   = _save_csv(tbl, dirs, "policy_thresholds")
    artefacts["latex"] = _save_latex(tbl.round(4), dirs, "policy_thresholds")
    return artefacts


# ── Figure 15 – Composition-aware per-device performance ─────────────────────

def figure_composition_domain_performance(
    domain_metrics_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Grouped bar chart of per-device accuracy, balanced accuracy, and macro F1
    for the composition-aware structured model.

    Input:  domain_metrics_df from composition_aware_model
            (composition_aware_domain_metrics.csv)
            required columns: domain, accuracy / balanced_accuracy / macro_f1
    Output: figures/composition_domain_performance.{pdf,png}
            tables/composition_domain_performance.csv
            latex/composition_domain_performance.tex
    """
    dirs = setup_output_dirs(output_dir)

    df = domain_metrics_df.copy()
    if df.empty:
        return {}

    domain_col = next((c for c in ["domain", "instrument", "test_device"] if c in df.columns), None)
    if domain_col is None:
        return {}

    # Fix #41: _safe_domain_metrics writes "label_accuracy" etc; also accept that form.
    _col_map = {
        "label_accuracy": "accuracy",
        "label_balanced_accuracy": "balanced_accuracy",
        "label_macro_f1": "macro_f1",
    }
    df = df.rename(columns={k: v for k, v in _col_map.items() if k in df.columns})
    metrics = [m for m in ["accuracy", "balanced_accuracy", "macro_f1"] if m in df.columns]
    if not metrics:
        return {}
    labels = [m.replace("_", " ").title() for m in metrics]

    devices = df[domain_col].tolist()
    n_d = len(devices)
    n_m = len(metrics)
    x   = np.arange(n_d)
    w   = 0.22

    with _ctx():
        fig, ax = plt.subplots(figsize=(_W2 if n_d > 3 else _W1, 2.6))
        for gi, (metric, label) in enumerate(zip(metrics, labels)):
            offset = (gi - (n_m - 1) / 2) * w
            vals = df[metric].to_numpy(dtype=float)
            bars = ax.bar(x + offset, vals, w, label=label,
                          color=_BAR_COLORS[gi], edgecolor="white",
                          linewidth=0.4, alpha=0.88)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(devices, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.18)
        ax.legend(loc="upper right", ncol=3)
        ax.set_title("Composition-Aware Model: Per-Device Performance")
        fig.tight_layout()
        artefacts = _save_fig(fig, dirs, "composition_domain_performance")
    artefacts["csv"]   = _save_csv(df[[domain_col] + metrics], dirs,
                                    "composition_domain_performance")
    artefacts["latex"] = _save_latex(df[[domain_col] + metrics].round(4),
                                      dirs, "composition_domain_performance")
    return artefacts


# ── Convenience: run all figures from saved output directories ────────────────

def generate_all_figures(
    *,
    group_split_dir: Optional[str | Path] = None,
    pca_dir: Optional[str | Path] = None,
    freeze_dir: Optional[str | Path] = None,
    device_domain_dir: Optional[str | Path] = None,
    breakthrough_dir: Optional[str | Path] = None,
    circularity_dir: Optional[str | Path] = None,
    pub_output_dir: str | Path = "pub_outputs",
) -> dict[str, dict]:
    """
    Convenience entry point: load saved CSV artefacts from each pipeline stage
    and call every figure function.  Pass only the directories that exist.

    Returns a nested dict of file paths keyed by figure name.
    """
    out_dir = Path(pub_output_dir)
    all_artefacts: dict[str, dict] = {}

    # ── group_split_baseline ──────────────────────────────────────────────────
    if group_split_dir is not None:
        d = Path(group_split_dir)
        results_path = d / "baseline_results.csv"
        if results_path.exists():
            all_artefacts["preprocessing_comparison"] = figure_preprocessing_comparison(
                pd.read_csv(results_path), out_dir
            )

    # ── pca_baseline ─────────────────────────────────────────────────────────
    if pca_dir is not None:
        d = Path(pca_dir)
        pca_path = d / "pca_results.csv"
        if pca_path.exists():
            all_artefacts["pca_selection"] = figure_pca_selection(
                pd.read_csv(pca_path), out_dir
            )

    # ── freeze_champion_model ─────────────────────────────────────────────────
    if freeze_dir is not None:
        d = Path(freeze_dir)

        conf_path = d / "confusion_matrix.csv"
        if conf_path.exists():
            conf_df = pd.read_csv(conf_path, index_col=0)
            all_artefacts["confusion_matrix"] = figure_confusion_matrix(conf_df, out_dir)

        report_path = d / "classification_report.csv"
        if report_path.exists():
            all_artefacts["classification_report"] = figure_classification_report(
                pd.read_csv(report_path, index_col=0), out_dir
            )

        pca_scree_path = d / "pca_explained_variance.csv"
        model_config_path = d / "model_config.json"
        if pca_scree_path.exists():
            champion_n = None
            if model_config_path.exists():
                cfg = json.loads(model_config_path.read_text(encoding="utf-8"))
                champion_n = int(cfg.get("pca", {}).get("used_n_components", 0)) or None
            all_artefacts["pca_scree"] = figure_pca_scree(
                pd.read_csv(pca_scree_path), out_dir, champion_n=champion_n
            )

    # ── device_domain_experiments ─────────────────────────────────────────────
    if device_domain_dir is not None:
        d = Path(device_domain_dir)
        metrics_path = d / "device_domain_metrics.csv"
        if metrics_path.exists():
            _dde_df = pd.read_csv(metrics_path)
            # Fix #42: if multiple training strategies are present, pick the one with
            # the highest mean balanced_accuracy to avoid silently mixing strategies.
            if "training_strategy" in _dde_df.columns and _dde_df["training_strategy"].nunique() > 1:
                _best_strategy = (
                    _dde_df.groupby("training_strategy")["balanced_accuracy"]
                    .mean()
                    .idxmax()
                )
                _dde_df = _dde_df[_dde_df["training_strategy"] == _best_strategy].reset_index(drop=True)
            all_artefacts["device_transfer"] = figure_device_transfer_heatmap(_dde_df, out_dir)

    # ── run_breakthrough_pipeline ─────────────────────────────────────────────
    if breakthrough_dir is not None:
        d = Path(breakthrough_dir)

        pareto_path = d / "pareto_front.csv"
        if pareto_path.exists():
            # Fix #30: pareto_front.csv now uses corrected column names
            # (objective_purity_loss etc.) matching optimize_abstention_policy output.
            all_artefacts["pareto_front"] = figure_pareto_front(
                pd.read_csv(pareto_path), out_dir
            )

        history_path = d / "rvea_history.csv"
        if history_path.exists():
            all_artefacts["rvea_convergence"] = figure_rvea_convergence(
                pd.read_csv(history_path), out_dir
            )

        robustness_path = d / "robustness_results.csv"
        if robustness_path.exists():
            all_artefacts["robustness_analysis"] = figure_robustness_analysis(
                pd.read_csv(robustness_path), out_dir
            )

        policy_path = d / "policy_metrics.csv"
        if policy_path.exists():
            all_artefacts["policy_metrics"] = figure_policy_metrics(
                pd.read_csv(policy_path), out_dir
            )

        open_set_metrics_path = d / "open_set_unknown" / "open_set_metrics.csv"
        if open_set_metrics_path.exists():
            _os_df = pd.read_csv(open_set_metrics_path)
            _auroc = float(_os_df["unknown_auroc"].iloc[0]) if not _os_df.empty and "unknown_auroc" in _os_df.columns else 0.0
            _n_known = int(_os_df["n_known_classes"].iloc[0]) if not _os_df.empty and "n_known_classes" in _os_df.columns else 0
            # Only generate open-set figures when the evaluation is non-degenerate:
            # AUROC must exceed chance (0.5) and at least 2 known classes must be present.
            # A single known class forces AUROC → 0.5 regardless of scorer quality.
            if _auroc > 0.5 and _n_known >= 2:
                all_artefacts["open_set_metrics"] = figure_open_set_metrics(_os_df, out_dir)

        open_set_predictions_path = d / "open_set_unknown" / "open_set_predictions.csv"
        if open_set_predictions_path.exists():
            _os_pred_df = pd.read_csv(open_set_predictions_path)
            _os_met_path = d / "open_set_unknown" / "open_set_metrics.csv"
            _skip_dist = False
            if _os_met_path.exists():
                _m = pd.read_csv(_os_met_path)
                if not _m.empty and "unknown_auroc" in _m.columns:
                    _skip_dist = float(_m["unknown_auroc"].iloc[0]) <= 0.5
            if not _skip_dist:
                all_artefacts["open_set_score_distribution"] = figure_open_set_score_distribution(
                    _os_pred_df, out_dir
                )

        composition_comparison_path = d / "composition_aware" / "model_comparison.csv"
        if composition_comparison_path.exists():
            all_artefacts["composition_model_comparison"] = figure_composition_model_comparison(
                pd.read_csv(composition_comparison_path), out_dir
            )

        composition_metrics_path = d / "composition_aware" / "composition_aware_metrics_flat.csv"
        if composition_metrics_path.exists():
            all_artefacts["composition_structured_metrics"] = figure_composition_structured_metrics(
                pd.read_csv(composition_metrics_path), out_dir
            )

        composition_domain_path = d / "composition_aware" / "composition_aware_domain_metrics.csv"
        if composition_domain_path.exists():
            all_artefacts["composition_domain_performance"] = figure_composition_domain_performance(
                pd.read_csv(composition_domain_path), out_dir
            )

        focus_pair_path = d / "focus_pair_summary.csv"
        if focus_pair_path.exists():
            fp_df = pd.read_csv(focus_pair_path)
            if not fp_df.empty:
                all_artefacts["focus_pairs_error_rates"] = figure_focus_pairs_error_rates(
                    fp_df, out_dir, title="High-Confusion Class Pairs (RVEA Focus Pairs)"
                )

        dangerous_pairs_path = d / "dangerous_pairs.csv"
        if dangerous_pairs_path.exists():
            dp_df = pd.read_csv(dangerous_pairs_path)
            if not dp_df.empty:
                all_artefacts["dangerous_pairs_error_rates"] = figure_focus_pairs_error_rates(
                    dp_df, out_dir, title="Dangerous Class Pairs (Risk-Control Screen)"
                )

        policy_path = d / "policy.json"
        if policy_path.exists():
            import json as _json
            policy_dict = _json.loads(policy_path.read_text(encoding="utf-8"))
            all_artefacts["policy_thresholds"] = figure_policy_thresholds(
                policy_dict, out_dir
            )

    # ── circularity_proxy ─────────────────────────────────────────────────────
    if circularity_dir is not None:
        d = Path(circularity_dir)
        proxy_path = d / "circularity_proxy_table.csv"
        if proxy_path.exists():
            all_artefacts["circularity_proxy"] = figure_circularity_proxy(
                pd.read_csv(proxy_path), out_dir
            )

    return all_artefacts


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Generate all publication figures from saved pipeline output directories."
    )
    parser.add_argument("--group-split-dir",   default=None)
    parser.add_argument("--pca-dir",           default=None)
    parser.add_argument("--freeze-dir",        default=None)
    parser.add_argument("--device-domain-dir", default=None)
    parser.add_argument("--breakthrough-dir",  default=None)
    parser.add_argument("--circularity-dir",   default=None)
    parser.add_argument("--pub-output-dir",    default="pub_outputs")
    args = parser.parse_args()

    artefacts = generate_all_figures(
        group_split_dir   = args.group_split_dir,
        pca_dir           = args.pca_dir,
        freeze_dir        = args.freeze_dir,
        device_domain_dir = args.device_domain_dir,
        breakthrough_dir  = args.breakthrough_dir,
        circularity_dir   = args.circularity_dir,
        pub_output_dir    = args.pub_output_dir,
    )
    import json as _json_cli
    manifest = {k: list(v.keys()) for k, v in artefacts.items()}
    manifest_path = Path(args.pub_output_dir) / "figures_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_json_cli.dumps(manifest, indent=2), encoding="utf-8")
