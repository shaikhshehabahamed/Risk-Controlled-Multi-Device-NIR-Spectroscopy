from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from reliability import apply_abstention_policy, evaluate_policy


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_rvea_optimizer"


def _compute_pareto_front(df: pd.DataFrame, objective_cols: list[str]) -> pd.DataFrame:
    """Return the Pareto-non-dominated subset of *df*.

    Each column in *objective_cols* is treated as a minimisation objective
    (all three stored as 1-metric, so lower is better).  Candidate *j*
    dominates candidate *i* when *j* is no worse on every objective and
    strictly better on at least one.

    Complexity is O(n²) in the number of candidates, which is acceptable
    for the typical population sizes used here (≤ a few thousand rows).
    """
    if df.empty or not objective_cols:
        return df.copy()
    values = df[objective_cols].to_numpy(dtype=float)
    n = len(values)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i when j ≤ i on all objectives and j < i on at least one.
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                is_dominated[i] = True
                break
    return df[~is_dominated].reset_index(drop=True)


def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def _canonical_pair_key(a: str, b: str) -> str:
    ordered = sorted([str(a).strip(), str(b).strip()])
    return f"{ordered[0]}||{ordered[1]}"


def _ensure_pair_columns(prediction_df: pd.DataFrame) -> pd.DataFrame:
    df = prediction_df.copy()
    if "predicted_label" not in df.columns:
        raise ValueError("prediction_df must contain 'predicted_label'.")
    if "second_best_label" not in df.columns:
        proba_cols = [c for c in df.columns if str(c).startswith("proba::")]
        if len(proba_cols) >= 2:
            probs = df[proba_cols].to_numpy(dtype=float)
            labels = np.asarray([str(c).split("::", 1)[1] for c in proba_cols], dtype=object)
            order = np.argsort(probs, axis=1)
            second_idx = order[:, -2]
            df["second_best_label"] = labels[second_idx].astype(str)
        else:
            df["second_best_label"] = ""
    if "top2_pair" not in df.columns:
        df["top2_pair"] = [
            _canonical_pair_key(a, b)
            for a, b in zip(df["predicted_label"].astype(str), df["second_best_label"].astype(str))
        ]
    return df


def _summarize_focus_pairs(prediction_df: pd.DataFrame, y_true: Sequence[str], domains: Sequence[str], *, min_errors: int, max_focus_pairs: int) -> pd.DataFrame:
    df = _ensure_pair_columns(prediction_df)
    work = pd.DataFrame({
        "domain": pd.Series(domains).astype(str).to_numpy(),
        "y_true": pd.Series(y_true).astype(str).to_numpy(),
        "predicted_label": df["predicted_label"].astype(str).to_numpy(),
        "top2_pair": df["top2_pair"].astype(str).to_numpy(),
        "normalized_entropy": pd.to_numeric(df.get("normalized_entropy", 0.0), errors="coerce").fillna(0.0).to_numpy(),
    })
    work["is_error"] = (work["y_true"] != work["predicted_label"]).astype(int)
    out = (
        work.groupby(["domain", "predicted_label", "top2_pair"], dropna=False)
        .agg(n=("is_error", "size"), n_errors=("is_error", "sum"), mean_entropy=("normalized_entropy", "mean"))
        .reset_index()
    )
    out = out[out["n_errors"] >= int(min_errors)].copy()
    if out.empty:
        return out
    out["error_rate"] = out["n_errors"] / out["n"].clip(lower=1)
    return out.sort_values(["n_errors", "error_rate"], ascending=[False, False]).head(int(max_focus_pairs)).reset_index(drop=True)


def optimize_abstention_policy(
    prediction_df: pd.DataFrame,
    y_true: Sequence[str],
    domains: Sequence[str],
    device_order: Sequence[str],
    threshold_bounds: tuple[float, float] = (0.35, 0.995),
    min_margin_bounds: tuple[float, float] = (0.0, 0.60),
    entropy_bounds: tuple[float, float] = (0.02, 0.60),
    focus_min_errors: int = 2,
    max_focus_pairs: int = 4,
    population_size: int = 64,
    generations: int = 40,
    random_state: int = 42,
) -> dict[str, Any]:
    """Multi-objective abstention-policy search via uniform random sampling.

    Evaluates ``population_size * generations`` (min 16) independently drawn
    candidates.  ``population_size`` and ``generations`` are budget controls
    only — no evolutionary selection, crossover, or mutation is performed.

    ``focus_pairs`` inside each candidate policy is informational metadata
    recording which class-pairs received tightened thresholds.  It has no
    effect on ``apply_abstention_policy``; the operative per-pair controls
    are in ``thresholds_by_domain_and_pair`` and
    ``max_entropy_by_domain_and_pair``.

    Returns a ``pareto_df`` of Pareto-non-dominated candidates (w.r.t.
    objective_purity_loss, objective_coverage_loss,
    objective_worst_domain_loss) and a ``candidate_df`` with the full
    evaluation history.
    """
    y_true = pd.Series(y_true).astype(str).to_numpy()
    domains = pd.Series(domains).astype(str).to_numpy()
    rng = np.random.default_rng(int(random_state))
    df = _ensure_pair_columns(prediction_df)
    focus_pair_summary_df = _summarize_focus_pairs(df, y_true, domains, min_errors=focus_min_errors, max_focus_pairs=max_focus_pairs)

    n_candidates = max(16, int(population_size) * max(1, int(generations)))
    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    selected_policy = None
    best_score = -np.inf

    lo_t, hi_t = map(float, threshold_bounds)
    lo_m, hi_m = map(float, min_margin_bounds)
    lo_e, hi_e = map(float, entropy_bounds)

    for idx in range(n_candidates):
        thresholds_by_domain = {d: float(rng.uniform(lo_t, hi_t)) for d in device_order}
        min_margin = float(rng.uniform(lo_m, hi_m))
        max_entropy = float(rng.uniform(lo_e, hi_e))
        thresholds_by_domain_and_pair: dict[str, dict[str, float]] = {}
        max_entropy_by_domain_and_pair: dict[str, dict[str, float]] = {}
        focus_pairs: list[dict[str, Any]] = []
        for _, pair_row in focus_pair_summary_df.iterrows():
            domain = str(pair_row["domain"])
            pair_key = str(pair_row["top2_pair"])
            thresholds_by_domain_and_pair.setdefault(domain, {})[pair_key] = min(0.995, thresholds_by_domain.get(domain, lo_t) + 0.05)
            max_entropy_by_domain_and_pair.setdefault(domain, {})[pair_key] = min(max_entropy, float(pair_row.get("mean_entropy", max_entropy)))
            focus_pairs.append({
                "domain": domain,
                "predicted_label": str(pair_row.get("predicted_label", "")),
                "pair": pair_key,
                "n_errors": int(pair_row.get("n_errors", 0)),
                "error_rate": float(pair_row.get("error_rate", 0.0)),
            })
        policy = {
            "thresholds_by_domain": thresholds_by_domain,
            "min_margin": min_margin,
            "max_entropy": max_entropy,
            "thresholds_by_domain_and_pair": thresholds_by_domain_and_pair,
            "max_entropy_by_domain_and_pair": max_entropy_by_domain_and_pair,
            "focus_pairs": focus_pairs,
        }
        accepted = apply_abstention_policy(df, domains, policy=policy)
        metrics = evaluate_policy(y_true, domains, df, accepted, thresholds_by_domain, min_margin, policy=policy)
        score = float(metrics.purity + 0.60 * metrics.coverage + 0.40 * metrics.worst_domain_performance)
        row = {
            "candidate_idx": int(idx),
            "purity": float(metrics.purity),
            "coverage": float(metrics.coverage),
            "worst_domain_performance": float(metrics.worst_domain_performance),
            "accepted_accuracy": float(metrics.accepted_accuracy),
            "accepted_macro_f1": float(metrics.accepted_macro_f1),
            "manual_review_rate": float(metrics.manual_review_rate),
            "objective_purity_loss": 1.0 - float(metrics.purity),
            "objective_coverage_loss": 1.0 - float(metrics.coverage),
            "objective_worst_domain_loss": 1.0 - float(metrics.worst_domain_performance),
            "min_margin": float(min_margin),
            "max_entropy": float(max_entropy),
            "selection_score": score,
        }
        for d, v in thresholds_by_domain.items():
            row[f"threshold::{d}"] = float(v)
        rows.append(row)
        history_rows.append({"generation": int(idx // max(1, population_size)), "candidate_idx": int(idx), "selection_score": score})
        if score > best_score:
            best_score = score
            selected_policy = policy

    candidate_df = pd.DataFrame(rows).sort_values(["selection_score", "purity", "coverage"], ascending=[False, False, False]).reset_index(drop=True)

    # Fix #2: compute a genuine Pareto front by filtering dominated candidates.
    _objective_cols = ["objective_purity_loss", "objective_coverage_loss", "objective_worst_domain_loss"]
    pareto_df = _compute_pareto_front(candidate_df, _objective_cols)
    return {
        "pareto_df": pareto_df,
        "candidate_df": candidate_df,
        "history_df": pd.DataFrame(history_rows),
        "selected_policy": selected_policy or {"thresholds_by_domain": {d: lo_t for d in device_order}, "min_margin": lo_m, "max_entropy": hi_e, "focus_pairs": []},
        "focus_pair_summary_df": focus_pair_summary_df,
    }


def save_policy_optimization_outputs(output_dir: str | Path, outputs: Mapping[str, Any]) -> dict[str, str]:
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if isinstance(outputs.get("pareto_df"), pd.DataFrame):
        path = output_dir / "pareto_front.csv"
        outputs["pareto_df"].to_csv(path, index=False)
        paths["pareto_df"] = str(path)
    if isinstance(outputs.get("candidate_df"), pd.DataFrame):
        path = output_dir / "policy_candidates.csv"
        outputs["candidate_df"].to_csv(path, index=False)
        paths["candidate_df"] = str(path)
    if isinstance(outputs.get("history_df"), pd.DataFrame):
        path = output_dir / "rvea_history.csv"
        outputs["history_df"].to_csv(path, index=False)
        paths["history_df"] = str(path)
    if isinstance(outputs.get("focus_pair_summary_df"), pd.DataFrame):
        path = output_dir / "focus_pair_summary.csv"
        outputs["focus_pair_summary_df"].to_csv(path, index=False)
        paths["focus_pair_summary_df"] = str(path)
    if outputs.get("selected_policy") is not None:
        path = output_dir / "optimized_policy.json"
        path.write_text(json.dumps(outputs["selected_policy"], indent=2), encoding="utf-8")
        paths["selected_policy"] = str(path)
    return paths
