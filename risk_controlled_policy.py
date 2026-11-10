from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from circularity_proxy import build_proxy_table
from reliability import (
    apply_abstention_policy,
    evaluate_policy,
    normalize_abstention_policy,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_risk_controlled_policy"


@dataclass(frozen=True)
class RiskControlTargets:
    target_purity: float = 0.98
    min_coverage: float = 0.30
    max_manual_review_rate: float = 0.70
    min_worst_domain_performance: float = 0.60


@dataclass(frozen=True)
class UtilityWeights:
    correct_accept_reward_scale: float = 1.0
    wrong_accept_penalty_scale: float = 1.0
    manual_review_cost: float = -0.05
    dominant_mismatch_penalty: float = 0.30
    fraction_distance_penalty: float = 1.20
    difficulty_penalty_scale: float = 0.80


def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def _canonical_pair_key(label_a: str, label_b: str) -> str:
    ordered = sorted([str(label_a).strip(), str(label_b).strip()])
    return f"{ordered[0]}||{ordered[1]}"


def _ensure_top2_columns(prediction_df: pd.DataFrame) -> pd.DataFrame:
    df = prediction_df.copy()
    if "predicted_label" not in df.columns:
        raise ValueError("prediction_df must contain 'predicted_label'.")

    proba_cols = [c for c in df.columns if str(c).startswith("proba::")]
    if "second_best_label" not in df.columns:
        if len(proba_cols) >= 2:
            probs = df[proba_cols].to_numpy(dtype=float)
            class_labels = np.asarray([str(c).split("::", 1)[1] for c in proba_cols], dtype=object)
            order = np.argsort(probs, axis=1)
            second_idx = order[:, -2]
            df["second_best_label"] = class_labels[second_idx].astype(str)
            df["second_best_probability"] = probs[np.arange(len(df)), second_idx]
        else:
            df["second_best_label"] = ""
            df["second_best_probability"] = 0.0
    if "top2_pair" not in df.columns:
        df["top2_pair"] = [
            _canonical_pair_key(a, b)
            for a, b in zip(df["predicted_label"].astype(str), df["second_best_label"].astype(str))
        ]
    return df


def build_sorting_utility_matrix(
    class_labels: Sequence[str],
    *,
    weights: UtilityWeights | Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    if weights is None:
        weights = UtilityWeights()
    elif isinstance(weights, Mapping):
        weights = UtilityWeights(**dict(weights))

    proxy_df = build_proxy_table(class_labels)
    if proxy_df.empty:
        return {}

    fraction_cols = [c for c in proxy_df.columns if str(c).startswith("fraction_")]
    proxy_df = proxy_df.set_index("composition_code", drop=False)

    utility_matrix: dict[str, dict[str, float]] = {}
    for true_label, true_row in proxy_df.iterrows():
        row_map: dict[str, float] = {}
        true_rec = float(true_row["recoverability_proxy"])
        true_diff = float(true_row["sorting_difficulty_proxy"])
        true_dom = str(true_row["dominant_component"])
        true_frac = true_row[fraction_cols].to_numpy(dtype=float)
        for pred_label, pred_row in proxy_df.iterrows():
            pred_rec = float(pred_row["recoverability_proxy"])
            pred_dom = str(pred_row["dominant_component"])
            pred_frac = pred_row[fraction_cols].to_numpy(dtype=float)
            frac_distance = float(np.abs(true_frac - pred_frac).sum() / 2.0)
            dominant_mismatch = 1.0 if true_dom != pred_dom else 0.0
            if str(true_label) == str(pred_label):
                utility = weights.correct_accept_reward_scale * true_rec
            else:
                contamination_harm = (
                    weights.fraction_distance_penalty * frac_distance
                    + weights.dominant_mismatch_penalty * dominant_mismatch
                    + weights.difficulty_penalty_scale * true_diff
                    + max(0.0, true_rec - pred_rec)
                )
                utility = -weights.wrong_accept_penalty_scale * contamination_harm
            row_map[str(pred_label)] = float(utility)
        utility_matrix[str(true_label)] = row_map
    return utility_matrix


def summarize_dangerous_pairs(
    prediction_df: pd.DataFrame,
    y_true: Sequence[str],
    domains: Sequence[str],
    *,
    min_errors: int = 2,
    top_k: int = 8,
) -> pd.DataFrame:
    df = _ensure_top2_columns(prediction_df)
    work = pd.DataFrame({
        "domain": pd.Series(domains).astype(str).to_numpy(),
        "y_true": pd.Series(y_true).astype(str).to_numpy(),
        "predicted_label": df["predicted_label"].astype(str).to_numpy(),
        "second_best_label": df["second_best_label"].astype(str).to_numpy(),
        "top2_pair": df["top2_pair"].astype(str).to_numpy(),
        "max_probability": pd.to_numeric(df.get("max_probability", 0.0), errors="coerce").fillna(0.0).to_numpy(),
        "margin": pd.to_numeric(df.get("margin", 0.0), errors="coerce").fillna(0.0).to_numpy(),
        "normalized_entropy": pd.to_numeric(df.get("normalized_entropy", 0.0), errors="coerce").fillna(0.0).to_numpy(),
    })
    work["is_error"] = (work["y_true"] != work["predicted_label"]).astype(int)
    out = (
        work.groupby(["domain", "predicted_label", "top2_pair"], dropna=False)
        .agg(
            n=("is_error", "size"),
            n_errors=("is_error", "sum"),
            mean_confidence=("max_probability", "mean"),
            mean_margin=("margin", "mean"),
            mean_entropy=("normalized_entropy", "mean"),
        )
        .reset_index()
    )
    out = out[out["n_errors"] >= int(min_errors)].copy()
    if out.empty:
        return out
    out["error_rate"] = out["n_errors"] / out["n"].clip(lower=1)
    out = out.sort_values(["n_errors", "error_rate", "mean_confidence"], ascending=[False, False, False]).head(int(top_k)).reset_index(drop=True)
    return out


def _score_policy_candidate(metrics, targets: RiskControlTargets) -> float:
    score = 0.0
    if metrics.purity < targets.target_purity:
        score += 2000.0 * (targets.target_purity - metrics.purity)
    if metrics.coverage < targets.min_coverage:
        score += 600.0 * (targets.min_coverage - metrics.coverage)
    if metrics.manual_review_rate > targets.max_manual_review_rate:
        score += 400.0 * (metrics.manual_review_rate - targets.max_manual_review_rate)
    if metrics.worst_domain_performance < targets.min_worst_domain_performance:
        score += 1200.0 * (targets.min_worst_domain_performance - metrics.worst_domain_performance)

    score += 100.0 * (1.0 - metrics.purity)
    score += 20.0 * (1.0 - metrics.coverage)
    score += 40.0 * (1.0 - metrics.worst_domain_performance)
    score += 5.0 * metrics.manual_review_rate
    if metrics.realized_utility is not None:
        score -= 20.0 * float(metrics.realized_utility)
    return float(score)


def select_risk_controlled_policy(
    prediction_df: pd.DataFrame,
    y_true: Sequence[str],
    domains: Sequence[str],
    *,
    base_policy: Mapping[str, Any] | None = None,
    targets: RiskControlTargets | Mapping[str, Any] | None = None,
    dangerous_pairs_df: pd.DataFrame | None = None,
    top_k_pairs: int = 4,
    threshold_grid: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    margin_grid: Sequence[float] = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20),
    utility_weights: UtilityWeights | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    df = _ensure_top2_columns(prediction_df)
    y_true = pd.Series(y_true).astype(str).to_numpy()
    domains = pd.Series(domains).astype(str).to_numpy()
    if len(df) != len(y_true) or len(df) != len(domains):
        raise ValueError("prediction_df, y_true and domains must have the same length.")

    if targets is None:
        targets = RiskControlTargets()
    elif isinstance(targets, Mapping):
        targets = RiskControlTargets(**dict(targets))

    normalized = normalize_abstention_policy(base_policy)
    unique_domains = sorted(pd.Series(domains).dropna().astype(str).unique().tolist())
    base_thresholds = {d: float(normalized["thresholds_by_domain"].get(d, 0.50)) for d in unique_domains}
    base_margin = float(normalized.get("min_margin", 0.0))

    if dangerous_pairs_df is None:
        dangerous_pairs_df = summarize_dangerous_pairs(df, y_true, domains, min_errors=2, top_k=top_k_pairs)
    else:
        dangerous_pairs_df = dangerous_pairs_df.copy().head(int(top_k_pairs)).reset_index(drop=True)

    utility_matrix = build_sorting_utility_matrix(sorted(pd.Series(y_true).unique().tolist()), weights=utility_weights)
    manual_review_cost = UtilityWeights().manual_review_cost if utility_weights is None or isinstance(utility_weights, Mapping) else utility_weights.manual_review_cost

    candidate_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_policy: dict[str, Any] | None = None
    best_metrics = None

    threshold_grid = sorted({float(x) for x in threshold_grid}.union(set(base_thresholds.values())))
    margin_grid = sorted({float(x) for x in margin_grid}.union({base_margin}))

    # Use shared threshold candidates around the base policy for conservative search.
    for threshold in threshold_grid:
        thresholds_by_domain = {domain: max(base_thresholds[domain], float(threshold)) for domain in unique_domains}
        for min_margin in margin_grid:
            candidate_policy = {
                **normalized,
                "thresholds_by_domain": thresholds_by_domain,
                "min_margin": float(min_margin),
                "utility_matrix": utility_matrix,
                "manual_review_cost": float(manual_review_cost),
            }

            thresholds_by_domain_and_pair: dict[str, dict[str, float]] = {}
            max_entropy_by_domain_and_pair: dict[str, dict[str, float]] = {}
            blocked_pairs_by_domain: dict[str, list[str]] = {}
            focus_pairs: list[dict[str, Any]] = []
            for _, row in dangerous_pairs_df.iterrows():
                domain = str(row["domain"])
                pair = str(row["top2_pair"])
                thresholds_by_domain_and_pair.setdefault(domain, {})
                max_entropy_by_domain_and_pair.setdefault(domain, {})
                blocked_pairs_by_domain.setdefault(domain, [])
                tightened = min(0.995, thresholds_by_domain[domain] + 0.05 + 0.10 * float(row.get("error_rate", 0.0)))
                thresholds_by_domain_and_pair[domain][pair] = float(tightened)
                max_entropy_by_domain_and_pair[domain][pair] = min(0.50, float(row.get("mean_entropy", 0.50)))
                if float(row.get("error_rate", 0.0)) >= 0.75 and int(row.get("n_errors", 0)) >= 3:
                    blocked_pairs_by_domain[domain].append(pair)
                focus_pairs.append({
                    "domain": domain,
                    "predicted_label": str(row.get("predicted_label", "")),
                    "pair": pair,
                    "n_errors": int(row.get("n_errors", 0)),
                    "error_rate": float(row.get("error_rate", 0.0)),
                })

            candidate_policy["thresholds_by_domain_and_pair"] = thresholds_by_domain_and_pair
            candidate_policy["max_entropy_by_domain_and_pair"] = max_entropy_by_domain_and_pair
            candidate_policy["blocked_pairs_by_domain"] = blocked_pairs_by_domain
            candidate_policy["focus_pairs"] = focus_pairs

            accepted_mask = apply_abstention_policy(df, domains, policy=candidate_policy)
            metrics = evaluate_policy(
                y_true=y_true,
                domains=domains,
                prediction_df=df,
                accepted_mask=accepted_mask,
                thresholds_by_domain=thresholds_by_domain,
                min_margin=float(min_margin),
                policy=candidate_policy,
            )
            score = _score_policy_candidate(metrics, targets)
            row = {
                "score": score,
                "purity": float(metrics.purity),
                "coverage": float(metrics.coverage),
                "manual_review_rate": float(metrics.manual_review_rate),
                "worst_domain_performance": float(metrics.worst_domain_performance),
                "accepted_accuracy": float(metrics.accepted_accuracy),
                "accepted_balanced_accuracy": float(metrics.accepted_balanced_accuracy),
                "accepted_macro_f1": float(metrics.accepted_macro_f1),
                "contamination_rate": float(metrics.contamination_rate),
                "expected_utility": metrics.expected_utility,
                "realized_utility": metrics.realized_utility,
                "min_margin": float(min_margin),
            }
            for domain, value in thresholds_by_domain.items():
                row[f"threshold::{domain}"] = float(value)
            candidate_rows.append(row)
            if score < best_score:
                best_score = score
                best_policy = candidate_policy
                best_metrics = metrics

    if best_policy is None or best_metrics is None:
        raise ValueError("Could not select a risk-controlled policy.")

    return {
        "selected_policy": best_policy,
        "selected_metrics": best_metrics,
        "candidate_df": pd.DataFrame(candidate_rows).sort_values(["score", "purity"], ascending=[True, False]).reset_index(drop=True),
        "dangerous_pairs_df": dangerous_pairs_df,
        "summary": {
            "selection_targets": asdict(targets),
            "utility_weights": asdict(UtilityWeights() if utility_weights is None or isinstance(utility_weights, Mapping) else utility_weights),
            "best_score": float(best_score),
            "n_candidates": int(len(candidate_rows)),
            "dangerous_pairs": dangerous_pairs_df.to_dict(orient="records"),
        },
    }


def save_risk_control_outputs(output_dir: str | Path, outputs: Mapping[str, Any]) -> dict[str, str]:
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    candidate_df = outputs.get("candidate_df")
    if isinstance(candidate_df, pd.DataFrame):
        path = output_dir / "risk_controlled_policy_candidates.csv"
        candidate_df.to_csv(path, index=False)
        paths["candidate_df"] = str(path)

    dangerous_pairs_df = outputs.get("dangerous_pairs_df")
    if isinstance(dangerous_pairs_df, pd.DataFrame):
        path = output_dir / "dangerous_pairs.csv"
        dangerous_pairs_df.to_csv(path, index=False)
        paths["dangerous_pairs_df"] = str(path)

    summary = outputs.get("summary")
    if summary is not None:
        path = output_dir / "risk_control_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths["summary"] = str(path)

    selected_policy = outputs.get("selected_policy")
    if selected_policy is not None:
        path = output_dir / "risk_controlled_policy.json"
        path.write_text(json.dumps(selected_policy, indent=2), encoding="utf-8")
        paths["selected_policy"] = str(path)

    return paths
