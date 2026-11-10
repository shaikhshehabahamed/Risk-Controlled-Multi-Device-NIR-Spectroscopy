from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

import preprocessing
from device_domain_experiments import build_multidevice_dataset
from reliability import (
    apply_abstention_policy,
    fit_calibrated_model,
    make_group_holdout_split,
    predict_with_uncertainty,
)


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_open_set_unknown"
DEFAULT_DEVICES = ["Benchtop_N", "Handheld_F", "Handheld_S"]
UNKNOWN_LABEL = "__unknown__"
MANUAL_REVIEW_LABEL = "__manual_review__"


@dataclass(frozen=True)
class OpenSetSplit:
    known_fit_groups: tuple[str, ...]
    known_calibration_groups: tuple[str, ...]
    known_test_groups: tuple[str, ...]
    unknown_calibration_groups: tuple[str, ...]
    unknown_test_groups: tuple[str, ...]
    known_classes: tuple[str, ...]
    unknown_classes: tuple[str, ...]
    calibration_strategy: str


@dataclass(frozen=True)
class OpenSetRejector:
    method: str
    threshold: float
    threshold_selection: str
    target_known_acceptance: float
    unknown_label: str = UNKNOWN_LABEL


@dataclass(frozen=True)
class OpenSetMetrics:
    unknown_auroc: Optional[float]
    unknown_auprc: Optional[float]
    known_acceptance_rate: float
    unknown_rejection_rate: float
    combined_open_set_accuracy: float
    combined_open_set_macro_f1: float
    accepted_known_accuracy: float
    accepted_known_balanced_accuracy: float
    accepted_known_macro_f1: float
    known_test_size: int
    unknown_test_size: int
    n_known_classes: int
    n_unknown_classes: int
    unknown_detection_f1: Optional[float]
    scoring_method: str
    threshold: float
    threshold_selection: str



def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()



def _normalize_group_labels(group_labels: pd.DataFrame) -> pd.DataFrame:
    required = {"specimen_group", "composition_code"}
    missing = required - set(group_labels.columns)
    if missing:
        raise ValueError(f"group_labels is missing required columns: {sorted(missing)}")

    normalized = group_labels[["specimen_group", "composition_code"]].copy()
    normalized["specimen_group"] = normalized["specimen_group"].astype(str)
    normalized["composition_code"] = normalized["composition_code"].astype(str)
    normalized = normalized.drop_duplicates().reset_index(drop=True)

    n_labels_per_group = normalized.groupby("specimen_group")["composition_code"].nunique()
    bad_groups = n_labels_per_group[n_labels_per_group > 1]
    if not bad_groups.empty:
        raise ValueError(
            "Each specimen_group must map to exactly one composition_code before grouped splitting. "
            f"Found inconsistent labels for groups: {bad_groups.index.tolist()[:10]}"
        )
    return normalized



def choose_unknown_classes(
    group_labels: pd.DataFrame,
    *,
    n_unknown_classes: int = 2,
    min_groups_per_unknown_class: int = 3,
    random_state: int = 42,
    explicit_unknown_classes: Sequence[str] | None = None,
) -> list[str]:
    group_labels = _normalize_group_labels(group_labels)
    counts = (
        group_labels.groupby("composition_code")["specimen_group"]
        .nunique()
        .sort_values(ascending=True)
    )
    eligible = counts[counts >= int(min_groups_per_unknown_class)].index.astype(str).tolist()
    if explicit_unknown_classes is not None:
        requested = [str(x) for x in explicit_unknown_classes]
        missing = sorted(set(requested) - set(eligible))
        if missing:
            raise ValueError(
                "Some explicit_unknown_classes do not satisfy the minimum-group requirement or are absent: "
                f"{missing}. Eligible examples: {eligible[:10]}"
            )
        return requested

    if len(eligible) < int(n_unknown_classes):
        raise ValueError(
            f"Not enough eligible classes to select {int(n_unknown_classes)} unknown classes. "
            f"Eligible classes={eligible}"
        )

    rng = np.random.default_rng(int(random_state))
    chosen = rng.choice(np.asarray(eligible, dtype=object), size=int(n_unknown_classes), replace=False)
    return sorted(pd.Series(chosen).astype(str).tolist())



def make_open_set_group_splits(
    group_labels: pd.DataFrame,
    *,
    unknown_classes: Sequence[str],
    known_outer_test_size: float = 0.30,
    known_inner_calibration_size: float = 0.25,
    unknown_calibration_size: float = 0.50,
    outer_random_state: int = 42,
    inner_random_state: int = 142,
    unknown_random_state: int = 242,
    max_seed_tries: int = 500,
) -> OpenSetSplit:
    group_labels = _normalize_group_labels(group_labels)
    unknown_classes = sorted({str(x) for x in unknown_classes})
    if not unknown_classes:
        raise ValueError("unknown_classes must contain at least one class.")

    unknown_mask = group_labels["composition_code"].astype(str).isin(unknown_classes)
    known_group_labels = group_labels.loc[~unknown_mask].reset_index(drop=True)
    unknown_group_labels = group_labels.loc[unknown_mask].reset_index(drop=True)
    if known_group_labels.empty or unknown_group_labels.empty:
        raise ValueError("Open-set split requires both known and unknown groups.")

    outer = make_group_holdout_split(
        known_group_labels,
        test_size=known_outer_test_size,
        random_state=outer_random_state,
        max_seed_tries=max_seed_tries,
    )

    train_pool = known_group_labels[
        known_group_labels["specimen_group"].astype(str).isin(outer.train_groups)
    ].reset_index(drop=True)
    inner = make_group_holdout_split(
        train_pool,
        test_size=known_inner_calibration_size,
        random_state=inner_random_state,
        max_seed_tries=max_seed_tries,
    )
    known_fit_groups = tuple(pd.Series(inner.train_groups).astype(str).tolist())
    known_calibration_groups = tuple(pd.Series(inner.test_groups).astype(str).tolist())
    calibration_strategy = "heldout_known_calibration"

    unknown_split = make_group_holdout_split(
        unknown_group_labels,
        test_size=unknown_calibration_size,
        random_state=unknown_random_state,
        max_seed_tries=max_seed_tries,
    )
    # Fix #14: make_group_holdout_split(test_size=unknown_calibration_size) places
    # `unknown_calibration_size` fraction into .test_groups and the remainder into
    # .train_groups.  Assign variable names to match those semantics.
    unknown_calibration_groups = tuple(pd.Series(unknown_split.test_groups).astype(str).tolist())
    unknown_test_groups = tuple(pd.Series(unknown_split.train_groups).astype(str).tolist())

    known_classes = tuple(sorted(known_group_labels["composition_code"].astype(str).unique().tolist()))

    return OpenSetSplit(
        known_fit_groups=tuple(sorted(set(known_fit_groups))),
        known_calibration_groups=tuple(sorted(set(known_calibration_groups))),
        known_test_groups=tuple(sorted(set(pd.Series(outer.test_groups).astype(str).tolist()))),
        unknown_calibration_groups=tuple(sorted(set(unknown_calibration_groups))),
        unknown_test_groups=tuple(sorted(set(unknown_test_groups))),
        known_classes=known_classes,
        unknown_classes=tuple(sorted(unknown_classes)),
        calibration_strategy=calibration_strategy,
    )



def _ensure_prediction_columns(prediction_df: pd.DataFrame) -> pd.DataFrame:
    required = {"max_probability", "margin", "normalized_entropy", "predicted_label"}
    missing = required - set(prediction_df.columns)
    if missing:
        raise ValueError(f"prediction_df is missing required columns: {sorted(missing)}")
    return prediction_df.copy()



def compute_unknown_score(prediction_df: pd.DataFrame, *, method: str = "blended") -> np.ndarray:
    df = _ensure_prediction_columns(prediction_df)
    method = str(method).strip().lower()

    max_prob = df["max_probability"].to_numpy(dtype=float)
    margin = np.clip(df["margin"].to_numpy(dtype=float), 0.0, 1.0)
    entropy = np.clip(df["normalized_entropy"].to_numpy(dtype=float), 0.0, 1.0)

    if method in {"max_probability", "max_prob", "msp"}:
        score = 1.0 - max_prob
    elif method == "margin":
        score = 1.0 - margin
    elif method == "entropy":
        score = entropy
    elif method == "blended":
        score = ((1.0 - max_prob) + (1.0 - margin) + entropy) / 3.0
    else:
        raise ValueError(
            "Unsupported open-set score method. Expected one of: "
            "'max_probability', 'margin', 'entropy', 'blended'."
        )
    return np.asarray(score, dtype=float)



def fit_open_set_rejector(
    known_calibration_predictions: pd.DataFrame,
    *,
    unknown_calibration_predictions: pd.DataFrame | None = None,
    method: str = "blended",
    threshold_selection: str = "known_acceptance",
    target_known_acceptance: float = 0.95,
    unknown_label: str = UNKNOWN_LABEL,
) -> OpenSetRejector:
    known_scores = compute_unknown_score(known_calibration_predictions, method=method)
    threshold_selection = str(threshold_selection).strip().lower()
    target_known_acceptance = float(target_known_acceptance)
    if known_scores.size == 0:
        raise ValueError("known_calibration_predictions must contain at least one row.")

    if threshold_selection == "known_acceptance" or unknown_calibration_predictions is None or len(unknown_calibration_predictions) == 0:
        # Fix #33 (documentation): this threshold is the `target_known_acceptance`-th
        # quantile of scores computed on the *calibration* known set, not a held-out
        # test set.  The resulting threshold accepts exactly `target_known_acceptance`
        # fraction of calibration-known samples by construction, but its acceptance
        # rate on new known samples depends on distribution shift.  Users should
        # provide a sufficiently large and representative calibration set for the
        # threshold to generalise well.
        threshold = float(np.quantile(known_scores, target_known_acceptance))
        selection_name = "known_acceptance"
    elif threshold_selection in {"known_unknown_balanced", "balanced"}:
        unknown_scores = compute_unknown_score(unknown_calibration_predictions, method=method)
        candidate_thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))
        best_threshold = float(candidate_thresholds[0])
        best_objective = -np.inf
        for threshold in candidate_thresholds:
            known_accept = float(np.mean(known_scores <= threshold)) if known_scores.size else 0.0
            unknown_reject = float(np.mean(unknown_scores > threshold)) if unknown_scores.size else 0.0
            objective = 0.5 * (known_accept + unknown_reject)
            if objective > best_objective:
                best_objective = float(objective)
                best_threshold = float(threshold)
        threshold = best_threshold
        selection_name = "known_unknown_balanced"
    else:
        raise ValueError(
            "Unsupported threshold_selection. Expected one of: 'known_acceptance', 'known_unknown_balanced'."
        )

    return OpenSetRejector(
        method=str(method),
        threshold=float(threshold),
        threshold_selection=selection_name,
        target_known_acceptance=target_known_acceptance,
        unknown_label=str(unknown_label),
    )



def apply_open_set_rejector(
    prediction_df: pd.DataFrame,
    rejector: OpenSetRejector,
    *,
    domains: Sequence[str] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    df = _ensure_prediction_columns(prediction_df)
    unknown_score = compute_unknown_score(df, method=rejector.method)
    is_unknown_pred = unknown_score > float(rejector.threshold)

    out = df.copy()
    out["unknown_score"] = unknown_score
    out["is_unknown_pred"] = is_unknown_pred.astype(int)
    out["predicted_open_label"] = np.where(
        is_unknown_pred,
        str(rejector.unknown_label),
        out["predicted_label"].astype(str),
    )

    if policy is not None:
        if domains is None:
            raise ValueError("domains must be provided when applying an abstention policy on top of open-set rejection.")
        policy_accept = apply_abstention_policy(
            prediction_df=df,
            domains=domains,
            policy=policy,
        )
        final_label = out["predicted_open_label"].astype(str).to_numpy()
        keep_mask = (~is_unknown_pred) & (~policy_accept)
        final_label[keep_mask] = MANUAL_REVIEW_LABEL
        out["policy_accept"] = policy_accept.astype(int)
        out["predicted_open_label"] = final_label
    return out



def evaluate_open_set_predictions(
    *,
    known_predictions: pd.DataFrame,
    y_known_true: Sequence[str],
    unknown_predictions: pd.DataFrame,
    rejector: OpenSetRejector,
    known_domains: Sequence[str] | None = None,
    unknown_domains: Sequence[str] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> tuple[OpenSetMetrics, pd.DataFrame]:
    y_known_true = pd.Series(y_known_true).astype(str).to_numpy()
    if len(known_predictions) != len(y_known_true):
        raise ValueError("known_predictions and y_known_true must have the same length.")

    known_out = apply_open_set_rejector(
        known_predictions,
        rejector,
        domains=known_domains,
        policy=policy,
    )
    unknown_out = apply_open_set_rejector(
        unknown_predictions,
        rejector,
        domains=unknown_domains,
        policy=policy,
    )

    known_out = known_out.copy()
    known_out["true_open_label"] = y_known_true
    known_out["is_unknown_true"] = 0
    unknown_out = unknown_out.copy()
    unknown_out["true_open_label"] = str(rejector.unknown_label)
    unknown_out["is_unknown_true"] = 1

    combined = pd.concat([known_out, unknown_out], axis=0, ignore_index=True)
    y_true_unknown = combined["is_unknown_true"].to_numpy(dtype=int)
    y_score_unknown = combined["unknown_score"].to_numpy(dtype=float)
    y_pred_unknown = combined["is_unknown_pred"].to_numpy(dtype=int)

    unknown_detection_f1: Optional[float] = float(
        f1_score(y_true_unknown, y_pred_unknown, zero_division=0)
    )
    # Fix #31: roc_auc_score / average_precision_score raise ValueError when only one
    # class is present in y_true (e.g., no unknown samples in combined set).
    if len(np.unique(y_true_unknown)) < 2:
        unknown_auroc: Optional[float] = None
        unknown_auprc: Optional[float] = None
    else:
        unknown_auroc = float(roc_auc_score(y_true_unknown, y_score_unknown))
        unknown_auprc = float(average_precision_score(y_true_unknown, y_score_unknown))

    known_acceptance_rate = float(np.mean(known_out["is_unknown_pred"].to_numpy(dtype=int) == 0)) if len(known_out) else 0.0
    unknown_rejection_rate = float(np.mean(unknown_out["is_unknown_pred"].to_numpy(dtype=int) == 1)) if len(unknown_out) else 0.0

    combined_true = combined["true_open_label"].astype(str).to_numpy()
    combined_pred = combined["predicted_open_label"].astype(str).to_numpy()
    combined_open_set_accuracy = float(accuracy_score(combined_true, combined_pred)) if len(combined) else 0.0
    combined_open_set_macro_f1 = float(f1_score(combined_true, combined_pred, average="macro", zero_division=0)) if len(combined) else 0.0

    accepted_known_mask = known_out["predicted_open_label"].astype(str).to_numpy() == known_out["predicted_label"].astype(str).to_numpy()
    if accepted_known_mask.any():
        accepted_known_true = y_known_true[accepted_known_mask]
        accepted_known_pred = known_out.loc[accepted_known_mask, "predicted_label"].astype(str).to_numpy()
        accepted_known_accuracy = float(accuracy_score(accepted_known_true, accepted_known_pred))
        _n_labels = len(np.unique(np.concatenate([accepted_known_true, accepted_known_pred])))
        accepted_known_balanced_accuracy = (
            float(balanced_accuracy_score(accepted_known_true, accepted_known_pred))
            if _n_labels >= 2
            else float(accuracy_score(accepted_known_true, accepted_known_pred))
        )
        accepted_known_macro_f1 = float(f1_score(accepted_known_true, accepted_known_pred, average="macro", zero_division=0))
    else:
        accepted_known_accuracy = 0.0
        accepted_known_balanced_accuracy = 0.0
        accepted_known_macro_f1 = 0.0

    metrics = OpenSetMetrics(
        unknown_auroc=unknown_auroc,
        unknown_auprc=unknown_auprc,
        unknown_detection_f1=unknown_detection_f1,
        known_acceptance_rate=known_acceptance_rate,
        unknown_rejection_rate=unknown_rejection_rate,
        combined_open_set_accuracy=combined_open_set_accuracy,
        combined_open_set_macro_f1=combined_open_set_macro_f1,
        accepted_known_accuracy=accepted_known_accuracy,
        accepted_known_balanced_accuracy=accepted_known_balanced_accuracy,
        accepted_known_macro_f1=accepted_known_macro_f1,
        known_test_size=int(len(known_out)),
        unknown_test_size=int(len(unknown_out)),
        n_known_classes=int(pd.Series(y_known_true).nunique()),
        n_unknown_classes=int(1 if len(unknown_out) else 0),
        scoring_method=str(rejector.method),
        threshold=float(rejector.threshold),
        threshold_selection=str(rejector.threshold_selection),
    )
    return metrics, combined



def run_open_set_experiment(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    instruments: Sequence[str] = DEFAULT_DEVICES,
    measurement: str = "Reflectance",
    preprocessing_method: str = "snv",
    pca_components: int = 20,
    alpha: float = 1.0,
    calibration_method: str = "sigmoid",
    common_grid_points: int = 128,
    min_common_groups_for_eval: int = 5,
    n_unknown_classes: int = 2,
    min_groups_per_unknown_class: int = 3,
    explicit_unknown_classes: Sequence[str] | None = None,
    unknown_score_method: str = "blended",
    threshold_selection: str = "known_acceptance",
    target_known_acceptance: float = 0.95,
    bundle: dict | None = None,
    policy: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    dataset_path = _resolve_local_path(dataset_path)
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bundle is None:
        bundle = build_multidevice_dataset(
            dataset_path=dataset_path,
            loader_script_path=loader_script_path,
            instruments=instruments,
            measurement=measurement,
            common_grid_points=common_grid_points,
            min_common_groups_for_eval=min_common_groups_for_eval,
        )

    metadata = bundle["metadata"].copy().reset_index(drop=True)
    X_raw = np.asarray(bundle["X"], dtype=float)
    y = pd.Series(bundle["y"]).astype(str).to_numpy()
    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)
    group_labels = metadata[["specimen_group", "composition_code"]].drop_duplicates().reset_index(drop=True)

    unknown_classes = choose_unknown_classes(
        group_labels,
        n_unknown_classes=n_unknown_classes,
        min_groups_per_unknown_class=min_groups_per_unknown_class,
        random_state=random_state,
        explicit_unknown_classes=explicit_unknown_classes,
    )
    split = make_open_set_group_splits(
        group_labels,
        unknown_classes=unknown_classes,
        outer_random_state=random_state,
        inner_random_state=random_state + 100,
        unknown_random_state=random_state + 200,
    )

    fit_mask = metadata["specimen_group"].astype(str).isin(split.known_fit_groups)
    known_cal_mask = metadata["specimen_group"].astype(str).isin(split.known_calibration_groups)
    known_test_mask = metadata["specimen_group"].astype(str).isin(split.known_test_groups)
    unknown_cal_mask = metadata["specimen_group"].astype(str).isin(split.unknown_calibration_groups)
    unknown_test_mask = metadata["specimen_group"].astype(str).isin(split.unknown_test_groups)

    preprocessing_state = preprocessing.fit_preprocessing_state(
        X_raw[fit_mask],
        wavelengths,
        method=preprocessing_method,
    )

    X_fit = preprocessing.preprocess_spectra(
        X_raw[fit_mask],
        wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )

    X_known_cal = preprocessing.preprocess_spectra(
        X_raw[known_cal_mask],
        wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    ) if bool(known_cal_mask.sum()) else None

    y_known_cal = y[known_cal_mask] if bool(known_cal_mask.sum()) else None

    model_bundle = fit_calibrated_model(
        X_train=X_fit,
        y_train=y[fit_mask],
        X_cal=X_known_cal,
        y_cal=y_known_cal,
        n_components=pca_components,
        alpha=alpha,
        preprocessing_method=preprocessing_method,
        calibration_method=calibration_method,
        preprocessing_state=preprocessing_state,
    )

    if X_known_cal is None:
        known_cal_predictions = predict_with_uncertainty(model_bundle, X_fit)
    else:
        known_cal_predictions = predict_with_uncertainty(model_bundle, X_known_cal)

    if bool(unknown_cal_mask.sum()):
        X_unknown_cal = preprocessing.preprocess_spectra(
            X_raw[unknown_cal_mask],
            wavelengths,
            method=preprocessing_method,
            preprocessing_state=preprocessing_state,
        )
        unknown_cal_predictions = predict_with_uncertainty(model_bundle, X_unknown_cal)
    else:
        unknown_cal_predictions = None

    rejector = fit_open_set_rejector(
        known_calibration_predictions=known_cal_predictions,
        unknown_calibration_predictions=unknown_cal_predictions,
        method=unknown_score_method,
        threshold_selection=threshold_selection,
        target_known_acceptance=target_known_acceptance,
    )

    X_known_test = preprocessing.preprocess_spectra(
        X_raw[known_test_mask],
        wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )
    X_unknown_test = preprocessing.preprocess_spectra(
        X_raw[unknown_test_mask],
        wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )

    known_test_predictions = predict_with_uncertainty(model_bundle, X_known_test)
    unknown_test_predictions = predict_with_uncertainty(model_bundle, X_unknown_test)

    known_test_domains = metadata.loc[known_test_mask, "instrument"].astype(str).to_numpy()
    unknown_test_domains = metadata.loc[unknown_test_mask, "instrument"].astype(str).to_numpy()

    # Fix #36: evaluate_open_set_predictions already calls apply_open_set_rejector
    # internally and returns the fully annotated combined DataFrame.  Reuse it
    # directly instead of calling apply_open_set_rejector a second time.
    metrics, eval_combined = evaluate_open_set_predictions(
        known_predictions=known_test_predictions,
        y_known_true=y[known_test_mask],
        unknown_predictions=unknown_test_predictions,
        rejector=rejector,
        known_domains=known_test_domains,
        unknown_domains=unknown_test_domains,
        policy=policy,
    )

    # Attach specimen metadata to the combined predictions for the output CSV.
    known_meta = pd.DataFrame({
        "specimen_group": metadata.loc[known_test_mask, "specimen_group"].astype(str).tolist(),
        "instrument": metadata.loc[known_test_mask, "instrument"].astype(str).tolist(),
        "composition_code": y[known_test_mask].tolist(),
    })
    unknown_meta = pd.DataFrame({
        "specimen_group": metadata.loc[unknown_test_mask, "specimen_group"].astype(str).tolist(),
        "instrument": metadata.loc[unknown_test_mask, "instrument"].astype(str).tolist(),
        "composition_code": y[unknown_test_mask].tolist(),
    })
    # eval_combined rows: known rows first, then unknown rows (same order as evaluate_open_set_predictions)
    n_known = int(known_test_mask.sum())
    n_unknown = int(unknown_test_mask.sum())
    meta_combined = pd.concat([known_meta, unknown_meta], axis=0, ignore_index=True)
    full_output_predictions = pd.concat(
        [meta_combined.reset_index(drop=True), eval_combined.reset_index(drop=True)],
        axis=1,
    )
    full_output_predictions.to_csv(output_dir / "open_set_predictions.csv", index=False)

    summary = {
        "unknown_classes": list(split.unknown_classes),
        "known_classes": list(split.known_classes),
        "split": asdict(split),
        "rejector": asdict(rejector),
        "metrics": asdict(metrics),
        "dataset_subset": {
            "instrument": list(instruments),
            "measurement": measurement,
        },
    }
    (output_dir / "open_set_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(metrics)]).to_csv(output_dir / "open_set_metrics.csv", index=False)
    joblib.dump(
        {
            "model_bundle": model_bundle,
            "rejector": asdict(rejector),
            "preprocessing_state": preprocessing.serialize_preprocessing_state(preprocessing_state),
            "wavelengths": wavelengths,
            "unknown_classes": list(split.unknown_classes),
            "known_classes": list(split.known_classes),
            "policy": dict(policy) if policy is not None else None,
        },
        output_dir / "open_set_artifacts.joblib",
    )

    return {
        "summary": summary,
        "metrics": metrics,
        "predictions_df": full_output_predictions,
        "output_dir": str(output_dir),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Run an open-set / unknown-material evaluation on the NIR pipeline.")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--loader-script-path", type=str, default="nir_loader.py")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preprocessing", type=str, default="snv")
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--common-grid-points", type=int, default=128)
    parser.add_argument("--min-common-groups-for-eval", type=int, default=5)
    parser.add_argument("--n-unknown-classes", type=int, default=2)
    parser.add_argument("--min-groups-per-unknown-class", type=int, default=3)
    parser.add_argument("--unknown-score-method", type=str, default="blended")
    parser.add_argument("--threshold-selection", type=str, default="known_acceptance")
    parser.add_argument("--target-known-acceptance", type=float, default=0.95)
    args = parser.parse_args()

    run_open_set_experiment(
        dataset_path=args.dataset_path,
        loader_script_path=args.loader_script_path,
        output_dir=args.output_dir,
        preprocessing_method=args.preprocessing,
        pca_components=args.pca_components,
        alpha=args.alpha,
        common_grid_points=args.common_grid_points,
        min_common_groups_for_eval=args.min_common_groups_for_eval,
        n_unknown_classes=args.n_unknown_classes,
        min_groups_per_unknown_class=args.min_groups_per_unknown_class,
        unknown_score_method=args.unknown_score_method,
        threshold_selection=args.threshold_selection,
        target_known_acceptance=args.target_known_acceptance,
    )
    # Summary saved to <output_dir>/open_set_summary.json


if __name__ == "__main__":
    main()
