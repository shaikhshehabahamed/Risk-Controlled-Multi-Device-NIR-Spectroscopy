from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score as _accuracy_score, balanced_accuracy_score, f1_score


def _safe_balanced_accuracy(y_true, y_pred) -> float:
    """balanced_accuracy_score wrapper that avoids the sklearn UserWarning when
    only a single label is present in y_true/y_pred (which triggers an internal
    confusion_matrix call without a labels= argument).  Falls back to plain
    accuracy in the degenerate single-label case."""
    if len(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))) < 2:
        return float(_accuracy_score(y_true, y_pred))
    return float(balanced_accuracy_score(y_true, y_pred))
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from circularity_proxy import composition_to_proxy, expected_circularity_capture, realized_circularity_capture
from gpu_models import build_gpu_pca_ridge_pipeline

from sklearn.frozen import FrozenEstimator


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_reliability"
DEFAULT_INSTRUMENTS = ["Benchtop_N", "Handheld_F", "Handheld_S"]
DEFAULT_THRESHOLD = 0.5


def _canonical_pair_key(label_a: str, label_b: str) -> str:
    a = str(label_a).strip()
    b = str(label_b).strip()
    ordered = sorted([a, b])
    return f"{ordered[0]}||{ordered[1]}"


def _canonical_pair_array(labels_a: Sequence[str], labels_b: Sequence[str]) -> np.ndarray:
    return np.asarray([_canonical_pair_key(a, b) for a, b in zip(labels_a, labels_b)], dtype=object)


@dataclass
class GroupSplit:
    train_groups: np.ndarray
    test_groups: np.ndarray


@dataclass
class CalibratedModelBundle:
    base_pipeline: Pipeline
    calibrated_model: CalibratedClassifierCV
    class_labels: np.ndarray
    pca_components: int
    preprocessing_method: str
    preprocessing_state: Optional[Dict[str, Any]] = None


@dataclass
class PolicyEvaluation:
    thresholds: Dict[str, float]
    min_margin: float
    purity: float
    coverage: float
    manual_review_rate: float
    worst_domain_performance: float
    accepted_accuracy: float
    accepted_balanced_accuracy: float
    accepted_macro_f1: float
    contamination_rate: float
    expected_circularity_proxy: Optional[float]
    realized_circularity_proxy: Optional[float]
    expected_utility: Optional[float]
    realized_utility: Optional[float]
    n_accepted: int


@dataclass
class ReliabilityStudySplit:
    fit_groups: np.ndarray
    calibration_groups: np.ndarray
    test_groups: np.ndarray
    calibration_strategy: str
    outer_random_state: int
    inner_random_state: int



def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()



def build_ridge_pca_pipeline(n_components: int = 20, alpha: float = 1.0) -> Pipeline:
    """GPU-accelerated Pipeline: GPUStandardScaler → GPUPCA → GPURidgeClassifier."""
    return build_gpu_pca_ridge_pipeline(n_components=n_components, alpha=alpha)



def _validate_group_label_consistency(group_labels: pd.DataFrame) -> pd.DataFrame:
    validated = group_labels[["specimen_group", "composition_code"]].copy()
    validated["specimen_group"] = validated["specimen_group"].astype(str)
    validated["composition_code"] = validated["composition_code"].astype(str)
    validated = validated.drop_duplicates().reset_index(drop=True)

    n_labels_per_group = validated.groupby("specimen_group")["composition_code"].nunique()
    bad_groups = n_labels_per_group[n_labels_per_group > 1]
    if not bad_groups.empty:
        details = (
            validated[validated["specimen_group"].isin(bad_groups.index)]
            .sort_values(["specimen_group", "composition_code"])
            .to_dict(orient="records")
        )
        raise ValueError(
            "Each specimen_group must map to exactly one composition_code before grouped splitting. "
            f"Found inconsistent labels for {len(bad_groups)} group(s): {details[:10]}"
        )

    return validated



def make_group_holdout_split(
    group_labels: pd.DataFrame,
    test_size: float = 0.4,
    random_state: int = 42,
    max_seed_tries: int = 500,
) -> GroupSplit:
    required = {"specimen_group", "composition_code"}
    missing = required - set(group_labels.columns)
    if missing:
        raise ValueError(f"group_labels is missing required columns: {sorted(missing)}")

    group_labels = _validate_group_label_consistency(group_labels)

    for seed in range(random_state, random_state + max_seed_tries):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        dummy = np.zeros((len(group_labels), 1), dtype=float)
        y = group_labels["composition_code"].astype(str).to_numpy()
        try:
            split_result = list(splitter.split(dummy, y))
        except ValueError:
            # sklearn hard-rejects the data (e.g. a class has only 1 member and
            # cannot be stratified at all).  No point trying further seeds.
            break
        for train_idx, test_idx in split_result:
            train_groups = group_labels.iloc[train_idx]["specimen_group"].astype(str).to_numpy()
            test_groups = group_labels.iloc[test_idx]["specimen_group"].astype(str).to_numpy()
            train_classes = set(group_labels.iloc[train_idx]["composition_code"].astype(str))
            test_classes = set(group_labels.iloc[test_idx]["composition_code"].astype(str))
            if train_classes == test_classes:
                return GroupSplit(train_groups=train_groups, test_groups=test_groups)
    raise ValueError("Could not create a class-complete grouped holdout split.")



def make_nested_group_splits(
    group_labels: pd.DataFrame,
    *,
    outer_test_size: float = 0.30,
    inner_calibration_size: float = 0.25,
    outer_random_state: int = 42,
    inner_random_state: int = 142,
    max_outer_seed_tries: int = 500,
    max_inner_seed_tries: int = 500,
) -> ReliabilityStudySplit:
    group_labels = _validate_group_label_consistency(group_labels)
    outer = make_group_holdout_split(
        group_labels,
        test_size=outer_test_size,
        random_state=outer_random_state,
        max_seed_tries=max_outer_seed_tries,
    )

    train_pool = group_labels[group_labels["specimen_group"].astype(str).isin(outer.train_groups)].reset_index(drop=True)
    inner = make_group_holdout_split(
        train_pool,
        test_size=inner_calibration_size,
        random_state=inner_random_state,
        max_seed_tries=max_inner_seed_tries,
    )
    calibration_groups = inner.test_groups.astype(str)
    fit_groups = inner.train_groups.astype(str)
    calibration_strategy = "heldout_calibration"

    return ReliabilityStudySplit(
        fit_groups=np.asarray(fit_groups, dtype=object),
        calibration_groups=np.asarray(calibration_groups, dtype=object),
        test_groups=np.asarray(outer.test_groups, dtype=object),
        calibration_strategy=calibration_strategy,
        outer_random_state=int(outer_random_state),
        inner_random_state=int(inner_random_state),
    )



def _make_calibrated_classifier(estimator, method: str, cv=None, *, ensemble=None) -> CalibratedClassifierCV:
    sig = inspect.signature(CalibratedClassifierCV)
    kwargs = {"method": method}
    if "cv" in sig.parameters:
        kwargs["cv"] = cv
    if ensemble is not None and "ensemble" in sig.parameters:
        kwargs["ensemble"] = ensemble
    if "estimator" in sig.parameters:
        kwargs["estimator"] = estimator
    else:  # pragma: no cover - compatibility path for older sklearn
        kwargs["base_estimator"] = estimator
    return CalibratedClassifierCV(**kwargs)



def _build_prefit_calibrator(
    base_estimator: Pipeline,
    y_cal: Sequence[str] | np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    y_cal = pd.Series(np.asarray(y_cal).astype(str))
    min_count = int(y_cal.value_counts().min()) if not y_cal.empty else 0
    if min_count < 2:
        raise ValueError(
            "Calibration split must contain at least 2 samples per class for prefit calibration."
        )

    # cv='prefit' is the correct mode: fit the calibration layer on *all* of
    # X_cal without any internal splitting.  cv=None (default 5-fold) fails
    # when the calibration set is small (fewer samples than folds) because some
    # folds contain 0 samples, producing a 0-d array that column_or_1d rejects.
    # cv='prefit' is deprecated in sklearn 1.6 (to be removed in 1.8); we
    # suppress the FutureWarning here and will migrate once 1.8 is released.
    import warnings as _warnings
    frozen = FrozenEstimator(base_estimator)
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*prefit.*", category=FutureWarning)
        return _make_calibrated_classifier(estimator=frozen, method=method, cv="prefit", ensemble=False)



def _resolve_valid_pca_components(X_train: np.ndarray, requested_components: int) -> int:
    X_train = np.asarray(X_train)
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")
    if X_train.shape[0] < 1 or X_train.shape[1] < 1:
        raise ValueError("X_train must contain at least one sample and one feature.")

    max_valid = min(X_train.shape[0], X_train.shape[1])
    return max(1, min(int(requested_components), int(max_valid)))



def fit_calibrated_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    n_components: int = 20,
    alpha: float = 1.0,
    preprocessing_method: str = "snv",
    calibration_method: str = "sigmoid",
    preprocessing_state: Optional[Dict[str, Any]] = None,
) -> CalibratedModelBundle:
    class_labels = np.array(sorted(pd.Series(y_train).astype(str).unique().tolist()), dtype=object)
    n_components = _resolve_valid_pca_components(X_train, n_components)

    base_pipeline = build_ridge_pca_pipeline(n_components=n_components, alpha=alpha)

    calibration_args_provided = (X_cal is not None) or (y_cal is not None)
    use_prefit = X_cal is not None and y_cal is not None
    if calibration_args_provided and not use_prefit:
        raise ValueError("Provide both X_cal and y_cal, or leave both as None.")

    if use_prefit:
        y_cal_series = pd.Series(np.asarray(y_cal).astype(str))
        cal_classes = set(y_cal_series.unique().tolist())
        train_classes = set(pd.Series(np.asarray(y_train).astype(str)).unique().tolist())
        if cal_classes != train_classes:
            missing_in_cal = sorted(train_classes - cal_classes)
            extra_in_cal = sorted(cal_classes - train_classes)
            raise ValueError(
                "Held-out calibration set must contain exactly the same class set as the training set. "
                f"Missing in calibration: {missing_in_cal}; unexpected in calibration: {extra_in_cal}."
            )

    if use_prefit:
        base_pipeline.fit(X_train, np.asarray(y_train).astype(str))
        calibrated = _build_prefit_calibrator(base_pipeline, y_cal=y_cal, method=calibration_method)
        calibrated.fit(np.asarray(X_cal, dtype=float), np.asarray(y_cal).astype(str))

    if not use_prefit:
        class_counts = pd.Series(np.asarray(y_train).astype(str)).value_counts()
        min_count = int(class_counts.min()) if not class_counts.empty else 0
        if min_count < 2:
            raise ValueError(
                "Training split must contain at least 2 samples per class for cross-validated calibration. "
                f"Smallest class count: {min_count}."
            )
        cv = min(3, min_count)
        calibrated = _make_calibrated_classifier(
            estimator=base_pipeline,
            method=calibration_method,
            cv=cv,
            ensemble=False,
        )
        calibrated.fit(np.asarray(X_train, dtype=float), np.asarray(y_train).astype(str))
        # Refit a standalone base pipeline for artifact export and deterministic access to PCA.
        base_pipeline.fit(np.asarray(X_train, dtype=float), np.asarray(y_train).astype(str))

    return CalibratedModelBundle(
        base_pipeline=base_pipeline,
        calibrated_model=calibrated,
        class_labels=class_labels,
        pca_components=n_components,
        preprocessing_method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )



def _coerce_calibrated_model(model_bundle_or_model) -> CalibratedClassifierCV:
    if isinstance(model_bundle_or_model, CalibratedModelBundle):
        return model_bundle_or_model.calibrated_model
    if hasattr(model_bundle_or_model, "predict_proba") and hasattr(model_bundle_or_model, "classes_"):
        return model_bundle_or_model
    raise TypeError(
        "model_bundle_or_model must be a CalibratedModelBundle or a fitted calibrated classifier with "
        "predict_proba and classes_."
    )



def predict_with_uncertainty(model_bundle_or_model: CalibratedModelBundle | CalibratedClassifierCV, X: np.ndarray) -> pd.DataFrame:
    calibrated_model = _coerce_calibrated_model(model_bundle_or_model)
    probabilities = np.asarray(calibrated_model.predict_proba(X), dtype=float)
    class_labels = np.asarray(calibrated_model.classes_, dtype=object)
    pred_idx = np.argmax(probabilities, axis=1)
    max_prob = probabilities[np.arange(len(probabilities)), pred_idx]

    if probabilities.shape[1] > 1:
        # Fix #47: np.argsort is unstable on ties, so argsort[:, -2] can equal
        # argmax[:, -1] when the top two probabilities are exactly equal.
        # Walk the sorted order to find the first index that differs from pred_idx.
        sorted_idx = np.argsort(probabilities, axis=1)  # ascending
        n_samples = len(probabilities)
        second_idx = np.empty(n_samples, dtype=int)
        for _s in range(n_samples):
            _si = sorted_idx[_s, -2]          # candidate: second-largest position
            if _si != pred_idx[_s]:
                second_idx[_s] = _si
            else:
                # Tie: walk further left until we find a different index
                _found = False
                for _k in range(-3, -(probabilities.shape[1] + 1), -1):
                    if sorted_idx[_s, _k] != pred_idx[_s]:
                        second_idx[_s] = sorted_idx[_s, _k]
                        _found = True
                        break
                if not _found:
                    second_idx[_s] = sorted_idx[_s, -2]   # all classes tied; keep original
        second_best = probabilities[np.arange(n_samples), second_idx]
        second_best_label = class_labels[second_idx].astype(str)
    else:
        second_idx = np.zeros(len(probabilities), dtype=int)
        second_best = np.zeros(len(probabilities), dtype=float)
        second_best_label = np.full(len(probabilities), "", dtype=object)

    margin = max_prob - second_best
    if probabilities.shape[1] > 1:
        entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1) / np.log(probabilities.shape[1])
    else:
        entropy = np.zeros(len(probabilities), dtype=float)

    predicted_label = class_labels[pred_idx].astype(str)
    df = pd.DataFrame({
        "predicted_label": predicted_label,
        "second_best_label": second_best_label,
        "top2_pair": _canonical_pair_array(predicted_label, second_best_label),
        "max_probability": max_prob,
        "second_best_probability": second_best,
        "margin": margin,
        "normalized_entropy": entropy,
    })
    for j, class_label in enumerate(class_labels):
        df[f"proba::{class_label}"] = probabilities[:, j]
    return df



def _normalize_float_mapping(mapping: Mapping[str, Any] | None, *, field_name: str) -> Dict[str, float]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided.")
    normalized: Dict[str, float] = {}
    for key, value in mapping.items():
        normalized[str(key)] = float(value)
    return normalized



def _normalize_nested_float_mapping(
    mapping: Mapping[str, Mapping[str, Any]] | None,
    *,
    field_name: str,
) -> Dict[str, Dict[str, float]]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided.")
    normalized: Dict[str, Dict[str, float]] = {}
    for outer_key, inner_mapping in mapping.items():
        if inner_mapping is None:
            continue
        if not isinstance(inner_mapping, Mapping):
            raise ValueError(
                f"{field_name}[{outer_key!r}] must itself be a mapping of label -> float."
            )
        inner_normalized: Dict[str, float] = {}
        for inner_key, value in inner_mapping.items():
            inner_normalized[str(inner_key)] = float(value)
        if inner_normalized:
            normalized[str(outer_key)] = inner_normalized
    return normalized


def _normalize_nested_string_collection_mapping(
    mapping: Mapping[str, Sequence[Any] | Mapping[str, Any]] | None,
    *,
    field_name: str,
) -> Dict[str, list[str]]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided.")
    normalized: Dict[str, list[str]] = {}
    for outer_key, values in mapping.items():
        if values is None:
            continue
        if isinstance(values, Mapping):
            normalized_values = [str(inner_key) for inner_key, flag in values.items() if bool(flag)]
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            normalized_values = [str(value) for value in values]
        else:
            raise ValueError(
                f"{field_name}[{outer_key!r}] must be a sequence of strings or mapping of string -> truthy flag."
            )
        normalized[str(outer_key)] = sorted({str(value) for value in normalized_values if str(value).strip()})
    return normalized



def normalize_abstention_policy(policy: Mapping[str, Any] | None) -> Dict[str, Any]:
    if policy is None:
        policy = {}
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be a mapping when provided.")

    normalized = dict(policy)
    normalized["thresholds_by_domain"] = _normalize_float_mapping(
        policy.get("thresholds_by_domain"),
        field_name="policy['thresholds_by_domain']",
    )
    normalized["min_margin"] = float(policy.get("min_margin", 0.0))

    max_entropy = policy.get("max_entropy")
    normalized["max_entropy"] = None if max_entropy is None else float(max_entropy)

    normalized["thresholds_by_domain_and_label"] = _normalize_nested_float_mapping(
        policy.get("thresholds_by_domain_and_label"),
        field_name="policy['thresholds_by_domain_and_label']",
    )
    normalized["min_margin_by_domain_and_label"] = _normalize_nested_float_mapping(
        policy.get("min_margin_by_domain_and_label"),
        field_name="policy['min_margin_by_domain_and_label']",
    )
    normalized["max_entropy_by_domain_and_label"] = _normalize_nested_float_mapping(
        policy.get("max_entropy_by_domain_and_label"),
        field_name="policy['max_entropy_by_domain_and_label']",
    )
    normalized["thresholds_by_domain_and_pair"] = _normalize_nested_float_mapping(
        policy.get("thresholds_by_domain_and_pair"),
        field_name="policy['thresholds_by_domain_and_pair']",
    )
    normalized["min_margin_by_domain_and_pair"] = _normalize_nested_float_mapping(
        policy.get("min_margin_by_domain_and_pair"),
        field_name="policy['min_margin_by_domain_and_pair']",
    )
    normalized["max_entropy_by_domain_and_pair"] = _normalize_nested_float_mapping(
        policy.get("max_entropy_by_domain_and_pair"),
        field_name="policy['max_entropy_by_domain_and_pair']",
    )
    normalized["blocked_pairs_by_domain"] = _normalize_nested_string_collection_mapping(
        policy.get("blocked_pairs_by_domain"),
        field_name="policy['blocked_pairs_by_domain']",
    )
    return normalized



def apply_abstention_policy(
    prediction_df: pd.DataFrame,
    domains: Sequence[str],
    thresholds_by_domain: Optional[Dict[str, float]] = None,
    min_margin: float | None = None,
    thresholds_by_domain_and_label: Mapping[str, Mapping[str, float]] | None = None,
    min_margin_by_domain_and_label: Mapping[str, Mapping[str, float]] | None = None,
    max_entropy: float | None = None,
    max_entropy_by_domain_and_label: Mapping[str, Mapping[str, float]] | None = None,
    thresholds_by_domain_and_pair: Mapping[str, Mapping[str, float]] | None = None,
    min_margin_by_domain_and_pair: Mapping[str, Mapping[str, float]] | None = None,
    max_entropy_by_domain_and_pair: Mapping[str, Mapping[str, float]] | None = None,
    blocked_pairs_by_domain: Mapping[str, Sequence[str] | Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> np.ndarray:
    required_cols = {"max_probability", "margin"}
    missing = required_cols - set(prediction_df.columns)
    if missing:
        raise ValueError(f"prediction_df is missing required columns: {sorted(missing)}")

    # Fix #26: the explicit keyword arguments (thresholds_by_domain, min_margin, etc.)
    # were documented but their values were always overwritten by the policy dict.
    # Build a merged policy so that explicit kwargs take precedence over the dict,
    # giving callers a consistent and predictable interface.
    _explicit: Dict[str, Any] = {}
    if thresholds_by_domain is not None:
        _explicit["thresholds_by_domain"] = thresholds_by_domain
    if min_margin is not None:
        _explicit["min_margin"] = float(min_margin)
    if max_entropy is not None:
        _explicit["max_entropy"] = float(max_entropy)
    if thresholds_by_domain_and_label is not None:
        _explicit["thresholds_by_domain_and_label"] = thresholds_by_domain_and_label
    if min_margin_by_domain_and_label is not None:
        _explicit["min_margin_by_domain_and_label"] = min_margin_by_domain_and_label
    if max_entropy_by_domain_and_label is not None:
        _explicit["max_entropy_by_domain_and_label"] = max_entropy_by_domain_and_label
    if thresholds_by_domain_and_pair is not None:
        _explicit["thresholds_by_domain_and_pair"] = thresholds_by_domain_and_pair
    if min_margin_by_domain_and_pair is not None:
        _explicit["min_margin_by_domain_and_pair"] = min_margin_by_domain_and_pair
    if max_entropy_by_domain_and_pair is not None:
        _explicit["max_entropy_by_domain_and_pair"] = max_entropy_by_domain_and_pair
    if blocked_pairs_by_domain is not None:
        _explicit["blocked_pairs_by_domain"] = blocked_pairs_by_domain
    # Merge: policy dict provides the base; explicit kwargs override it.
    _merged_policy: Dict[str, Any] = dict(policy) if isinstance(policy, Mapping) else {}
    _merged_policy.update(_explicit)

    normalized_policy = normalize_abstention_policy(_merged_policy)

    domains = pd.Series(domains).astype(str).to_numpy()
    if len(domains) != len(prediction_df):
        raise ValueError(
            "domains must have the same length as prediction_df. "
            f"Got {len(domains)} domains for {len(prediction_df)} prediction rows."
        )

    if thresholds_by_domain is None:
        thresholds_by_domain = normalized_policy["thresholds_by_domain"]
    thresholds_by_domain = _normalize_float_mapping(
        thresholds_by_domain,
        field_name="thresholds_by_domain",
    )

    if thresholds_by_domain_and_label is None:
        thresholds_by_domain_and_label = normalized_policy["thresholds_by_domain_and_label"]
    thresholds_by_domain_and_label = _normalize_nested_float_mapping(
        thresholds_by_domain_and_label,
        field_name="thresholds_by_domain_and_label",
    )

    if thresholds_by_domain_and_pair is None:
        thresholds_by_domain_and_pair = normalized_policy["thresholds_by_domain_and_pair"]
    thresholds_by_domain_and_pair = _normalize_nested_float_mapping(
        thresholds_by_domain_and_pair,
        field_name="thresholds_by_domain_and_pair",
    )

    if min_margin is None:
        min_margin = float(normalized_policy["min_margin"])
    min_margin = float(min_margin)

    if min_margin_by_domain_and_label is None:
        min_margin_by_domain_and_label = normalized_policy["min_margin_by_domain_and_label"]
    min_margin_by_domain_and_label = _normalize_nested_float_mapping(
        min_margin_by_domain_and_label,
        field_name="min_margin_by_domain_and_label",
    )

    if min_margin_by_domain_and_pair is None:
        min_margin_by_domain_and_pair = normalized_policy["min_margin_by_domain_and_pair"]
    min_margin_by_domain_and_pair = _normalize_nested_float_mapping(
        min_margin_by_domain_and_pair,
        field_name="min_margin_by_domain_and_pair",
    )

    if max_entropy is None:
        max_entropy = normalized_policy["max_entropy"]
    max_entropy = None if max_entropy is None else float(max_entropy)

    if max_entropy_by_domain_and_label is None:
        max_entropy_by_domain_and_label = normalized_policy["max_entropy_by_domain_and_label"]
    max_entropy_by_domain_and_label = _normalize_nested_float_mapping(
        max_entropy_by_domain_and_label,
        field_name="max_entropy_by_domain_and_label",
    )

    if max_entropy_by_domain_and_pair is None:
        max_entropy_by_domain_and_pair = normalized_policy["max_entropy_by_domain_and_pair"]
    max_entropy_by_domain_and_pair = _normalize_nested_float_mapping(
        max_entropy_by_domain_and_pair,
        field_name="max_entropy_by_domain_and_pair",
    )

    if blocked_pairs_by_domain is None:
        blocked_pairs_by_domain = normalized_policy["blocked_pairs_by_domain"]
    blocked_pairs_by_domain = _normalize_nested_string_collection_mapping(
        blocked_pairs_by_domain,
        field_name="blocked_pairs_by_domain",
    )

    if thresholds_by_domain:
        unknown_domains = sorted(set(domains) - set(thresholds_by_domain))
        if unknown_domains:
            raise ValueError(
                "Thresholds are missing for domain(s): "
                f"{unknown_domains}. Known domains: {sorted(thresholds_by_domain)}."
            )

    uses_predicted_label = bool(
        thresholds_by_domain_and_label
        or min_margin_by_domain_and_label
        or max_entropy_by_domain_and_label
        or thresholds_by_domain_and_pair
        or min_margin_by_domain_and_pair
        or max_entropy_by_domain_and_pair
        or blocked_pairs_by_domain
    )
    if uses_predicted_label and "predicted_label" not in prediction_df.columns:
        raise ValueError(
            "prediction_df must contain 'predicted_label' when class-aware policy overrides are used."
        )

    uses_pair_overrides = bool(
        thresholds_by_domain_and_pair
        or min_margin_by_domain_and_pair
        or max_entropy_by_domain_and_pair
        or blocked_pairs_by_domain
    )
    if uses_pair_overrides and "second_best_label" not in prediction_df.columns:
        raise ValueError(
            "prediction_df must contain 'second_best_label' when pair-aware policy overrides are used."
        )

    uses_entropy = max_entropy is not None or bool(max_entropy_by_domain_and_label) or bool(max_entropy_by_domain_and_pair)
    if uses_entropy and "normalized_entropy" not in prediction_df.columns:
        raise ValueError(
            "prediction_df must contain 'normalized_entropy' when entropy-based abstention is used."
        )

    predicted_labels = (
        prediction_df["predicted_label"].astype(str).to_numpy()
        if uses_predicted_label
        else np.full(len(prediction_df), "", dtype=object)
    )
    second_best_labels = (
        prediction_df["second_best_label"].astype(str).to_numpy()
        if uses_pair_overrides
        else np.full(len(prediction_df), "", dtype=object)
    )
    pair_keys = (
        _canonical_pair_array(predicted_labels, second_best_labels)
        if uses_pair_overrides
        else np.full(len(prediction_df), "", dtype=object)
    )
    entropies = (
        prediction_df["normalized_entropy"].to_numpy(dtype=float)
        if uses_entropy
        else np.full(len(prediction_df), -np.inf, dtype=float)
    )

    accepted = np.zeros(len(prediction_df), dtype=bool)
    for i, domain in enumerate(domains):
        predicted_label = str(predicted_labels[i])
        pair_key = str(pair_keys[i])

        domain_threshold = float(thresholds_by_domain.get(domain, DEFAULT_THRESHOLD))
        if domain in thresholds_by_domain_and_label and predicted_label in thresholds_by_domain_and_label[domain]:
            domain_threshold = float(thresholds_by_domain_and_label[domain][predicted_label])
        if domain in thresholds_by_domain_and_pair and pair_key in thresholds_by_domain_and_pair[domain]:
            domain_threshold = max(domain_threshold, float(thresholds_by_domain_and_pair[domain][pair_key]))

        margin_threshold = float(min_margin)
        if domain in min_margin_by_domain_and_label and predicted_label in min_margin_by_domain_and_label[domain]:
            margin_threshold = float(min_margin_by_domain_and_label[domain][predicted_label])
        if domain in min_margin_by_domain_and_pair and pair_key in min_margin_by_domain_and_pair[domain]:
            margin_threshold = max(margin_threshold, float(min_margin_by_domain_and_pair[domain][pair_key]))

        entropy_threshold = np.inf if max_entropy is None else float(max_entropy)
        if domain in max_entropy_by_domain_and_label and predicted_label in max_entropy_by_domain_and_label[domain]:
            entropy_threshold = float(max_entropy_by_domain_and_label[domain][predicted_label])
        if domain in max_entropy_by_domain_and_pair and pair_key in max_entropy_by_domain_and_pair[domain]:
            entropy_threshold = min(entropy_threshold, float(max_entropy_by_domain_and_pair[domain][pair_key]))

        if domain in blocked_pairs_by_domain and pair_key in set(blocked_pairs_by_domain[domain]):
            accepted[i] = False
            continue

        accepted[i] = bool(
            prediction_df.iloc[i]["max_probability"] >= domain_threshold
            and prediction_df.iloc[i]["margin"] >= margin_threshold
            and entropies[i] <= entropy_threshold
        )
    return accepted



def _supports_circularity_proxy_labels(labels: Sequence[str]) -> bool:
    """Return True only if every unique label can be parsed as a composition code."""
    unique_labels = pd.Series(labels).dropna().astype(str).unique().tolist()
    for label in unique_labels:
        try:
            composition_to_proxy(label)
        except Exception:
            # Fix #4: propagation of ValueError here crashed evaluate_policy;
            # return False so the caller skips the circularity proxy computation.
            return False
    return True


def evaluate_policy(
    y_true: Sequence[str],
    domains: Sequence[str],
    prediction_df: pd.DataFrame,
    accepted_mask: np.ndarray,
    thresholds_by_domain: Dict[str, float],
    min_margin: float,
    policy: Mapping[str, Any] | None = None,
) -> PolicyEvaluation:
    y_true = pd.Series(y_true).astype(str).to_numpy()
    domains = pd.Series(domains).astype(str).to_numpy()
    y_pred = prediction_df["predicted_label"].astype(str).to_numpy()
    accepted_mask = np.asarray(accepted_mask, dtype=bool)
    normalized_policy = normalize_abstention_policy(policy)

    n_total = len(y_true)
    n_accepted = int(accepted_mask.sum())
    coverage = float(n_accepted / n_total) if n_total else 0.0
    manual_review_rate = 1.0 - coverage

    if n_accepted > 0:
        accepted_true = y_true[accepted_mask]
        accepted_pred = y_pred[accepted_mask]
        purity = float(np.mean(accepted_true == accepted_pred))
        accepted_accuracy = purity
        accepted_balanced_accuracy = _safe_balanced_accuracy(accepted_true, accepted_pred)
        accepted_macro_f1 = float(f1_score(accepted_true, accepted_pred, average="macro", zero_division=0))
        contamination_rate = float(1.0 - purity)
    else:
        purity = 0.0
        accepted_accuracy = 0.0
        accepted_balanced_accuracy = 0.0
        accepted_macro_f1 = 0.0
        contamination_rate = 0.0

    domain_scores = []
    domain_table = pd.DataFrame({
        "domain": domains,
        "accepted": accepted_mask,
        "correct": (y_true == y_pred).astype(int),
    })
    for _, sub in domain_table.groupby("domain"):
        accepted_sub = sub[sub["accepted"]]
        if accepted_sub.empty:
            domain_scores.append(0.0)
            continue
        score = float(accepted_sub["correct"].mean())
        domain_scores.append(score)
    worst_domain_performance = float(min(domain_scores)) if domain_scores else 0.0

    labels_support_proxy = _supports_circularity_proxy_labels(np.concatenate([y_true, y_pred]))
    if labels_support_proxy:
        expected_proxy = float(
            np.mean(
                expected_circularity_capture(
                    predicted_labels=y_pred,
                    acceptance_mask=accepted_mask,
                    confidence=prediction_df["max_probability"].to_numpy(dtype=float),
                )
            )
        )
        realized_proxy = float(
            np.mean(
                realized_circularity_capture(
                    true_labels=y_true,
                    predicted_labels=y_pred,
                    acceptance_mask=accepted_mask,
                )
            )
        )
    else:
        expected_proxy = None
        realized_proxy = None

    utility_matrix = normalized_policy.get("utility_matrix")
    manual_review_cost = normalized_policy.get("manual_review_cost")
    expected_utility = None
    realized_utility = None
    if isinstance(utility_matrix, Mapping):
        class_prob_cols = [col for col in prediction_df.columns if str(col).startswith("proba::")]
        manual_review_cost = float(-0.05 if manual_review_cost is None else manual_review_cost)
        expected_values = np.full(len(prediction_df), manual_review_cost, dtype=float)
        realized_values = np.full(len(prediction_df), manual_review_cost, dtype=float)
        if class_prob_cols:
            probabilities = prediction_df[class_prob_cols].to_numpy(dtype=float)
            proba_labels = [str(col).split("::", 1)[1] for col in class_prob_cols]
        else:
            probabilities = None
            proba_labels = []

        for i in range(len(prediction_df)):
            pred_label = str(y_pred[i])
            row_mapping = utility_matrix.get(y_true[i], {}) if isinstance(utility_matrix.get(y_true[i], {}), Mapping) else {}
            if accepted_mask[i]:
                if probabilities is not None:
                    value = 0.0
                    for j, true_candidate in enumerate(proba_labels):
                        candidate_row = utility_matrix.get(true_candidate, {})
                        if isinstance(candidate_row, Mapping):
                            value += float(probabilities[i, j]) * float(candidate_row.get(pred_label, 0.0))
                    expected_values[i] = value
                elif row_mapping:
                    expected_values[i] = float(row_mapping.get(pred_label, 0.0))
                realized_values[i] = float(row_mapping.get(pred_label, 0.0)) if row_mapping else 0.0
        expected_utility = float(np.mean(expected_values))
        realized_utility = float(np.mean(realized_values))

    return PolicyEvaluation(
        thresholds=dict(thresholds_by_domain),
        min_margin=float(min_margin),
        purity=float(purity),
        coverage=float(coverage),
        manual_review_rate=float(manual_review_rate),
        worst_domain_performance=float(worst_domain_performance),
        accepted_accuracy=float(accepted_accuracy),
        accepted_balanced_accuracy=float(accepted_balanced_accuracy),
        accepted_macro_f1=float(accepted_macro_f1),
        contamination_rate=float(contamination_rate),
        expected_circularity_proxy=expected_proxy,
        realized_circularity_proxy=realized_proxy,
        expected_utility=expected_utility,
        realized_utility=realized_utility,
        n_accepted=n_accepted,
    )



def _prepare_reliability_dataset(
    dataset_path: str | Path,
    loader_script_path: str | Path,
    instruments: Sequence[str],
    measurement: str,
    common_grid_points: int,
    min_groups_per_class: int,
    aggregate: Optional[str] = None,
) -> dict:
    import training_matrix

    df = training_matrix.load_nir_subset(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        instruments=list(instruments),
        measurement=measurement,
    )
    if df.empty:
        raise ValueError(
            "No spectra were loaded for the requested reliability dataset subset. "
            f"Instruments={list(instruments)}, measurement={measurement!r}."
        )

    if len(instruments) > 1:
        df, common_classes, class_presence = training_matrix.keep_common_classes(df, instruments=instruments)
        if df.empty or not common_classes:
            raise ValueError(
                "No composition classes are shared across the requested instruments. "
                f"Instruments={list(instruments)}, measurement={measurement!r}."
            )
        if int(min_groups_per_class) > 1:
            df, eligible_classes, class_group_summary = training_matrix.filter_classes_by_min_groups_across_instruments(
                df,
                min_groups_per_class=min_groups_per_class,
                instruments=instruments,
            )
            if df.empty or not eligible_classes:
                raise ValueError(
                    "No shared composition classes satisfy the minimum grouped-count requirement for a leakage-safe "
                    "reliability study. "
                    f"min_groups_per_class={min_groups_per_class}, instruments={list(instruments)}, measurement={measurement!r}."
                )
            common_classes = eligible_classes
            class_presence = class_group_summary
        common_grid = training_matrix.infer_common_wavelength_grid(df, instruments=instruments, n_points=common_grid_points)
        df = training_matrix.resample_dataframe_to_grid(df, common_grid)
    else:
        group_counts = (
            df.groupby("composition_code")["specimen_group"]
            .nunique()
            .sort_values(ascending=True)
        )
        eligible_classes = group_counts[group_counts >= int(min_groups_per_class)].index.astype(str).tolist()
        if not eligible_classes:
            raise ValueError(
                "No classes satisfy the minimum grouped-count requirement for the selected instrument. "
                f"min_groups_per_class={min_groups_per_class}, instrument={instruments[0]!r}, measurement={measurement!r}."
            )
        df = df[df["composition_code"].astype(str).isin(eligible_classes)].reset_index(drop=True)
        common_classes = sorted(df["composition_code"].dropna().astype(str).unique().tolist())
        class_presence = pd.DataFrame({
            "composition_code": common_classes,
            "is_common": True,
            f"n_groups_{instruments[0]}": [
                int(df[df["composition_code"].astype(str) == label]["specimen_group"].astype(str).nunique())
                for label in common_classes
            ],
        })
        class_presence["min_groups_across_instruments"] = class_presence[f"n_groups_{instruments[0]}"]
        common_grid = np.asarray(df.iloc[0]["wavelength_axis"], dtype=float)

    X, y, _, groups, wavelengths, metadata, label_map = training_matrix.build_training_matrix(df, aggregate=aggregate)
    return {
        "X": np.asarray(X, dtype=float),
        "y": np.asarray(y, dtype=object),
        "groups": np.asarray(groups, dtype=object),
        "wavelengths": np.asarray(wavelengths, dtype=float),
        "metadata": metadata.reset_index(drop=True),
        "label_map": label_map,
        "common_classes": common_classes,
        "class_presence": class_presence,
        "common_grid": np.asarray(common_grid, dtype=float),
    }



def _row_for_policy_metrics(split_name: str, metrics: PolicyEvaluation) -> dict:
    return {
        "split": split_name,
        "purity": metrics.purity,
        "coverage": metrics.coverage,
        "manual_review_rate": metrics.manual_review_rate,
        "worst_domain_performance": metrics.worst_domain_performance,
        "accepted_accuracy": metrics.accepted_accuracy,
        "accepted_balanced_accuracy": metrics.accepted_balanced_accuracy,
        "accepted_macro_f1": metrics.accepted_macro_f1,
        "contamination_rate": metrics.contamination_rate,
        "expected_circularity_proxy": metrics.expected_circularity_proxy,
        "realized_circularity_proxy": metrics.realized_circularity_proxy,
        "expected_utility": metrics.expected_utility,
        "realized_utility": metrics.realized_utility,
        "n_accepted": metrics.n_accepted,
    }



def save_reliability_outputs(output_dir: str | Path, outputs: dict) -> dict:
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summary_path = output_dir / "split_summary.json"
    split_summary_path.write_text(json.dumps(outputs["split_summary"], indent=2), encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(outputs["summary"], indent=2), encoding="utf-8")

    outputs["metrics_df"].to_csv(output_dir / "reliability_metrics.csv", index=False)
    outputs["test_predictions"].to_csv(output_dir / "test_predictions.csv", index=False)
    if outputs["calibration_predictions"] is not None and not outputs["calibration_predictions"].empty:
        outputs["calibration_predictions"].to_csv(output_dir / "calibration_predictions.csv", index=False)
    outputs["class_presence"].to_csv(output_dir / "class_presence.csv", index=False)
    outputs["label_map"].to_csv(output_dir / "label_map.csv", index=False)

    model_artifact = {
        "calibrated_model": outputs["model_bundle"].calibrated_model,
        "base_pipeline": outputs["model_bundle"].base_pipeline,
        "wavelengths": outputs["wavelengths"],
        "preprocessing_method": outputs["preprocessing_method"],
        "preprocessing_state": outputs["preprocessing_state_serialized"],
        "policy": outputs["policy"],
        "class_labels": outputs["model_bundle"].class_labels,
        "dataset_subset": {
            "instrument": list(outputs["instruments"]),
            "measurement": outputs["measurement"],
        },
    }
    model_path = output_dir / "calibrated_reliability_model.joblib"
    joblib.dump(model_artifact, model_path)
    return {
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "split_summary_path": str(split_summary_path),
    }



def run_reliability_study(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    instruments: Sequence[str] = DEFAULT_INSTRUMENTS,
    measurement: str = "Reflectance",
    preprocessing_method: str = "snv",
    pca_components: int = 20,
    alpha: float = 1.0,
    calibration_method: str = "sigmoid",
    common_grid_points: int = 128,
    min_groups_per_class: int = 3,
    outer_test_size: float = 0.30,
    inner_calibration_size: float = 0.25,
    aggregate: Optional[str] = None,
) -> dict:
    import preprocessing

    instruments = list(instruments)
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = _prepare_reliability_dataset(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        instruments=instruments,
        measurement=measurement,
        common_grid_points=common_grid_points,
        min_groups_per_class=min_groups_per_class,
        aggregate=aggregate,
    )
    metadata = dataset_bundle["metadata"].copy().reset_index(drop=True)
    X_raw = dataset_bundle["X"]
    y = pd.Series(dataset_bundle["y"]).astype(str).to_numpy()
    wavelengths = dataset_bundle["wavelengths"]

    group_labels = metadata[["specimen_group", "composition_code"]].drop_duplicates().reset_index(drop=True)
    split = make_nested_group_splits(
        group_labels,
        outer_test_size=outer_test_size,
        inner_calibration_size=inner_calibration_size,
    )

    fit_mask = metadata["specimen_group"].astype(str).isin(split.fit_groups)
    cal_mask = metadata["specimen_group"].astype(str).isin(split.calibration_groups)
    test_mask = metadata["specimen_group"].astype(str).isin(split.test_groups)
    if not bool(fit_mask.sum()) or not bool(test_mask.sum()):
        raise ValueError("Reliability study created an empty fit or test split, which is not valid.")

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
    X_test = preprocessing.preprocess_spectra(
        X_raw[test_mask],
        wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )

    if bool(cal_mask.sum()):
        X_cal = preprocessing.preprocess_spectra(
            X_raw[cal_mask],
            wavelengths,
            method=preprocessing_method,
            preprocessing_state=preprocessing_state,
        )
        y_cal = y[cal_mask]
    else:
        X_cal = None
        y_cal = None

    model_bundle = fit_calibrated_model(
        X_train=X_fit,
        y_train=y[fit_mask],
        X_cal=X_cal,
        y_cal=y_cal,
        n_components=pca_components,
        alpha=alpha,
        preprocessing_method=preprocessing_method,
        calibration_method=calibration_method,
        preprocessing_state=preprocessing_state,
    )

    policy = {
        "thresholds_by_domain": {domain: float(DEFAULT_THRESHOLD) for domain in sorted(metadata["instrument"].astype(str).unique().tolist())},
        "min_margin": 0.0,
        "selected_from": "default_fixed_threshold",
    }

    calibration_predictions = None
    calibration_metrics = None
    if X_cal is not None and y_cal is not None and len(X_cal) > 0:
        calibration_predictions = predict_with_uncertainty(model_bundle, X_cal)
        calibration_meta = metadata.loc[cal_mask].reset_index(drop=True)
        calibration_predictions = pd.concat([calibration_meta, calibration_predictions], axis=1)
        calibration_accept = apply_abstention_policy(
            prediction_df=calibration_predictions,
            domains=calibration_meta["instrument"].astype(str).to_numpy(),
            thresholds_by_domain=policy["thresholds_by_domain"],
            min_margin=policy["min_margin"],
            policy=policy,
        )
        calibration_predictions["is_accepted"] = calibration_accept.astype(int)
        calibration_predictions["requires_manual_review"] = (~calibration_accept).astype(int)
        calibration_metrics = evaluate_policy(
            y_true=y[cal_mask],
            domains=calibration_meta["instrument"].astype(str).to_numpy(),
            prediction_df=calibration_predictions,
            accepted_mask=calibration_accept,
            thresholds_by_domain=policy["thresholds_by_domain"],
            min_margin=policy["min_margin"],
            policy=policy,
        )

    test_predictions = predict_with_uncertainty(model_bundle, X_test)
    test_meta = metadata.loc[test_mask].reset_index(drop=True)
    test_predictions = pd.concat([test_meta, test_predictions], axis=1)
    test_accept = apply_abstention_policy(
        prediction_df=test_predictions,
        domains=test_meta["instrument"].astype(str).to_numpy(),
        thresholds_by_domain=policy["thresholds_by_domain"],
        min_margin=policy["min_margin"],
        policy=policy,
    )
    test_predictions["is_accepted"] = test_accept.astype(int)
    test_predictions["requires_manual_review"] = (~test_accept).astype(int)
    test_metrics = evaluate_policy(
        y_true=y[test_mask],
        domains=test_meta["instrument"].astype(str).to_numpy(),
        prediction_df=test_predictions,
        accepted_mask=test_accept,
        thresholds_by_domain=policy["thresholds_by_domain"],
        min_margin=policy["min_margin"],
        policy=policy,
    )

    metrics_rows = []
    if calibration_metrics is not None:
        metrics_rows.append(_row_for_policy_metrics("calibration", calibration_metrics))
    metrics_rows.append(_row_for_policy_metrics("test", test_metrics))
    metrics_df = pd.DataFrame(metrics_rows)

    split_summary = {
        "fit_groups": sorted(pd.Series(split.fit_groups).astype(str).tolist()),
        "calibration_groups": sorted(pd.Series(split.calibration_groups).astype(str).tolist()),
        "test_groups": sorted(pd.Series(split.test_groups).astype(str).tolist()),
        "calibration_strategy": split.calibration_strategy,
        "outer_random_state": split.outer_random_state,
        "inner_random_state": split.inner_random_state,
        "n_total_groups": int(pd.Series(dataset_bundle["groups"]).nunique()),
        "n_fit_groups": int(pd.Series(split.fit_groups).nunique()),
        "n_calibration_groups": int(pd.Series(split.calibration_groups).nunique()),
        "n_test_groups": int(pd.Series(split.test_groups).nunique()),
    }

    summary = {
        "dataset_path": str(_resolve_local_path(dataset_path)),
        "measurement": measurement,
        "instruments": instruments,
        "n_samples_total": int(len(X_raw)),
        "n_classes": int(pd.Series(y).nunique()),
        "n_groups": int(pd.Series(dataset_bundle["groups"]).nunique()),
        "min_groups_per_class": int(min_groups_per_class),
        "pca_components": int(model_bundle.pca_components),
        "preprocessing_method": preprocessing_method,
        "calibration_method": calibration_method,
        "calibration_strategy": split.calibration_strategy,
        "test_metrics": {
            "purity": test_metrics.purity,
            "coverage": test_metrics.coverage,
            "manual_review_rate": test_metrics.manual_review_rate,
            "worst_domain_performance": test_metrics.worst_domain_performance,
            "accepted_accuracy": test_metrics.accepted_accuracy,
            "accepted_balanced_accuracy": test_metrics.accepted_balanced_accuracy,
            "accepted_macro_f1": test_metrics.accepted_macro_f1,
            "expected_circularity_proxy": test_metrics.expected_circularity_proxy,
            "realized_circularity_proxy": test_metrics.realized_circularity_proxy,
        },
    }

    outputs = {
        "summary": summary,
        "split_summary": split_summary,
        "metrics_df": metrics_df,
        "test_predictions": test_predictions,
        "calibration_predictions": calibration_predictions,
        "class_presence": dataset_bundle["class_presence"],
        "label_map": dataset_bundle["label_map"],
        "model_bundle": model_bundle,
        "wavelengths": wavelengths,
        "preprocessing_method": preprocessing_method,
        "preprocessing_state_serialized": preprocessing.serialize_preprocessing_state(preprocessing_state),
        "policy": policy,
        "measurement": measurement,
        "instruments": instruments,
    }
    outputs["saved_paths"] = save_reliability_outputs(output_dir, outputs)
    outputs["output_dir"] = str(output_dir)
    return outputs



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a calibrated reliability study on the local NIR dataset.")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--loader-script-path", type=str, default="nir_loader.py")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--instrument", type=str, nargs="*", default=list(DEFAULT_INSTRUMENTS))
    parser.add_argument("--measurement", type=str, default="Reflectance")
    parser.add_argument("--preprocessing", type=str, default="snv")
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--calibration-method", type=str, default="sigmoid")
    parser.add_argument("--common-grid-points", type=int, default=128)
    parser.add_argument("--min-groups-per-class", type=int, default=3)
    parser.add_argument("--outer-test-size", type=float, default=0.30)
    parser.add_argument("--inner-calibration-size", type=float, default=0.25)
    parser.add_argument("--aggregate", type=str, default=None)
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    run_reliability_study(
        dataset_path=args.dataset_path,
        loader_script_path=args.loader_script_path,
        output_dir=args.output_dir,
        instruments=args.instrument,
        measurement=args.measurement,
        preprocessing_method=args.preprocessing,
        pca_components=args.pca_components,
        alpha=args.alpha,
        calibration_method=args.calibration_method,
        common_grid_points=args.common_grid_points,
        min_groups_per_class=args.min_groups_per_class,
        outer_test_size=args.outer_test_size,
        inner_calibration_size=args.inner_calibration_size,
        aggregate=args.aggregate,
    )
    # Summary saved to <output_dir>/summary.json


if __name__ == "__main__":
    main()
