from __future__ import annotations
import logging

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from gpu_models import build_gpu_ridge_pipeline


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_group_split_baseline"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_trusted_project_file(
    path_like: str | Path,
    *,
    allowed_suffixes: tuple[str, ...],
    purpose: str,
) -> Path:
    path = _resolve_local_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Required {purpose} not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected {purpose} to be a file, got: {path}")
    if path.suffix.lower() not in {suffix.lower() for suffix in allowed_suffixes}:
        raise ValueError(
            f"Unsafe {purpose} suffix for {path.name!r}. Expected one of {sorted(set(allowed_suffixes))}."
        )
    try:
        path.relative_to(_project_root())
    except ValueError as exc:
        raise ValueError(
            f"Refusing to load {purpose} from outside the project directory: {path}. "
            "Only trusted project-local code is allowed."
        ) from exc
    return path


def _load_module_from_path(module_name: str, module_path: Path):
    module_path = _ensure_trusted_project_file(
        module_path,
        allowed_suffixes=(".py",),
        purpose=f"Python module for {module_name}",
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_name} from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()



def _load_preprocessing_module(preprocessing_script_path: str | Path):
    preprocessing_script_path = _resolve_local_path(preprocessing_script_path)
    return _load_module_from_path("preprocessing", preprocessing_script_path)



def filter_closed_set_classes(
    metadata: pd.DataFrame,
    target_col: str = "composition_code",
    group_col: str = "specimen_group",
    min_groups_per_class: int = 3,
):
    group_counts = metadata.groupby(target_col)[group_col].nunique().sort_values(ascending=False)
    eligible_classes = group_counts[group_counts >= min_groups_per_class].index.tolist()
    mask = metadata[target_col].isin(eligible_classes).to_numpy()

    summary_df = group_counts.rename("n_groups").reset_index().assign(
        is_eligible=lambda d: d[target_col].isin(eligible_classes)
    )
    return mask, eligible_classes, summary_df



def _validate_closed_set_split(
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[bool, list[str], list[str]]:
    train_classes = set(pd.Series(y[train_idx]).unique().tolist())
    test_classes = set(pd.Series(y[test_idx]).unique().tolist())

    missing_in_train = sorted(test_classes - train_classes)
    missing_in_test = sorted(train_classes - test_classes)

    has_group_leakage = bool(set(groups[train_idx]).intersection(set(groups[test_idx])))
    is_valid = (not missing_in_train) and (not missing_in_test) and (not has_group_leakage)

    return is_valid, missing_in_train, missing_in_test



def _validate_group_label_consistency(y: np.ndarray, groups: np.ndarray) -> None:
    y = pd.Series(y).astype(str).to_numpy()
    groups = pd.Series(groups).astype(str).to_numpy()

    if len(y) != len(groups):
        raise ValueError("y and groups must have the same length for grouped splitting.")

    group_labels = pd.DataFrame({
        "specimen_group": groups,
        "composition_code": y,
    }).drop_duplicates()

    n_labels_per_group = group_labels.groupby("specimen_group")["composition_code"].nunique()
    bad_groups = n_labels_per_group[n_labels_per_group > 1]
    if not bad_groups.empty:
        details = (
            group_labels[group_labels["specimen_group"].isin(bad_groups.index)]
            .sort_values(["specimen_group", "composition_code"])
            .to_dict(orient="records")
        )
        raise ValueError(
            "Each specimen_group must map to exactly one composition_code before grouped splitting. "
            f"Found inconsistent labels for {len(bad_groups)} group(s): {details[:10]}"
        )



def make_group_safe_holdout_split(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 2,
    random_state: int = 42,
    max_seed_tries: int = 200,
):
    """
    Find one leakage-safe closed-set holdout split.

    Because some classes have very few specimen groups, a single
    StratifiedGroupKFold seed can place every group of a class into one fold.
    To avoid failing on an unlucky seed, try multiple shuffled seeds and
    return the first split where train and test contain the same class set.
    """
    _validate_group_label_consistency(y=y, groups=groups)

    dummy_X = np.zeros((len(y), 1), dtype=float)
    last_missing_train: list[str] = []
    last_missing_test: list[str] = []

    for seed in range(random_state, random_state + max_seed_tries):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        for train_idx, test_idx in splitter.split(dummy_X, y, groups):
            is_valid, missing_in_train, missing_in_test = _validate_closed_set_split(
                y=y,
                groups=groups,
                train_idx=train_idx,
                test_idx=test_idx,
            )
            if is_valid:
                return train_idx, test_idx

            last_missing_train = missing_in_train
            last_missing_test = missing_in_test

    raise ValueError(
        "Could not find a closed-set, leakage-safe grouped split after "
        f"{max_seed_tries} seed attempts. "
        f"Last missing in train: {last_missing_train}; "
        f"last missing in test: {last_missing_test}. "
        "Try increasing max_seed_tries or increasing min_groups_per_class."
    )



def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """GPU-accelerated Pipeline: GPUStandardScaler → GPURidgeClassifier (no PCA)."""
    return build_gpu_ridge_pipeline(alpha=alpha)



def evaluate_one_method(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    groups_test: np.ndarray,
    method_name: str,
) -> Dict:
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    model = build_ridge_pipeline(alpha=1.0)
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_test)

    return {
        "method": method_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_groups": int(pd.Series(groups_train).nunique()),
        "n_test_groups": int(pd.Series(groups_test).nunique()),
        "n_classes": int(pd.Series(y_train).nunique()),  # Fix #28: train-only, not union
        "accuracy": float(accuracy_score(y_test_encoded, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_encoded, y_pred)),
        "macro_f1": float(f1_score(y_test_encoded, y_pred, average="macro", zero_division=0)),
    }



def run_group_split_baseline(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    training_matrix_script_path: str | Path = "training_matrix.py",
    preprocessing_script_path: str | Path = "preprocessing.py",
    methods: Optional[List[str]] = None,
    min_groups_per_class: int = 3,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
):
    if methods is None:
        methods = ["raw", "snv", "snv_savgol1"]

    preprocessing = _load_preprocessing_module(preprocessing_script_path)
    bundle = preprocessing.build_preprocessing_bundle(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        training_matrix_script_path=training_matrix_script_path,
        aggregate=None,
        methods=None,
    )

    metadata = bundle["metadata"].copy()
    X_full = np.asarray(bundle["X"], dtype=float)
    y_full = bundle["y"].astype(str)
    groups_full = bundle["groups"].astype(str)
    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)

    mask, eligible_classes, class_group_summary = filter_closed_set_classes(
        metadata=metadata,
        target_col="composition_code",
        group_col="specimen_group",
        min_groups_per_class=min_groups_per_class,
    )

    metadata_f = metadata.loc[mask].reset_index(drop=True)
    X_f = X_full[mask]
    y_f = y_full[mask]
    groups_f = groups_full[mask]

    train_idx, test_idx = make_group_safe_holdout_split(
        y=y_f,
        groups=groups_f,
        n_splits=2,
        random_state=42,
    )

    results = []
    for method in methods:
        X_train, X_test, state = preprocessing.fit_and_apply_preprocessing(
            X_fit=X_f[train_idx],
            X_apply=X_f[test_idx],
            wavelengths=wavelengths,
            method=method,
        )
        row = evaluate_one_method(
            X_train=X_train,
            X_test=X_test,
            y_train=y_f[train_idx],
            y_test=y_f[test_idx],
            groups_train=groups_f[train_idx],
            groups_test=groups_f[test_idx],
            method_name=method,
        )
        row["preprocessing_is_stateful"] = preprocessing.requires_preprocessing_state(method)
        row["preprocessing_fit_scope"] = "train_only"
        results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        by=["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    split_summary = {
        "eligible_classes": eligible_classes,
        "n_eligible_classes": int(len(eligible_classes)),
        "n_total_classes": int(metadata["composition_code"].nunique()),
        "n_rows_after_filter": int(mask.sum()),
        "n_groups_after_filter": int(pd.Series(groups_f).nunique()),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_groups": int(pd.Series(groups_f[train_idx]).nunique()),
        "test_groups": int(pd.Series(groups_f[test_idx]).nunique()),
        "train_class_counts": pd.Series(y_f[train_idx]).value_counts().to_dict(),
        "test_class_counts": pd.Series(y_f[test_idx]).value_counts().to_dict(),
    }

    outputs = {
        "bundle": bundle,
        "metadata_filtered": metadata_f,
        "y_filtered": y_f,
        "groups_filtered": groups_f,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "class_group_summary": class_group_summary,
        "results_df": results_df,
        "split_summary": split_summary,
    }

    if output_dir is not None:
        save_group_split_outputs(output_dir, outputs)
        outputs["saved_output_dir"] = str(_resolve_local_path(output_dir))

    return outputs



def save_group_split_outputs(output_dir: str | Path, outputs: dict):
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data artefacts ────────────────────────────────────────────────────────
    outputs["results_df"].to_csv(output_dir / "baseline_results.csv", index=False)
    outputs["class_group_summary"].to_csv(output_dir / "class_group_summary.csv", index=False)
    outputs["metadata_filtered"].to_csv(output_dir / "filtered_metadata.csv", index=False)

    np.savez_compressed(
        output_dir / "group_split_indices.npz",
        train_idx=outputs["train_idx"],
        test_idx=outputs["test_idx"],
        y_filtered=outputs["y_filtered"],
        groups_filtered=outputs["groups_filtered"],
    )

    # ── Publication figures ───────────────────────────────────────────────────
    from publication_outputs import figure_preprocessing_comparison  # noqa: PLC0415
    figure_preprocessing_comparison(outputs["results_df"], output_dir)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    run_group_split_baseline(
        dataset_path=base_dir,
        loader_script_path=base_dir / "nir_loader.py",
        training_matrix_script_path=base_dir / "training_matrix.py",
        preprocessing_script_path=base_dir / "preprocessing.py",
        methods=["raw", "snv", "snv_savgol1"],
        min_groups_per_class=3,
        output_dir=base_dir / "outputs_group_split_baseline",
    )
    # All outputs saved to outputs_group_split_baseline/
