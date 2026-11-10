from __future__ import annotations
import logging

import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from gpu_models import build_gpu_pca_ridge_pipeline


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_GROUP_SPLIT_OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs_group_split_baseline"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_pca_baseline"


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



def _load_group_split_module(group_split_script_path: str | Path):
    group_split_script_path = _resolve_local_path(group_split_script_path)
    return _load_module_from_path("group_split_baseline", group_split_script_path)



def _load_preprocessing_module(preprocessing_script_path: str | Path):
    preprocessing_script_path = _resolve_local_path(preprocessing_script_path)
    return _load_module_from_path("preprocessing", preprocessing_script_path)



def _load_required_saved_split(
    group_split_outputs_dir: str | Path = DEFAULT_GROUP_SPLIT_OUTPUTS_DIR,
) -> dict:
    group_split_outputs_dir = _resolve_local_path(group_split_outputs_dir)
    split_path = group_split_outputs_dir / "group_split_indices.npz"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Required grouped split file not found: {split_path}. "
            "Run group_split_baseline.py first."
        )

    data = np.load(split_path, allow_pickle=True)
    return {
        "train_idx": data["train_idx"],
        "test_idx": data["test_idx"],
        "y_filtered": data["y_filtered"].astype(str),
        "groups_filtered": data["groups_filtered"].astype(str),
        "source_path": str(split_path),
    }



def _resolve_requested_pca_components(X_train: np.ndarray, n_components: Union[int, float]) -> Union[int, float]:
    if isinstance(n_components, (float, np.floating)):
        value = float(n_components)
        if 0.0 < value < 1.0 and not value.is_integer():
            return value
        n_components = int(round(value))
    return max(1, min(int(n_components), int(min(X_train.shape[0], X_train.shape[1]))))



def build_snv_pca_ridge_pipeline(
    n_components: Union[int, float] = 50,
    alpha: float = 1.0,
) -> Pipeline:
    """GPU-accelerated Pipeline: GPUStandardScaler → GPUPCA → GPURidgeClassifier."""
    return build_gpu_pca_ridge_pipeline(n_components=n_components, alpha=alpha)



def evaluate_single_pca_setting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_components: Union[int, float],
    alpha: float = 1.0,
) -> Dict:
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_eval_encoded = label_encoder.transform(y_eval)

    resolved_n_components = _resolve_requested_pca_components(X_train, n_components)
    model = build_snv_pca_ridge_pipeline(n_components=resolved_n_components, alpha=alpha)
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_eval)
    fitted_pca = model.named_steps["pca"]

    return {
        "n_components_requested": n_components,
        "n_components_used": int(fitted_pca.n_components_),
        "explained_variance_ratio_sum": float(np.sum(fitted_pca.explained_variance_ratio_)),
        "validation_accuracy": float(accuracy_score(y_eval_encoded, y_pred)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_eval_encoded, y_pred)),
        "validation_macro_f1": float(f1_score(y_eval_encoded, y_pred, average="macro", zero_division=0)),
        "selection_train_size": int(len(y_train)),
        "selection_val_size": int(len(y_eval)),
        "n_classes": int(pd.Series(y_selection_train).nunique()),  # Fix #28: train-only
    }



def _make_inner_validation_split(
    group_split_module,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    random_state: int = 142,
    max_seed_tries: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    train_y = y[train_idx]
    train_groups = groups[train_idx]
    required_classes = set(pd.Series(train_y).astype(str).unique().tolist())

    # Classes with only one unique specimen group in the outer-train partition cannot
    # appear on both sides of a grouped split — their single group must land entirely
    # in one fold.  We therefore only require such classes in inner-train (the model
    # must see every class during training) but do NOT require them in inner-val.
    groups_per_class = (
        pd.DataFrame({"group": pd.Series(train_groups).astype(str),
                      "cls":   pd.Series(train_y).astype(str)})
        .drop_duplicates()
        .groupby("cls")["group"]
        .nunique()
    )
    # Classes that have >=2 groups in outer-train can realistically appear in both halves.
    splittable_classes = set(groups_per_class[groups_per_class >= 2].index.tolist())

    def _validate_candidate(inner_train_rel: np.ndarray, inner_val_rel: np.ndarray):
        inner_train_classes = set(pd.Series(train_y[inner_train_rel]).astype(str).unique().tolist())
        inner_val_classes = set(pd.Series(train_y[inner_val_rel]).astype(str).unique().tolist())
        # inner-train must contain every class (model must be trained on the full label set).
        # inner-val must contain every class that physically can appear there (splittable only).
        missing_in_train = sorted(required_classes - inner_train_classes)
        missing_in_val = sorted(splittable_classes - inner_val_classes)
        if missing_in_train or missing_in_val:
            return None, missing_in_train, missing_in_val
        abs_train = train_idx[inner_train_rel]
        abs_val = train_idx[inner_val_rel]
        return (abs_train, abs_val), [], []

    inner_train_rel, inner_val_rel = group_split_module.make_group_safe_holdout_split(
        y=train_y,
        groups=train_groups,
        n_splits=2,
        random_state=random_state,
        max_seed_tries=max_seed_tries,
    )
    candidate, _, _ = _validate_candidate(inner_train_rel, inner_val_rel)
    if candidate is not None:
        return candidate

    # Fix #29: the preferred grouped holdout split failed to produce a class-complete
    # split.  Warn before falling back to StratifiedGroupKFold so the user knows
    # the split strategy changed rather than failing silently.
    import warnings as _warnings
    _warnings.warn(
        "_make_inner_validation_split: make_group_safe_holdout_split did not produce "
        "a class-complete inner split; falling back to StratifiedGroupKFold with "
        f"up to {max_seed_tries} seeds.  Consider increasing min_groups_per_class.",
        UserWarning,
        stacklevel=3,
    )

    dummy_X = np.zeros((len(train_idx), 1), dtype=float)
    best_missing: tuple[int, int] | None = None
    best_details: tuple[list[str], list[str]] | None = None

    for seed in range(random_state, random_state + max_seed_tries):
        splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        for inner_train_rel, inner_val_rel in splitter.split(dummy_X, train_y, train_groups):
            candidate, missing_in_train, missing_in_val = _validate_candidate(inner_train_rel, inner_val_rel)
            if candidate is not None:
                return candidate
            missing_key = (len(missing_in_train), len(missing_in_val))
            if best_missing is None or missing_key < best_missing:
                best_missing = missing_key
                best_details = (missing_in_train, missing_in_val)

    detail_suffix = ""
    if best_details is not None:
        detail_suffix = (
            f" Best candidate still missed train classes {best_details[0]} and validation classes {best_details[1]}."
        )
    raise ValueError(
        "Could not derive a leakage-safe grouped inner validation split where inner-train contains all classes "
        "and inner-validation contains all classes that have >= 2 specimen groups in the outer-train partition. "
        "Try increasing max_seed_tries or increasing min_groups_per_class."
        + detail_suffix
    )



def run_pca_experiment(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    group_split_script_path: str | Path = "group_split_baseline.py",
    loader_script_path: str | Path = "nir_loader.py",
    training_matrix_script_path: str | Path = "training_matrix.py",
    preprocessing_script_path: str | Path = "preprocessing.py",
    group_split_outputs_dir: str | Path = DEFAULT_GROUP_SPLIT_OUTPUTS_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    min_groups_per_class: int = 3,
    pca_grid: Optional[List[Union[int, float]]] = None,
    preprocessing_method: str = "snv",
):
    """
    Leakage-safe PCA experiment:
    - rebuild the filtered raw matrix from the dataset
    - load the required saved outer grouped train/test split
    - derive a grouped validation split from the saved outer-train partition only
    - fit preprocessing state on the inner-train partition only
    - rank PCA settings on validation metrics only

    The outer test split is reserved for freeze_champion_model.py and is not used for
    PCA model selection here.
    """
    if pca_grid is None:
        pca_grid = [10, 20, 30, 50, 75, 100, 0.95]

    preprocessing_method = str(preprocessing_method).strip().lower()
    if not preprocessing_method:
        raise ValueError("preprocessing_method must be a non-empty string.")
    if int(min_groups_per_class) < 3:
        raise ValueError(
            "PCA model selection with a grouped inner validation split requires min_groups_per_class >= 3. "
            f"Got {min_groups_per_class}."
        )

    group_split = _load_group_split_module(group_split_script_path)
    preprocessing = _load_preprocessing_module(preprocessing_script_path)

    bundle = preprocessing.build_preprocessing_bundle(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        training_matrix_script_path=training_matrix_script_path,
        aggregate=None,
        methods=None,
    )

    metadata = bundle["metadata"].copy()
    X_raw = np.asarray(bundle["X"], dtype=float)
    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)
    mask, eligible_classes, class_group_summary = group_split.filter_closed_set_classes(
        metadata=metadata,
        target_col="composition_code",
        group_col="specimen_group",
        min_groups_per_class=min_groups_per_class,
    )

    X_filtered = X_raw[mask]
    y = bundle["y"][mask].astype(str)
    groups = bundle["groups"][mask].astype(str)
    metadata_filtered = metadata.loc[mask].reset_index(drop=True)

    saved_split = _load_required_saved_split(group_split_outputs_dir=group_split_outputs_dir)

    if not np.array_equal(saved_split["y_filtered"], y):
        raise ValueError(
            "Saved split y_filtered does not match the current filtered dataset ordering. "
            "Delete outputs_group_split_baseline and rerun group_split_baseline.py."
        )
    if not np.array_equal(saved_split["groups_filtered"], groups):
        raise ValueError(
            "Saved split groups_filtered does not match the current filtered dataset ordering. "
            "Delete outputs_group_split_baseline and rerun group_split_baseline.py."
        )

    train_idx = saved_split["train_idx"]
    test_idx = saved_split["test_idx"]
    split_source = saved_split["source_path"]
    selection_train_idx, selection_val_idx = _make_inner_validation_split(
        group_split_module=group_split,
        y=y,
        groups=groups,
        train_idx=train_idx,
        random_state=142,
        max_seed_tries=200,
    )

    X_train, X_val, preprocessing_state = preprocessing.fit_and_apply_preprocessing(
        X_fit=X_filtered[selection_train_idx],
        X_apply=X_filtered[selection_val_idx],
        wavelengths=wavelengths,
        method=preprocessing_method,
    )

    rows = []
    for n_components in pca_grid:
        row = evaluate_single_pca_setting(
            X_train=X_train,
            y_train=y[selection_train_idx],
            X_eval=X_val,
            y_eval=y[selection_val_idx],
            n_components=n_components,
            alpha=1.0,
        )
        row["preprocessing_method"] = preprocessing_method
        row["selection_train_groups"] = int(pd.Series(groups[selection_train_idx]).nunique())
        row["selection_val_groups"] = int(pd.Series(groups[selection_val_idx]).nunique())
        row["selection_val_size"] = int(len(selection_val_idx))
        row["outer_train_size"] = int(len(train_idx))
        row["outer_test_size"] = int(len(test_idx))
        row["outer_train_groups"] = int(pd.Series(groups[train_idx]).nunique())
        row["outer_test_groups"] = int(pd.Series(groups[test_idx]).nunique())
        row["preprocessing_is_stateful"] = preprocessing.requires_preprocessing_state(preprocessing_method)
        row["preprocessing_fit_scope"] = "selection_train_only"
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    summary = {
        "preprocessing_method": preprocessing_method,
        "eligible_classes": eligible_classes,
        "n_eligible_classes": int(len(eligible_classes)),
        "n_samples": int(len(y)),
        "n_groups": int(pd.Series(groups).nunique()),
        "outer_train_size": int(len(train_idx)),
        "outer_test_size": int(len(test_idx)),
        "selection_train_size": int(len(selection_train_idx)),
        "selection_val_size": int(len(selection_val_idx)),
        "split_source": split_source,
        "best_setting": results_df.iloc[0].to_dict(),
    }

    outputs = {
        "X_filtered_raw": X_filtered,
        "wavelengths": wavelengths,
        "y_filtered": y,
        "groups_filtered": groups,
        "metadata_filtered": metadata_filtered,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "selection_train_idx": selection_train_idx,
        "selection_val_idx": selection_val_idx,
        "class_group_summary": class_group_summary,
        "results_df": results_df,
        "summary": summary,
    }

    if output_dir is not None:
        save_pca_outputs(output_dir, outputs)
        outputs["saved_output_dir"] = str(_resolve_local_path(output_dir))

    return outputs



def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_pca_outputs(output_dir: str | Path, outputs: dict):
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data artefacts ────────────────────────────────────────────────────────
    outputs["results_df"].to_csv(output_dir / "pca_results.csv", index=False)
    outputs["class_group_summary"].to_csv(output_dir / "class_group_summary.csv", index=False)
    outputs["metadata_filtered"].to_csv(output_dir / "filtered_metadata.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(outputs["summary"], indent=2, default=_json_default),
        encoding="utf-8",
    )

    np.savez_compressed(
        output_dir / "pca_arrays_and_split.npz",
        X_filtered_raw=outputs["X_filtered_raw"],
        wavelengths=outputs["wavelengths"],
        y_filtered=outputs["y_filtered"],
        groups_filtered=outputs["groups_filtered"],
        train_idx=outputs["train_idx"],
        test_idx=outputs["test_idx"],
        selection_train_idx=outputs["selection_train_idx"],
        selection_val_idx=outputs["selection_val_idx"],
    )

    # ── Publication figures ───────────────────────────────────────────────────
    from publication_outputs import figure_pca_selection  # noqa: PLC0415
    figure_pca_selection(outputs["results_df"], output_dir)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    run_pca_experiment(
        dataset_path=base_dir,
        group_split_script_path=base_dir / "group_split_baseline.py",
        loader_script_path=base_dir / "nir_loader.py",
        training_matrix_script_path=base_dir / "training_matrix.py",
        preprocessing_script_path=base_dir / "preprocessing.py",
        group_split_outputs_dir=base_dir / "outputs_group_split_baseline",
        output_dir=base_dir / "outputs_pca_baseline",
        min_groups_per_class=3,
        pca_grid=[20, 50, 0.95],
        preprocessing_method="snv",
    )
    # All outputs saved to outputs_pca_baseline/
