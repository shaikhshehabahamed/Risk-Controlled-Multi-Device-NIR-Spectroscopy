from __future__ import annotations
import logging

import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "final_model"
DEFAULT_GROUP_SPLIT_OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs_group_split_baseline"
DEFAULT_PCA_OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs_pca_baseline"
DEFAULT_BASELINE_METHODS = ("raw", "snv", "snv_savgol1")
DEFAULT_SELECTION_SORT_COLS = [
    "validation_selection_score",
    "validation_macro_f1",
    "validation_balanced_accuracy",
    "validation_accuracy",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_trusted_project_file(
    path_like: str | Path,
    *,
    allowed_suffixes: Sequence[str],
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
            "Only trusted project-local code and model artifacts are allowed."
        ) from exc
    return path


def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def _load_module(module_name: str, module_path: str | Path):
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


def _pca_sort_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    validation_cols = ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"]
    if all(col in df.columns for col in validation_cols):
        return validation_cols, "validation"
    legacy_cols = ["macro_f1", "balanced_accuracy", "accuracy"]
    if all(col in df.columns for col in legacy_cols):
        return legacy_cols, "legacy_test_ranked"
    raise ValueError(
        "pca_results.csv is missing the expected ranking columns. Expected either "
        f"{validation_cols} or {legacy_cols}."
    )


def _load_required_saved_split(
    group_split_outputs_dir: str | Path,
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


def _validate_reference_wavelength_axis(wavelengths: np.ndarray) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("Reference wavelength axis must be a non-empty 1D array.")
    if not np.isfinite(wavelengths).all():
        raise ValueError("Reference wavelength axis contains non-finite values.")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError(
            "Reference wavelength axis must be strictly increasing and contain no duplicates."
        )
    return wavelengths


def _compute_selection_score(
    macro_f1: float,
    balanced_accuracy: float,
) -> float:
    return float((float(macro_f1) + float(balanced_accuracy)) / 2.0)


def _sort_candidate_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in DEFAULT_SELECTION_SORT_COLS if col in df.columns]
    if not sort_cols:
        raise ValueError("No candidate ranking columns are available.")
    return df.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)


def _load_baseline_methods(
    group_split_outputs_dir: str | Path,
) -> list[str]:
    baseline_path = _resolve_local_path(group_split_outputs_dir) / "baseline_results.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        if "method" in df.columns:
            methods = [str(m).strip().lower() for m in df["method"].tolist() if str(m).strip()]
            methods = list(dict.fromkeys(methods))
            if methods:
                return methods
    return list(DEFAULT_BASELINE_METHODS)


def _evaluate_validation_candidate(
    *,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    wavelengths: np.ndarray,
    preprocessing_module,
    group_split_module,
    pca_module,
    preprocessing_method: str,
    pca_components: Optional[int],
    alpha: float,
) -> dict:
    preprocessing_method = str(preprocessing_method).strip().lower()
    X_train_proc, X_val_proc, _ = preprocessing_module.fit_and_apply_preprocessing(
        X_fit=X_train_raw,
        X_apply=X_val_raw,
        wavelengths=wavelengths,
        method=preprocessing_method,
    )

    if pca_components is None:
        model = group_split_module.build_ridge_pipeline(alpha=alpha)
        candidate_family = "ridge"
        candidate_name = f"{preprocessing_method}_ridge"
    else:
        model = pca_module.build_snv_pca_ridge_pipeline(n_components=int(pca_components), alpha=alpha)
        candidate_family = "pca_ridge"
        candidate_name = f"{preprocessing_method}_pca_{int(pca_components)}"

    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_val_proc)

    accuracy = float(accuracy_score(y_val, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

    return {
        "candidate_family": candidate_family,
        "candidate_name": candidate_name,
        "preprocessing_method": preprocessing_method,
        "pca_enabled": bool(pca_components is not None),
        "selected_n_components": None if pca_components is None else int(pca_components),
        "validation_accuracy": accuracy,
        "validation_balanced_accuracy": balanced_accuracy,
        "validation_macro_f1": macro_f1,
        "validation_selection_score": _compute_selection_score(macro_f1, balanced_accuracy),
        "source": "recomputed_inner_validation",
    }


def _load_best_integer_pca_candidates(
    pca_outputs_dir: str | Path = DEFAULT_PCA_OUTPUTS_DIR,
) -> pd.DataFrame:
    pca_outputs_dir = _resolve_local_path(pca_outputs_dir)
    pca_results_path = pca_outputs_dir / "pca_results.csv"
    if not pca_results_path.exists():
        raise FileNotFoundError(
            f"Required PCA results file not found: {pca_results_path}. "
            "Run pca_baseline.py first."
        )

    df = pd.read_csv(pca_results_path)
    if df.empty:
        raise ValueError(f"PCA results file is empty: {pca_results_path}. Rerun pca_baseline.py.")

    df = df.copy()
    if "preprocessing_method" not in df.columns:
        df["preprocessing_method"] = "snv"
    df["preprocessing_method"] = df["preprocessing_method"].astype(str).str.strip().str.lower()

    def _is_integer_like(value) -> bool:
        return float(value).is_integer()

    df = df[df["n_components_requested"].apply(_is_integer_like)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No integer PCA setting was found in pca_results.csv. "
            "Champion freezing only supports explicit integer PCA winners."
        )

    sort_cols, ranking_source = _pca_sort_columns(df)
    if ranking_source == "legacy_test_ranked":
        # Backfill validation columns so overall selection logic has a consistent schema,
        # but mark the provenance clearly because this is not the preferred leakage-safe path.
        df["validation_macro_f1"] = df["macro_f1"]
        df["validation_balanced_accuracy"] = df["balanced_accuracy"]
        df["validation_accuracy"] = df["accuracy"]

    df["validation_selection_score"] = (
        df["validation_macro_f1"].astype(float) + df["validation_balanced_accuracy"].astype(float)
    ) / 2.0

    candidate_rows: list[dict] = []
    for preprocessing_method, sub_df in df.groupby("preprocessing_method", sort=False):
        ranked = _sort_candidate_df(sub_df)
        best_row = ranked.iloc[0].to_dict()
        selected_n = best_row.get("n_components_used")
        if pd.isna(selected_n):
            selected_n = int(float(best_row["n_components_requested"]))
        else:
            selected_n = int(selected_n)

        candidate_rows.append({
            **best_row,
            "candidate_family": "pca_ridge",
            "candidate_name": f"{preprocessing_method}_pca_{selected_n}",
            "pca_enabled": True,
            "selected_n_components": selected_n,
            "source": str(pca_results_path.name),
            "ranking_source": ranking_source,
        })

    candidates_df = pd.DataFrame(candidate_rows)
    if candidates_df.empty:
        raise ValueError("Could not derive any PCA champion candidates from pca_results.csv.")
    return _sort_candidate_df(candidates_df)


def _select_champion_candidate(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    wavelengths: np.ndarray,
    train_idx: np.ndarray,
    preprocessing_module,
    group_split_module,
    pca_module,
    group_split_outputs_dir: str | Path,
    pca_outputs_dir: str | Path,
    alpha: float,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    selection_train_idx, selection_val_idx = pca_module._make_inner_validation_split(
        group_split_module=group_split_module,
        y=y,
        groups=groups,
        train_idx=train_idx,
        random_state=142,
        max_seed_tries=200,
    )

    X_sel_train_raw = X[selection_train_idx]
    X_sel_val_raw = X[selection_val_idx]
    y_sel_train = y[selection_train_idx]
    y_sel_val = y[selection_val_idx]

    candidate_rows: list[dict] = []
    for method in _load_baseline_methods(group_split_outputs_dir=group_split_outputs_dir):
        candidate_rows.append(
            _evaluate_validation_candidate(
                X_train_raw=X_sel_train_raw,
                y_train=y_sel_train,
                X_val_raw=X_sel_val_raw,
                y_val=y_sel_val,
                wavelengths=wavelengths,
                preprocessing_module=preprocessing_module,
                group_split_module=group_split_module,
                pca_module=pca_module,
                preprocessing_method=method,
                pca_components=None,
                alpha=alpha,
            )
        )

    pca_candidates_df = _load_best_integer_pca_candidates(pca_outputs_dir=pca_outputs_dir)
    candidate_rows.extend(pca_candidates_df.to_dict(orient="records"))

    # Fix #40: pca_candidates_df metrics come from a previous run of run_pca_experiment
    # which used _make_inner_validation_split(random_state=142) on the same outer
    # train_idx.  Fresh candidates above also use random_state=142 on the same inputs,
    # so the inner split is deterministically identical between runs as long as the
    # filtered dataset (y, groups, train_idx) is unchanged — which is enforced by the
    # np.array_equal consistency checks below.
    leaderboard_df = _sort_candidate_df(pd.DataFrame(candidate_rows))
    if leaderboard_df.empty:
        raise ValueError("No champion candidates were available for model selection.")

    winner = leaderboard_df.iloc[0].to_dict()
    return winner, leaderboard_df, selection_train_idx, selection_val_idx


def _fit_model_for_candidate(
    *,
    X_train_raw: np.ndarray,
    X_apply_raw: np.ndarray,
    y_train: np.ndarray,
    wavelengths: np.ndarray,
    preprocessing_module,
    group_split_module,
    pca_module,
    preprocessing_method: str,
    pca_components: Optional[int],
    alpha: float,
):
    preprocessing_state = preprocessing_module.fit_preprocessing_state(
        X_train_raw,
        wavelengths,
        method=preprocessing_method,
    )
    X_train = preprocessing_module.preprocess_spectra(
        X_train_raw,
        wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )
    X_apply = preprocessing_module.preprocess_spectra(
        X_apply_raw,
        wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )

    if pca_components is None:
        model = group_split_module.build_ridge_pipeline(alpha=alpha)
        pca_info = {
            "enabled": False,
            "requested_n_components": 0,
            "used_n_components": 0,
            "explained_variance_ratio_sum": 0.0,
            "eval_split_used_n_components": 0,
            "eval_split_explained_variance_ratio_sum": 0.0,
        }
        pca_df = None
    else:
        model = pca_module.build_snv_pca_ridge_pipeline(n_components=int(pca_components), alpha=alpha)
        pca_df = None
        pca_info = None

    model.fit(X_train, y_train)

    if pca_components is not None:
        pca_obj = model.named_steps["pca"]
        pca_info = {
            "enabled": True,
            "requested_n_components": int(pca_components),
            "used_n_components": int(pca_obj.n_components_),
            "explained_variance_ratio_sum": float(np.sum(pca_obj.explained_variance_ratio_)),
            "eval_split_used_n_components": int(pca_obj.n_components_),
            "eval_split_explained_variance_ratio_sum": float(np.sum(pca_obj.explained_variance_ratio_)),
        }
        pca_df = pd.DataFrame({
            "component": np.arange(1, len(pca_obj.explained_variance_ratio_) + 1, dtype=int),
            "explained_variance_ratio": pca_obj.explained_variance_ratio_,
        })
        pca_df["cumulative_explained_variance"] = pca_df["explained_variance_ratio"].cumsum()

    return model, X_apply, preprocessing_state, pca_info, pca_df


def run_freeze_champion_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    loader_script_path: str | Path = "nir_loader.py",
    training_matrix_script_path: str | Path = "training_matrix.py",
    preprocessing_script_path: str | Path = "preprocessing.py",
    group_split_script_path: str | Path = "group_split_baseline.py",
    pca_script_path: str | Path = "pca_baseline.py",
    group_split_outputs_dir: str | Path = DEFAULT_GROUP_SPLIT_OUTPUTS_DIR,
    pca_outputs_dir: str | Path = DEFAULT_PCA_OUTPUTS_DIR,
    min_groups_per_class: int = 3,
    champion_preprocessing: str = "auto",
    champion_pca_components: Optional[int] = None,
    alpha: float = 1.0,
    instruments: Optional[Sequence[str]] = None,
    measurement: str = "Reflectance",
) -> Dict:
    """
    Freeze the final champion model using leakage-safe inner validation on the outer
    training partition.

    Selection procedure:
      1. Reuse the saved outer grouped split from group_split_baseline.py.
      2. Derive a grouped inner train/validation split from the outer-train partition.
      3. Compare no-PCA baseline methods and the best integer PCA candidate(s) using
         a class-balanced validation objective.
      4. Retrain the winning pipeline on the full outer-train partition.
      5. Evaluate exactly once on the untouched outer-test partition.

    The default selection objective is:
        validation_selection_score = 0.5 * (validation_macro_f1 + validation_balanced_accuracy)
    with validation_macro_f1, validation_balanced_accuracy, then validation_accuracy as tie-breakers.
    """
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_split = _load_module("group_split_baseline", group_split_script_path)
    pca_module = _load_module("pca_baseline", pca_script_path)
    preprocessing = _load_module("preprocessing", preprocessing_script_path)

    requested_champion_preprocessing = str(champion_preprocessing).strip().lower()
    if not requested_champion_preprocessing:
        raise ValueError("champion_preprocessing must be a non-empty string.")

    if instruments is None:
        instruments = ["Benchtop_N"]
    else:
        instruments = list(instruments)
    bundle = preprocessing.build_preprocessing_bundle(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        training_matrix_script_path=training_matrix_script_path,
        aggregate=None,
        methods=None,
        instruments=instruments,
        measurement=measurement,
    )

    metadata = bundle["metadata"].copy()
    wavelengths = _validate_reference_wavelength_axis(bundle["wavelengths"])

    mask, eligible_classes, class_group_summary = group_split.filter_closed_set_classes(
        metadata=metadata,
        target_col="composition_code",
        group_col="specimen_group",
        min_groups_per_class=min_groups_per_class,
    )

    X = np.asarray(bundle["X"], dtype=float)[mask]
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

    winner, leaderboard_df, selection_train_idx, selection_val_idx = _select_champion_candidate(
        X=X,
        y=y,
        groups=groups,
        wavelengths=wavelengths,
        train_idx=train_idx,
        preprocessing_module=preprocessing,
        group_split_module=group_split,
        pca_module=pca_module,
        group_split_outputs_dir=group_split_outputs_dir,
        pca_outputs_dir=pca_outputs_dir,
        alpha=alpha,
    )

    if requested_champion_preprocessing != "auto":
        filtered = leaderboard_df[
            leaderboard_df["preprocessing_method"].astype(str).str.lower() == requested_champion_preprocessing
        ].reset_index(drop=True)
        if filtered.empty:
            raise ValueError(
                "No champion candidates were available for the requested champion_preprocessing="
                f"{requested_champion_preprocessing!r}."
            )
        winner = _sort_candidate_df(filtered).iloc[0].to_dict()

    if champion_pca_components is not None:
        winner["pca_enabled"] = True
        winner["selected_n_components"] = int(champion_pca_components)
        winner["candidate_family"] = "pca_ridge"
        winner["candidate_name"] = f"{winner['preprocessing_method']}_pca_{int(champion_pca_components)}"

    champion_preprocessing_selected = str(winner["preprocessing_method"]).strip().lower()
    selected_n_components = winner.get("selected_n_components")
    if pd.isna(selected_n_components):
        selected_n_components = None
    elif selected_n_components is not None:
        selected_n_components = int(selected_n_components)

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    max_valid_pca_components = min(int(X_train_raw.shape[0]), int(X_train_raw.shape[1]))
    if selected_n_components is not None and not (1 <= int(selected_n_components) <= max_valid_pca_components):
        raise ValueError(
            "Selected champion PCA components are out of range for the filtered training split. "
            f"Requested {selected_n_components}, valid range is 1..{max_valid_pca_components}."
        )

    model_eval, X_test_eval, eval_preprocessing_state, eval_pca_info, eval_pca_df = _fit_model_for_candidate(
        X_train_raw=X_train_raw,
        X_apply_raw=X_test_raw,
        y_train=y_train,
        wavelengths=wavelengths,
        preprocessing_module=preprocessing,
        group_split_module=group_split,
        pca_module=pca_module,
        preprocessing_method=champion_preprocessing_selected,
        pca_components=selected_n_components,
        alpha=alpha,
    )
    y_pred = model_eval.predict(X_test_eval)

    class_labels = sorted(pd.Series(y).unique().tolist())
    accuracy = float(accuracy_score(y_test, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    conf = confusion_matrix(y_test, y_pred, labels=class_labels)
    conf_df = pd.DataFrame(conf, index=class_labels, columns=class_labels)

    report = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    test_metadata = metadata_filtered.iloc[test_idx].reset_index(drop=True).copy()
    test_metadata["true_label"] = y_test
    test_metadata["pred_label"] = y_pred
    test_metadata["is_correct"] = (test_metadata["true_label"] == test_metadata["pred_label"]).astype(int)

    if hasattr(model_eval, "decision_function"):
        scores = model_eval.decision_function(X_test_eval)
        if np.ndim(scores) == 1:
            test_metadata["decision_score"] = scores
        else:
            test_metadata["decision_score_max"] = np.max(scores, axis=1)

    label_map_df = pd.DataFrame({
        "class_index": np.arange(len(class_labels), dtype=int),
        "class_label": class_labels,
    })
    wavelengths_df = pd.DataFrame({
        "wavelength_axis": wavelengths,
    })

    joblib.dump(model_eval, output_dir / "best_model_eval_split.joblib")

    # Fix #15: the full-data refit is for artifact export only; the transformed
    # X_apply output is discarded (_).  Pass X_train_raw=X_apply_raw=X to avoid
    # a misleading asymmetry but explicitly discard the second return value.
    model_full, _, full_preprocessing_state, full_pca_info, full_pca_df = _fit_model_for_candidate(
        X_train_raw=X,
        X_apply_raw=X,
        y_train=y,
        wavelengths=wavelengths,
        preprocessing_module=preprocessing,
        group_split_module=group_split,
        pca_module=pca_module,
        preprocessing_method=champion_preprocessing_selected,
        pca_components=selected_n_components,
        alpha=alpha,
    )
    joblib.dump(model_full, output_dir / "best_model_full_data.joblib")

    label_map_df.to_csv(output_dir / "label_map.csv", index=False)
    conf_df.to_csv(output_dir / "confusion_matrix.csv")
    report_df.to_csv(output_dir / "classification_report.csv")
    test_metadata.to_csv(output_dir / "test_predictions.csv", index=False)
    wavelengths_df.to_csv(output_dir / "reference_wavelengths.csv", index=False)
    class_group_summary.to_csv(output_dir / "class_group_summary.csv", index=False)
    metadata_filtered.to_csv(output_dir / "filtered_metadata.csv", index=False)
    leaderboard_df.to_csv(output_dir / "selection_leaderboard.csv", index=False)

    if full_pca_df is not None:
        full_pca_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    else:
        pca_path = output_dir / "pca_explained_variance.csv"
        if pca_path.exists():
            pca_path.unlink()

    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_idx=train_idx,
        test_idx=test_idx,
        y_filtered=y,
        groups_filtered=groups,
        selection_train_idx=selection_train_idx,
        selection_val_idx=selection_val_idx,
    )

    instruments_used = sorted(metadata_filtered["instrument"].dropna().astype(str).unique().tolist())
    measurements_used = sorted(metadata_filtered["measurement"].dropna().astype(str).unique().tolist())
    dataset_subset = {
        "instrument": instruments_used[0] if len(instruments_used) == 1 else instruments_used,
        "measurement": measurements_used[0] if len(measurements_used) == 1 else measurements_used,
    }

    champion_pipeline = (
        f"{champion_preprocessing_selected.upper()} -> PCA({selected_n_components}) -> RidgeClassifier"
        if selected_n_components is not None
        else f"{champion_preprocessing_selected.upper()} -> RidgeClassifier"
    )

    model_config = {
        "champion_pipeline": champion_pipeline,
        "dataset_subset": dataset_subset,
        "class_filter": {
            "min_groups_per_class": min_groups_per_class,
            "n_eligible_classes": int(len(eligible_classes)),
            "eligible_classes": eligible_classes,
        },
        "preprocessing": champion_preprocessing_selected,
        "preprocessing_state": preprocessing.serialize_preprocessing_state(full_preprocessing_state),
        "model": {
            "name": "RidgeClassifier",
            "alpha": alpha,
            "class_weight": "balanced",
        },
        "pca": {
            **(full_pca_info or {}),
            "selection": winner,
        },
        "selection": {
            "strategy": "inner_grouped_validation",
            "objective_name": "balanced_composite",
            "objective_formula": "0.5 * (validation_macro_f1 + validation_balanced_accuracy)",
            "sort_priority": DEFAULT_SELECTION_SORT_COLS,
            "winner": winner,
            "leaderboard_artifact": "selection_leaderboard.csv",
        },
        "split": {
            "source": split_source,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_groups": int(pd.Series(groups[train_idx]).nunique()),
            "n_test_groups": int(pd.Series(groups[test_idx]).nunique()),
            "n_selection_train": int(len(selection_train_idx)),
            "n_selection_val": int(len(selection_val_idx)),
        },
        "test_metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
        },
        "artifacts": {
            "eval_model": "best_model_eval_split.joblib",
            "full_data_model": "best_model_full_data.joblib",
            "label_map": "label_map.csv",
            "confusion_matrix": "confusion_matrix.csv",
            "classification_report": "classification_report.csv",
            "test_predictions": "test_predictions.csv",
            "reference_wavelengths": "reference_wavelengths.csv",
            "split_indices": "split_indices.npz",
            "selection_leaderboard": "selection_leaderboard.csv",
            **({"pca_explained_variance": "pca_explained_variance.csv"} if full_pca_df is not None else {}),
        },
    }

    # Fix #49: the winner dict from the merged leaderboard can contain float("nan")
    # for columns that are absent in some candidate sources.  Python's json.dump
    # writes NaN as the literal "NaN", which is invalid per RFC 8259.  Sanitize
    # all NaN values to None (JSON null) before writing.
    def _sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(model_config), f, indent=2)

    summary = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "n_test_samples": int(len(test_idx)),
        "n_test_groups": int(pd.Series(groups[test_idx]).nunique()),
        "champion_pipeline": champion_pipeline,
        "champion_preprocessing": champion_preprocessing_selected,
        "champion_pca_components": selected_n_components,
        "selection_winner": winner,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)

    from publication_outputs import (
        figure_confusion_matrix,
        figure_classification_report,
        figure_pca_scree,
    )
    figure_confusion_matrix(conf_df, output_dir)
    figure_classification_report(report_df, output_dir)
    if full_pca_df is not None:
        figure_pca_scree(full_pca_df, output_dir, champion_n=int(full_pca_info["used_n_components"]))

    return {
        "model_config": model_config,
        "summary": summary,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "selection_winner": winner,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    run_freeze_champion_model()
    # All outputs saved to outputs_freeze_champion_model/
