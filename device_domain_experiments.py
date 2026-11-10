from __future__ import annotations
import logging

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import preprocessing
import training_matrix
from domain_generalization import (
    DEFAULT_DG_CONFIG,
    config_to_dict,
    fit_domain_generalized_calibrated_model,
)
from reliability import fit_calibrated_model, make_nested_group_splits, predict_with_uncertainty


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_DEVICES = ["Benchtop_N", "Handheld_F", "Handheld_S"]
DEFAULT_TRAINING_STRATEGIES = ("baseline", "domain_generalized")


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
        details = (
            normalized[normalized["specimen_group"].isin(bad_groups.index)]
            .sort_values(["specimen_group", "composition_code"])
            .to_dict(orient="records")
        )
        raise ValueError(
            "Each specimen_group must map to exactly one composition_code before grouped splitting. "
            f"Found inconsistent labels for {len(bad_groups)} group(s): {details[:10]}"
        )

    return normalized


def _iter_group_holdout_candidates(
    group_labels: pd.DataFrame,
    test_size: float,
    random_state: int,
    max_seed_tries: int,
):
    normalized = _normalize_group_labels(group_labels)
    if normalized.empty:
        return

    y = normalized["composition_code"].to_numpy(dtype=object)
    dummy = np.zeros((len(normalized), 1), dtype=float)

    for seed in range(int(random_state), int(random_state) + int(max_seed_tries)):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(dummy, y))

        train_labels = normalized.iloc[train_idx].reset_index(drop=True)
        test_labels = normalized.iloc[test_idx].reset_index(drop=True)
        train_classes = set(train_labels["composition_code"].astype(str).tolist())
        test_classes = set(test_labels["composition_code"].astype(str).tolist())
        if train_classes != test_classes:
            continue

        train_groups = train_labels["specimen_group"].astype(str).to_numpy()
        test_groups = test_labels["specimen_group"].astype(str).to_numpy()
        if set(train_groups).intersection(set(test_groups)):
            continue

        yield {
            "seed": seed,
            "train_groups": train_groups,
            "test_groups": test_groups,
            "train_labels": train_labels,
            "test_labels": test_labels,
        }


def make_nested_group_holdout_splits(
    common_group_labels: pd.DataFrame,
    metadata_group_labels: pd.DataFrame,
    outer_test_size: float = 0.4,
    inner_test_size: float = 0.35,
    outer_random_state: int = 42,
    inner_random_state: int = 142,
    max_outer_seed_tries: int = 500,
    max_inner_seed_tries: int = 500,
) -> dict:
    common_group_labels = _normalize_group_labels(common_group_labels)
    metadata_group_labels = _normalize_group_labels(metadata_group_labels)

    common_groups = set(common_group_labels["specimen_group"].astype(str).tolist())
    metadata_groups = set(metadata_group_labels["specimen_group"].astype(str).tolist())
    missing_groups = sorted(common_groups - metadata_groups)
    if missing_groups:
        raise ValueError(
            "metadata_group_labels is missing specimen_group values that appear in common_group_labels. "
            f"Examples: {missing_groups[:10]}"
        )

    split = make_nested_group_splits(
        common_group_labels,
        outer_test_size=outer_test_size,
        inner_calibration_size=inner_test_size,
        outer_random_state=outer_random_state,
        inner_random_state=inner_random_state,
        max_outer_seed_tries=max_outer_seed_tries,
        max_inner_seed_tries=max_inner_seed_tries,
    )

    fit_groups = sorted(pd.Series(split.fit_groups).astype(str).tolist())
    cal_groups = sorted(pd.Series(split.calibration_groups).astype(str).tolist())
    test_groups = sorted(pd.Series(split.test_groups).astype(str).tolist())
    train_common_groups = sorted(pd.Series(list(fit_groups) + list(cal_groups)).astype(str).unique().tolist())

    return {
        "train_common_groups": train_common_groups,
        "test_common_groups": test_groups,
        "fit_groups": fit_groups,
        "cal_groups": cal_groups,
        "outer_seed": int(split.outer_random_state),
        "inner_seed": int(split.inner_random_state),
        "inner_split_strategy": str(split.calibration_strategy),
        "calibration_strategy": str(split.calibration_strategy),
    }


def build_multidevice_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    instruments: Sequence[str] = DEFAULT_DEVICES,
    measurement: str = "Reflectance",
    common_grid_points: int = 128,
    min_common_groups_for_eval: int = 5,
) -> dict:
    df = training_matrix.load_nir_subset(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        instruments=instruments,
        measurement=measurement,
    )
    df, common_classes, class_presence = training_matrix.keep_common_classes(df, instruments=instruments)
    if df.empty or not common_classes:
        raise ValueError(
            "No composition classes are shared across all requested instruments after initial filtering. "
            f"Instruments: {list(instruments)}, measurement: {measurement}"
        )
    class_summary_common_groups = training_matrix.summarize_common_class_groups(
        training_matrix.keep_groups_present_in_all_instruments(df, instruments=instruments)[0],
        instruments=instruments,
    )
    eval_classes = class_summary_common_groups.loc[
        class_summary_common_groups["min_groups_across_instruments"] >= int(min_common_groups_for_eval),
        "composition_code",
    ].astype(str).tolist()
    if not eval_classes:
        raise ValueError(
            "No composition classes satisfy the requested multi-device evaluation filter. "
            f"min_common_groups_for_eval={int(min_common_groups_for_eval)}; "
            f"available per-class minima={class_summary_common_groups[['composition_code', 'min_groups_across_instruments']].to_dict(orient='records')[:10]}"
        )

    df = df[df["composition_code"].astype(str).isin(eval_classes)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "All rows were removed after filtering to evaluation classes. "
            f"eval_classes={eval_classes[:10]}"
        )

    common_group_df, common_groups = training_matrix.keep_groups_present_in_all_instruments(df, instruments=instruments)
    if common_group_df.empty or len(common_groups) == 0:
        raise ValueError(
            "No specimen groups remain that are present across all requested instruments after evaluation-class filtering. "
            f"Instruments: {list(instruments)}, eval_classes={eval_classes[:10]}"
        )
    target_grid = training_matrix.infer_common_wavelength_grid(df, instruments=instruments, n_points=common_grid_points)
    df_resampled = training_matrix.resample_dataframe_to_grid(df, target_grid)
    common_group_resampled = training_matrix.resample_dataframe_to_grid(common_group_df, target_grid)

    X, y, _, groups, wavelengths, metadata, label_map = training_matrix.build_training_matrix(df_resampled)
    _, _, _, _, _, metadata_common, _ = training_matrix.build_training_matrix(common_group_resampled)

    common_group_labels = metadata_common[["specimen_group", "composition_code"]].drop_duplicates().reset_index(drop=True)
    if common_group_labels.empty:
        raise ValueError(
            "No common specimen-group labels remain after resampling; cannot build grouped multi-device evaluation splits."
        )

    return {
        "raw_df": df,
        "df_resampled": df_resampled,
        "common_group_df": common_group_resampled,
        "X": X,
        "y": y,
        "groups": groups,
        "wavelengths": wavelengths,
        "metadata": metadata,
        "label_map": label_map,
        "common_groups": common_groups,
        "common_group_labels": common_group_labels,
        "eval_classes": eval_classes,
        "class_presence": class_presence,
        "class_summary_common_groups": class_summary_common_groups,
    }


def _prepare_features(
    X_subset: np.ndarray,
    wavelengths: np.ndarray,
    method: str = "snv",
    preprocessing_state: dict | None = None,
) -> np.ndarray:
    return preprocessing.preprocess_spectra(
        X_subset,
        wavelengths,
        method=method,
        preprocessing_state=preprocessing_state,
    )


def _metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment_name: str,
    train_devices: str,
    test_device: str,
    training_strategy: str,
) -> dict:
    return {
        "experiment": experiment_name,
        "training_strategy": str(training_strategy),
        "train_devices": train_devices,
        "test_device": test_device,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _normalize_training_strategies(training_strategies: Sequence[str] | None) -> list[str]:
    if training_strategies is None:
        training_strategies = DEFAULT_TRAINING_STRATEGIES
    normalized: list[str] = []
    for strategy in training_strategies:
        name = str(strategy).strip().lower()
        if not name:
            continue
        if name not in {"baseline", "domain_generalized"}:
            raise ValueError(
                "Unsupported training strategy. Expected one of ['baseline', 'domain_generalized'], "
                f"got {strategy!r}."
            )
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("At least one training strategy is required.")
    return normalized


def _fit_model_bundle(
    *,
    training_strategy: str,
    X_fit_raw: np.ndarray,
    y_fit: np.ndarray,
    fit_domains: np.ndarray,
    fit_groups: np.ndarray,
    X_cal_raw: np.ndarray | None,
    y_cal: np.ndarray | None,
    wavelengths: np.ndarray,
    preprocessing_method: str,
    pca_components: int,
    alpha: float,
    calibration_method: str,
    domain_generalization_config: Mapping[str, Any] | None,
):
    strategy = str(training_strategy).strip().lower()
    if strategy == "baseline":
        preprocessing_state = preprocessing.fit_preprocessing_state(
            X_fit_raw,
            wavelengths,
            method=preprocessing_method,
        )
        X_fit = _prepare_features(
            X_fit_raw,
            wavelengths,
            method=preprocessing_method,
            preprocessing_state=preprocessing_state,
        )
        if X_cal_raw is not None and y_cal is not None and len(X_cal_raw) > 0:
            X_cal = _prepare_features(
                X_cal_raw,
                wavelengths,
                method=preprocessing_method,
                preprocessing_state=preprocessing_state,
            )
        else:
            X_cal = None
            y_cal = None
        model_bundle = fit_calibrated_model(
            X_train=X_fit,
            y_train=y_fit,
            X_cal=X_cal,
            y_cal=y_cal,
            n_components=pca_components,
            alpha=alpha,
            preprocessing_method=preprocessing_method,
            calibration_method=calibration_method,
            preprocessing_state=preprocessing_state,
        )
        return model_bundle, {
            "training_strategy": strategy,
            "n_original_train": int(len(X_fit_raw)),
            "n_augmented_train": int(len(X_fit_raw)),
        }

    if strategy == "domain_generalized":
        fitted = fit_domain_generalized_calibrated_model(
            X_train_raw=X_fit_raw,
            y_train=y_fit,
            wavelengths=wavelengths,
            train_domains=fit_domains,
            train_groups=fit_groups,
            X_cal_raw=X_cal_raw,
            y_cal=y_cal,
            preprocessing_method=preprocessing_method,
            n_components=pca_components,
            alpha=alpha,
            calibration_method=calibration_method,
            config=domain_generalization_config,
        )
        return fitted["model_bundle"], {
            "training_strategy": strategy,
            "n_original_train": int(fitted["n_original_train"]),
            "n_augmented_train": int(fitted["n_augmented_train"]),
            "augmentation_multiplier": float(fitted["augmentation_multiplier"]),
        }

    raise ValueError(f"Unsupported training strategy: {training_strategy!r}")


def run_device_domain_experiments(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    instruments: Sequence[str] = DEFAULT_DEVICES,
    measurement: str = "Reflectance",
    preprocessing_method: str = "snv",
    pca_components: int = 20,
    alpha: float = 1.0,
    calibration_method: str = "sigmoid",
    common_grid_points: int = 128,
    min_common_groups_for_eval: int = 5,
    output_dir: str | Path | None = None,
    bundle: Optional[dict] = None,
    split_info: Optional[dict] = None,
    training_strategies: Sequence[str] | None = None,
    domain_generalization_config: Mapping[str, Any] | None = None,
) -> dict:
    instruments = list(instruments)
    training_strategies = _normalize_training_strategies(training_strategies)
    dg_config = config_to_dict(DEFAULT_DG_CONFIG if domain_generalization_config is None else domain_generalization_config)

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
    if split_info is None:
        split_info = make_nested_group_holdout_splits(
            common_group_labels=bundle["common_group_labels"],
            metadata_group_labels=metadata[["specimen_group", "composition_code"]],
            outer_test_size=0.4,
            inner_test_size=0.35,
            outer_random_state=42,
            inner_random_state=142,
        )

    train_common_groups = set(split_info["train_common_groups"])
    test_common_groups = set(split_info["test_common_groups"])
    fit_groups = set(split_info["fit_groups"])
    cal_groups = set(split_info["cal_groups"])
    calibration_strategy = str(split_info.get("inner_split_strategy", split_info.get("calibration_strategy", "grouped_inner_holdout")))

    X = np.asarray(bundle["X"], dtype=float)
    y = bundle["y"].astype(str)
    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)
    instrument_series = metadata["instrument"].astype(str)
    group_series = metadata["specimen_group"].astype(str)

    metrics = []
    prediction_tables: dict[str, pd.DataFrame] = {}
    fit_summaries: list[dict[str, Any]] = []

    for training_strategy in training_strategies:
        # Same-device and zero-shot device transfer from one source device.
        for train_device in instruments:
            fit_mask = instrument_series.eq(train_device) & group_series.isin(fit_groups)
            cal_mask = instrument_series.eq(train_device) & group_series.isin(cal_groups)
            if fit_mask.sum() == 0:
                continue

            X_fit_raw = X[fit_mask]
            y_fit = y[fit_mask]
            fit_domains = instrument_series[fit_mask].astype(str).to_numpy()
            fit_group_ids = group_series[fit_mask].astype(str).to_numpy()
            if bool(cal_mask.sum()):
                X_cal_raw = X[cal_mask]
                y_cal = y[cal_mask]
            else:
                X_cal_raw = None
                y_cal = None

            model_bundle, fit_summary = _fit_model_bundle(
                training_strategy=training_strategy,
                X_fit_raw=X_fit_raw,
                y_fit=y_fit,
                fit_domains=fit_domains,
                fit_groups=fit_group_ids,
                X_cal_raw=X_cal_raw,
                y_cal=y_cal,
                wavelengths=wavelengths,
                preprocessing_method=preprocessing_method,
                pca_components=pca_components,
                alpha=alpha,
                calibration_method=calibration_method,
                domain_generalization_config=dg_config,
            )
            fit_summaries.append(
                {
                    "experiment": "single_source_device_transfer",
                    "training_strategy": training_strategy,
                    "train_devices": train_device,
                    **fit_summary,
                }
            )

            for test_device in instruments:
                test_mask = instrument_series.eq(test_device) & group_series.isin(test_common_groups)
                if test_mask.sum() == 0:
                    continue

                X_test = _prepare_features(
                    X[test_mask],
                    wavelengths,
                    method=preprocessing_method,
                    preprocessing_state=model_bundle.preprocessing_state,
                )
                pred_df = predict_with_uncertainty(model_bundle, X_test)
                pred_df = pd.concat([metadata.loc[test_mask].reset_index(drop=True), pred_df], axis=1)
                pred_df.insert(0, "training_strategy", training_strategy)
                prediction_tables[f"{training_strategy}__{train_device}__to__{test_device}"] = pred_df
                metrics.append(
                    _metric_row(
                        y_true=y[test_mask],
                        y_pred=pred_df["predicted_label"].astype(str).to_numpy(),
                        experiment_name="single_source_device_transfer",
                        train_devices=train_device,
                        test_device=test_device,
                        training_strategy=training_strategy,
                    )
                )

        # Leave-one-device-out pooled training.
        for heldout_device in instruments:
            source_devices = [d for d in instruments if d != heldout_device]
            fit_mask = instrument_series.isin(source_devices) & group_series.isin(fit_groups)
            cal_mask = instrument_series.isin(source_devices) & group_series.isin(cal_groups)
            test_mask = instrument_series.eq(heldout_device) & group_series.isin(test_common_groups)
            if fit_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            X_fit_raw = X[fit_mask]
            y_fit = y[fit_mask]
            fit_domains = instrument_series[fit_mask].astype(str).to_numpy()
            fit_group_ids = group_series[fit_mask].astype(str).to_numpy()
            if bool(cal_mask.sum()):
                X_cal_raw = X[cal_mask]
                y_cal = y[cal_mask]
            else:
                X_cal_raw = None
                y_cal = None

            model_bundle, fit_summary = _fit_model_bundle(
                training_strategy=training_strategy,
                X_fit_raw=X_fit_raw,
                y_fit=y_fit,
                fit_domains=fit_domains,
                fit_groups=fit_group_ids,
                X_cal_raw=X_cal_raw,
                y_cal=y_cal,
                wavelengths=wavelengths,
                preprocessing_method=preprocessing_method,
                pca_components=pca_components,
                alpha=alpha,
                calibration_method=calibration_method,
                domain_generalization_config=dg_config,
            )
            fit_summaries.append(
                {
                    "experiment": "leave_one_device_out",
                    "training_strategy": training_strategy,
                    "train_devices": "+".join(source_devices),
                    **fit_summary,
                }
            )

            X_test = _prepare_features(
                X[test_mask],
                wavelengths,
                method=preprocessing_method,
                preprocessing_state=model_bundle.preprocessing_state,
            )
            pred_df = predict_with_uncertainty(model_bundle, X_test)
            pred_df = pd.concat([metadata.loc[test_mask].reset_index(drop=True), pred_df], axis=1)
            pred_df.insert(0, "training_strategy", training_strategy)
            prediction_tables[f"{training_strategy}__LODO__{heldout_device}"] = pred_df
            metrics.append(
                _metric_row(
                    y_true=y[test_mask],
                    y_pred=pred_df["predicted_label"].astype(str).to_numpy(),
                    experiment_name="leave_one_device_out",
                    train_devices="+".join(source_devices),
                    test_device=heldout_device,
                    training_strategy=training_strategy,
                )
            )

    if not metrics:
        raise ValueError(
            "No device-domain experiments could be executed with the requested dataset subset and grouped splits."
        )

    metrics_df = pd.DataFrame(metrics).sort_values(
        by=["training_strategy", "experiment", "test_device", "balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[True, True, True, False, False, False],
    ).reset_index(drop=True)
    fit_summary_df = pd.DataFrame(fit_summaries)

    outputs = {
        "bundle": bundle,
        "outer_split": {
            "train_common_groups": sorted(train_common_groups),
            "test_common_groups": sorted(test_common_groups),
            "fit_groups": sorted(fit_groups),
            "cal_groups": sorted(cal_groups),
            "calibration_strategy": calibration_strategy,
        },
        "training_strategies": training_strategies,
        "domain_generalization_config": dg_config,
        "metrics_df": metrics_df,
        "fit_summary_df": fit_summary_df,
        "prediction_tables": prediction_tables,
    }

    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_dir / "device_domain_metrics.csv", index=False)
        if not fit_summary_df.empty:
            fit_summary_df.to_csv(output_dir / "device_domain_fit_summary.csv", index=False)
        bundle["class_presence"].to_csv(output_dir / "class_presence.csv", index=False)
        bundle["class_summary_common_groups"].to_csv(output_dir / "common_group_class_summary.csv", index=False)
        split_summary = {
            **outputs["outer_split"],
            "training_strategies": training_strategies,
            "domain_generalization_config": dg_config,
        }
        (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
        for name, table in prediction_tables.items():
            table.to_csv(output_dir / f"predictions__{name}.csv", index=False)

        # ── Publication figures ───────────────────────────────────────────
        from publication_outputs import figure_device_transfer_heatmap  # noqa: PLC0415
        for strategy in training_strategies:
            strategy_df = metrics_df[metrics_df["training_strategy"] == strategy].reset_index(drop=True)
            if strategy_df.empty:
                continue
            figure_device_transfer_heatmap(strategy_df, output_dir / f"strategy_{strategy}")

    return outputs


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    run_device_domain_experiments(
        dataset_path=base_dir,
        loader_script_path=base_dir / "nir_loader.py",
        output_dir=base_dir / "outputs_device_domain_experiments",
    )
    # All outputs saved to outputs_device_domain_experiments/
