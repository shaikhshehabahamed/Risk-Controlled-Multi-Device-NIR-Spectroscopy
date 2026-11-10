from __future__ import annotations
import logging

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import preprocessing
from circularity_proxy import build_proxy_table
from composition_aware_model import run_composition_aware_multidevice_experiment
from device_domain_experiments import (
    build_multidevice_dataset,
    make_nested_group_holdout_splits,
    run_device_domain_experiments,
)
from domain_generalization import DomainGeneralizationConfig
from open_set_unknown import run_open_set_experiment
from reliability import (
    apply_abstention_policy,
    evaluate_policy,
    fit_calibrated_model,
    make_group_holdout_split,
    predict_with_uncertainty,
)
from robustness_analysis import evaluate_robustness_suite
from risk_controlled_policy import (
    RiskControlTargets,
    save_risk_control_outputs,
    select_risk_controlled_policy,
)
from rvea_optimizer import optimize_abstention_policy, save_policy_optimization_outputs
from publication_outputs import generate_all_figures, figure_mean_spectra


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_breakthrough_study"
DEFAULT_DEVICES = ["Benchtop_N", "Handheld_F", "Handheld_S"]



def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()



def _prepare_core_splits(bundle: dict) -> dict:
    metadata = bundle["metadata"].copy().reset_index(drop=True)
    return make_nested_group_holdout_splits(
        common_group_labels=bundle["common_group_labels"],
        metadata_group_labels=metadata[["specimen_group", "composition_code"]],
        outer_test_size=0.4,
        inner_test_size=0.35,   # 0.25 causes banker's rounding (round(0.5)=0) for
                                # 4-group minority classes, giving 0 calibration samples
                                # and making the class-complete check always fail.
                                # 0.35 → round(0.7)=1 safely allocates ≥1 per class.
        outer_random_state=42,
        inner_random_state=142,
    )



def _build_fit_proxy_predictions(
    X_raw: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    wavelengths: np.ndarray,
    preprocessing_method: str,
    pca_components: int,
    alpha: float,
    calibration_method: str,
) -> tuple[pd.DataFrame, str]:
    metadata = metadata.reset_index(drop=True).copy()
    y = pd.Series(np.asarray(y)).astype(str).to_numpy()
    group_series = metadata["specimen_group"].astype(str)
    fit_group_labels = metadata[["specimen_group", "composition_code"]].drop_duplicates().reset_index(drop=True)

    def _fit_predict_for_indices(train_idx: np.ndarray, val_idx: np.ndarray) -> pd.DataFrame:
        train_idx = np.asarray(train_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("Policy-fit proxy folds must contain at least one train sample and one validation sample.")

        preprocessing_state = preprocessing.fit_preprocessing_state(
            X_raw[train_idx],
            wavelengths,
            method=preprocessing_method,
        )
        X_train = preprocessing.preprocess_spectra(
            X_raw[train_idx],
            wavelengths,
            method=preprocessing_method,
            preprocessing_state=preprocessing_state,
        )
        X_val = preprocessing.preprocess_spectra(
            X_raw[val_idx],
            wavelengths,
            method=preprocessing_method,
            preprocessing_state=preprocessing_state,
        )
        model_bundle = fit_calibrated_model(
            X_train=X_train,
            y_train=y[train_idx],
            X_cal=None,
            y_cal=None,
            n_components=pca_components,
            alpha=alpha,
            preprocessing_method=preprocessing_method,
            calibration_method=calibration_method,
            preprocessing_state=preprocessing_state,
        )
        fold_pred = predict_with_uncertainty(model_bundle, X_val)
        out = pd.concat([metadata.loc[val_idx].reset_index(drop=True), fold_pred], axis=1)
        out["_row_id"] = val_idx.astype(int)
        return out

    def _finalize(prediction_chunks: list[pd.DataFrame], source_name: str) -> tuple[pd.DataFrame, str] | None:
        if not prediction_chunks:
            return None
        out = pd.concat(prediction_chunks, axis=0, ignore_index=True)
        if out["_row_id"].nunique() != len(metadata):
            return None
        out = out.sort_values("_row_id").drop_duplicates("_row_id", keep="last").reset_index(drop=True)
        out = out.drop(columns=["_row_id"])
        sort_cols = [col for col in ["instrument", "specimen_group", "filename"] if col in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)
        return out, source_name

    min_groups_per_class = int(fit_group_labels["composition_code"].value_counts().min()) if not fit_group_labels.empty else 0
    if min_groups_per_class >= 2:
        preferred_splits = min(3, min_groups_per_class)
        prediction_chunks: list[pd.DataFrame] = []
        splitter = StratifiedGroupKFold(n_splits=preferred_splits, shuffle=True, random_state=142)
        dummy = np.zeros((len(metadata), 1), dtype=float)
        for train_idx, val_idx in splitter.split(dummy, y, group_series.to_numpy()):
            prediction_chunks.append(_fit_predict_for_indices(train_idx, val_idx))
        finalized = _finalize(
            prediction_chunks,
            source_name=f"grouped_fit_proxy_stratified_group_kfold_{preferred_splits}",
        )
        if finalized is not None:
            return finalized

    try:
        prediction_chunks = []
        holdout = make_group_holdout_split(
            fit_group_labels,
            test_size=0.5,
            random_state=142,
            max_seed_tries=500,
        )
        fold_group_pairs = [
            (
                set(pd.Series(holdout.train_groups).astype(str).tolist()),
                set(pd.Series(holdout.test_groups).astype(str).tolist()),
            ),
            (
                set(pd.Series(holdout.test_groups).astype(str).tolist()),
                set(pd.Series(holdout.train_groups).astype(str).tolist()),
            ),
        ]
        for train_groups, val_groups in fold_group_pairs:
            train_idx = np.flatnonzero(group_series.isin(train_groups).to_numpy())
            val_idx = np.flatnonzero(group_series.isin(val_groups).to_numpy())
            prediction_chunks.append(_fit_predict_for_indices(train_idx, val_idx))
        finalized = _finalize(prediction_chunks, source_name="grouped_fit_proxy_two_holdout_folds")
        if finalized is not None:
            return finalized
    except (ValueError, Exception):
        # Fewer than 2 groups per class in the fit pool (common when
        # min_common_groups_for_eval is low and the nested split leaves only
        # 1 group per minority class).  Fall through to in-sample predictions.
        pass

    preprocessing_state = preprocessing.fit_preprocessing_state(
        X_raw,
        wavelengths,
        method=preprocessing_method,
    )
    X_fit = preprocessing.preprocess_spectra(
        X_raw,
        wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )
    model_bundle = fit_calibrated_model(
        X_train=X_fit,
        y_train=y,
        X_cal=None,
        y_cal=None,
        n_components=pca_components,
        alpha=alpha,
        preprocessing_method=preprocessing_method,
        calibration_method=calibration_method,
        preprocessing_state=preprocessing_state,
    )
    # Fix #37: in-sample predictions are overconfident — the model sees its own
    # training data, so calibrated probabilities are biased high.  Thresholds
    # derived from these predictions will be set too low and will accept more
    # samples than they should at test time.  Warn explicitly.
    import warnings as _warnings
    _warnings.warn(
        "_build_fit_proxy_predictions: all grouped cross-validation attempts failed; "
        "falling back to IN-SAMPLE predictions for policy threshold selection.  "
        "The resulting thresholds will be overconfident (biased toward acceptance) "
        "because the model was scored on its own training data.  "
        "Consider providing more specimen groups (min_common_groups_for_eval) to "
        "enable proper held-out proxy predictions.",
        UserWarning,
        stacklevel=4,
    )
    in_sample_pred = predict_with_uncertainty(model_bundle, X_fit)
    in_sample_pred = pd.concat([metadata, in_sample_pred], axis=1)
    sort_cols = [col for col in ["instrument", "specimen_group", "filename"] if col in in_sample_pred.columns]
    if sort_cols:
        in_sample_pred = in_sample_pred.sort_values(sort_cols).reset_index(drop=True)
    return in_sample_pred, "in_sample_fit_proxy_full_fit_model"



def run_breakthrough_pipeline(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    instruments: Sequence[str] = DEFAULT_DEVICES,
    measurement: str = "Reflectance",
    preprocessing_method: str = "snv_savgol1",
    pca_components: int = 30,
    alpha: float = 0.5,
    calibration_method: str = "isotonic",
    common_grid_points: int = 256,
    min_common_groups_for_eval: int = 3,
    population_size: int = 100,
    generations: int = 80,
    policy_focus_min_errors: int = 2,
    policy_max_focus_pairs: int = 4,
    policy_entropy_min: float = 0.02,
    policy_entropy_max: float = 0.60,
    risk_target_purity: float = 0.98,
    risk_min_coverage: float = 0.30,
    risk_max_manual_review_rate: float = 0.70,
    risk_min_worst_domain_performance: float = 0.60,
    run_open_set: bool = True,
    open_set_n_unknown_classes: int = 2,
    open_set_min_groups_per_unknown_class: int = 3,
    open_set_score_method: str = "entropy",
    open_set_threshold_selection: str = "known_acceptance",
    open_set_target_known_acceptance: float = 0.95,
    dg_augmented_copies: int = 5,
    min_groups_for_domain_eval: int = 5,
) -> dict:
    dataset_path = _resolve_local_path(dataset_path)
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_multidevice_dataset(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        instruments=instruments,
        measurement=measurement,
        common_grid_points=common_grid_points,
        min_common_groups_for_eval=min_common_groups_for_eval,
    )
    split_info = _prepare_core_splits(bundle)

    metadata = bundle["metadata"].copy().reset_index(drop=True)
    X_raw = np.asarray(bundle["X"], dtype=float)
    y = bundle["y"].astype(str)
    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)

    fit_mask = metadata["specimen_group"].astype(str).isin(split_info["fit_groups"])
    cal_mask = metadata["specimen_group"].astype(str).isin(split_info["cal_groups"])
    test_mask = metadata["specimen_group"].astype(str).isin(split_info["test_common_groups"])

    if not bool(fit_mask.sum()) or not bool(test_mask.sum()):
        raise ValueError("Breakthrough pipeline created an empty fit or test split, which is not valid.")

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

    probability_calibration_predictions = None
    probability_calibration_source = None
    if bool(cal_mask.sum()):
        X_cal = preprocessing.preprocess_spectra(
            X_raw[cal_mask],
            wavelengths,
            method=preprocessing_method,
            preprocessing_state=preprocessing_state,
        )
        y_cal = y[cal_mask]
        calibration_meta = metadata.loc[cal_mask].reset_index(drop=True)
        probability_calibration_source = "heldout_calibration_groups"
    else:
        X_cal = None
        y_cal = None
        calibration_meta = None

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

    if X_cal is not None and y_cal is not None and calibration_meta is not None:
        probability_calibration_predictions = predict_with_uncertainty(model_bundle, X_cal)
        probability_calibration_predictions = pd.concat(
            [calibration_meta, probability_calibration_predictions],
            axis=1,
        )

    policy_fit_pred, policy_fit_prediction_source = _build_fit_proxy_predictions(
        X_raw=X_raw[fit_mask],
        y=y[fit_mask],
        metadata=metadata.loc[fit_mask].reset_index(drop=True),
        wavelengths=wavelengths,
        preprocessing_method=preprocessing_method,
        pca_components=pca_components,
        alpha=alpha,
        calibration_method=calibration_method,
    )
    policy_fit_y = policy_fit_pred["composition_code"].astype(str).to_numpy()

    test_pred = predict_with_uncertainty(model_bundle, X_test)
    test_meta = metadata.loc[test_mask].reset_index(drop=True)
    test_pred = pd.concat([test_meta, test_pred], axis=1)

    device_order = list(instruments)
    policy_opt_outputs = optimize_abstention_policy(
        prediction_df=policy_fit_pred,
        y_true=policy_fit_y,
        domains=policy_fit_pred["instrument"].astype(str).to_numpy(),
        device_order=device_order,
        threshold_bounds=(0.35, 0.995),
        min_margin_bounds=(0.0, 0.60),
        entropy_bounds=(float(policy_entropy_min), float(policy_entropy_max)),
        focus_min_errors=int(policy_focus_min_errors),
        max_focus_pairs=int(policy_max_focus_pairs),
        population_size=population_size,
        generations=generations,
        random_state=42,
    )
    pareto_df = policy_opt_outputs["pareto_df"]
    base_policy_json = dict(policy_opt_outputs["selected_policy"])
    base_policy_json["policy_fit_prediction_source"] = policy_fit_prediction_source
    base_policy_json["probability_calibration_source"] = probability_calibration_source

    risk_targets = RiskControlTargets(
        target_purity=float(risk_target_purity),
        min_coverage=float(risk_min_coverage),
        max_manual_review_rate=float(risk_max_manual_review_rate),
        min_worst_domain_performance=float(risk_min_worst_domain_performance),
    )
    risk_control_outputs = select_risk_controlled_policy(
        prediction_df=policy_fit_pred,
        y_true=policy_fit_y,
        domains=policy_fit_pred["instrument"].astype(str).to_numpy(),
        base_policy=base_policy_json,
        targets=risk_targets,
        top_k_pairs=int(policy_max_focus_pairs),
    )
    save_risk_control_outputs(output_dir, risk_control_outputs)

    policy_json = dict(risk_control_outputs["selected_policy"])
    policy_json["policy_fit_prediction_source"] = policy_fit_prediction_source
    policy_json["probability_calibration_source"] = probability_calibration_source
    policy_json["risk_control_targets"] = {
        "target_purity": float(risk_target_purity),
        "min_coverage": float(risk_min_coverage),
        "max_manual_review_rate": float(risk_max_manual_review_rate),
        "min_worst_domain_performance": float(risk_min_worst_domain_performance),
    }

    policy_fit_accept = apply_abstention_policy(
        prediction_df=policy_fit_pred,
        domains=policy_fit_pred["instrument"].astype(str).to_numpy(),
        policy=policy_json,
    )
    test_accept = apply_abstention_policy(
        prediction_df=test_pred,
        domains=test_meta["instrument"].astype(str).to_numpy(),
        policy=policy_json,
    )

    policy_fit_metrics = evaluate_policy(
        y_true=policy_fit_y,
        domains=policy_fit_pred["instrument"].astype(str).to_numpy(),
        prediction_df=policy_fit_pred,
        accepted_mask=policy_fit_accept,
        thresholds_by_domain=policy_json["thresholds_by_domain"],
        min_margin=policy_json["min_margin"],
        policy=policy_json,
    )
    test_metrics = evaluate_policy(
        y_true=y[test_mask],
        domains=test_meta["instrument"].astype(str).to_numpy(),
        prediction_df=test_pred,
        accepted_mask=test_accept,
        thresholds_by_domain=policy_json["thresholds_by_domain"],
        min_margin=policy_json["min_margin"],
        policy=policy_json,
    )

    robustness_df = evaluate_robustness_suite(
        model_bundle=model_bundle,
        X_test_raw=X_raw[test_mask],
        y_test=y[test_mask],
        domains=test_meta["instrument"].astype(str).to_numpy(),
        wavelengths=wavelengths,
        policy=policy_json,
        preprocessing_method=preprocessing_method,
    )

    # Device domain experiments require at least min_groups_for_domain_eval groups
    # per class to produce reliable cross-device transfer estimates.  When
    # min_common_groups_for_eval is low (e.g. 3), minority classes end up with
    # only 1 fit group, which collapses LODO accuracy.  Build a separate,
    # stricter bundle so device transfer is always evaluated on well-represented
    # classes, regardless of what the main analysis includes.
    _effective_domain_threshold = max(min_common_groups_for_eval, min_groups_for_domain_eval)
    if _effective_domain_threshold > min_common_groups_for_eval:
        _domain_bundle = build_multidevice_dataset(
            dataset_path=dataset_path,
            loader_script_path=loader_script_path,
            instruments=instruments,
            measurement=measurement,
            common_grid_points=common_grid_points,
            min_common_groups_for_eval=_effective_domain_threshold,
        )
        _domain_split_info = None  # let run_device_domain_experiments build its own split
    else:
        _domain_bundle = bundle
        _domain_split_info = split_info

    domain_results = run_device_domain_experiments(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        instruments=instruments,
        measurement=measurement,
        preprocessing_method=preprocessing_method,
        pca_components=pca_components,
        alpha=alpha,
        calibration_method=calibration_method,
        common_grid_points=common_grid_points,
        min_common_groups_for_eval=_effective_domain_threshold,
        output_dir=output_dir / "device_domain_experiments",
        bundle=_domain_bundle,
        split_info=_domain_split_info,
        domain_generalization_config=DomainGeneralizationConfig(
            n_augmented_copies=dg_augmented_copies,
        ),
    )

    composition_aware_results = run_composition_aware_multidevice_experiment(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        output_dir=output_dir / "composition_aware",
        instruments=instruments,
        measurement=measurement,
        preprocessing_method=preprocessing_method,
        pca_components=pca_components,
        alpha=alpha,
        common_grid_points=common_grid_points,
        min_common_groups_for_eval=min_common_groups_for_eval,
        bundle=bundle,
        split_info=split_info,
    )
    comparison_df = pd.DataFrame([
        {
            "model": "flat_calibrated_classifier",
            "label_accuracy": float(test_metrics.accepted_accuracy),
            "label_balanced_accuracy": float(test_metrics.accepted_balanced_accuracy),
            "label_macro_f1": float(test_metrics.accepted_macro_f1),
        },
        {
            "model": "composition_aware_structured",
            **{k: v for k, v in composition_aware_results["metrics"].items() if isinstance(v, (int, float))},
        },
    ])
    comparison_df.to_csv(output_dir / "composition_aware" / "model_comparison.csv", index=False)
    pd.DataFrame([composition_aware_results["metrics"]]).to_csv(output_dir / "composition_aware" / "composition_aware_metrics_flat.csv", index=False)

    # Guard: open-set evaluation requires at least 2 known classes after designating
    # open_set_n_unknown_classes as unknown.  With fewer known classes the model is
    # trained on only 1 class, the score distribution is degenerate, and AUROC
    # collapses to 0.5 (random chance) — a result that is harmful to publish.
    _n_eval = int(bundle["common_group_labels"]["composition_code"].nunique())
    _n_known_after_split = _n_eval - open_set_n_unknown_classes
    if bool(run_open_set) and _n_known_after_split < 2:
        import warnings as _w
        _w.warn(
            f"Open-set experiment skipped: {_n_eval} evaluated class(es) minus "
            f"{open_set_n_unknown_classes} unknown = {_n_known_after_split} known class(es). "
            "A minimum of 2 known classes is required for a non-degenerate AUROC. "
            "Lower --min-common-groups-for-eval or reduce --open-set-n-unknown-classes.",
            UserWarning,
            stacklevel=2,
        )
        run_open_set = False

    open_set_results = None
    if bool(run_open_set):
        open_set_results = run_open_set_experiment(
            dataset_path=dataset_path,
            loader_script_path=loader_script_path,
            output_dir=output_dir / "open_set_unknown",
            instruments=instruments,
            measurement=measurement,
            preprocessing_method=preprocessing_method,
            pca_components=pca_components,
            alpha=alpha,
            calibration_method=calibration_method,
            common_grid_points=common_grid_points,
            min_common_groups_for_eval=min_common_groups_for_eval,
            n_unknown_classes=open_set_n_unknown_classes,
            min_groups_per_unknown_class=open_set_min_groups_per_unknown_class,
            unknown_score_method=open_set_score_method,
            threshold_selection=open_set_threshold_selection,
            target_known_acceptance=open_set_target_known_acceptance,
            bundle=bundle,
            policy=policy_json,
        )

    policy_fit_predictions_out = policy_fit_pred.copy()
    policy_fit_predictions_out["is_accepted"] = policy_fit_accept.astype(int)
    policy_fit_predictions_out["requires_manual_review"] = (~policy_fit_accept).astype(int)
    test_predictions_out = test_pred.copy()
    test_predictions_out["is_accepted"] = test_accept.astype(int)
    test_predictions_out["requires_manual_review"] = (~test_accept).astype(int)

    policy_fit_predictions_out.to_csv(output_dir / "policy_fit_predictions.csv", index=False)
    policy_fit_predictions_out.to_csv(output_dir / "calibration_predictions.csv", index=False)
    test_predictions_out.to_csv(output_dir / "test_predictions.csv", index=False)
    if probability_calibration_predictions is not None:
        probability_calibration_predictions.to_csv(
            output_dir / "probability_calibration_predictions.csv",
            index=False,
        )

    save_policy_optimization_outputs(output_dir, policy_opt_outputs)
    robustness_df.to_csv(output_dir / "robustness_results.csv", index=False)
    build_proxy_table(bundle["eval_classes"]).to_csv(output_dir / "circularity_proxy_table.csv", index=False)
    bundle["class_presence"].to_csv(output_dir / "class_presence.csv", index=False)
    bundle["class_summary_common_groups"].to_csv(output_dir / "common_group_class_summary.csv", index=False)

    policy_metrics_df = pd.DataFrame(
        [
            {
                "split": policy_fit_prediction_source,
                "purity": policy_fit_metrics.purity,
                "coverage": policy_fit_metrics.coverage,
                "manual_review_rate": policy_fit_metrics.manual_review_rate,
                "worst_domain_performance": policy_fit_metrics.worst_domain_performance,
                "accepted_accuracy": policy_fit_metrics.accepted_accuracy,
                "accepted_balanced_accuracy": policy_fit_metrics.accepted_balanced_accuracy,
                "accepted_macro_f1": policy_fit_metrics.accepted_macro_f1,
                "contamination_rate": policy_fit_metrics.contamination_rate,
                "expected_circularity_proxy": policy_fit_metrics.expected_circularity_proxy,
                "realized_circularity_proxy": policy_fit_metrics.realized_circularity_proxy,
                "expected_utility": policy_fit_metrics.expected_utility,
                "realized_utility": policy_fit_metrics.realized_utility,
                "n_accepted": policy_fit_metrics.n_accepted,
            },
            {
                "split": "test",
                "purity": test_metrics.purity,
                "coverage": test_metrics.coverage,
                "manual_review_rate": test_metrics.manual_review_rate,
                "worst_domain_performance": test_metrics.worst_domain_performance,
                "accepted_accuracy": test_metrics.accepted_accuracy,
                "accepted_balanced_accuracy": test_metrics.accepted_balanced_accuracy,
                "accepted_macro_f1": test_metrics.accepted_macro_f1,
                "contamination_rate": test_metrics.contamination_rate,
                "expected_circularity_proxy": test_metrics.expected_circularity_proxy,
                "realized_circularity_proxy": test_metrics.realized_circularity_proxy,
                "expected_utility": test_metrics.expected_utility,
                "realized_utility": test_metrics.realized_utility,
                "n_accepted": test_metrics.n_accepted,
            },
        ]
    )
    policy_metrics_df.to_csv(output_dir / "policy_metrics.csv", index=False)

    split_summary = {
        "train_common_groups": split_info["train_common_groups"],
        "test_common_groups": split_info["test_common_groups"],
        "fit_groups": split_info["fit_groups"],
        "cal_groups": split_info["cal_groups"],
        "eval_classes": bundle["eval_classes"],
        "devices": list(instruments),
        "policy_fit_prediction_source": policy_fit_prediction_source,
        "probability_calibration_prediction_source": probability_calibration_source,
        "policy_fit_predictions_alias": "calibration_predictions.csv",
    }
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    (output_dir / "policy.json").write_text(json.dumps(policy_json, indent=2), encoding="utf-8")

    model_artifacts = {
        "calibrated_model": model_bundle.calibrated_model,
        "base_pipeline": model_bundle.base_pipeline,
        "wavelengths": wavelengths,
        "preprocessing_method": preprocessing_method,
        "preprocessing_state": preprocessing.serialize_preprocessing_state(preprocessing_state),
        "policy": policy_json,
        "class_labels": model_bundle.class_labels,
        "dataset_subset": {
            "instrument": list(instruments),
            "measurement": measurement,
        },
    }
    joblib.dump(model_artifacts, output_dir / "calibrated_reliable_model.joblib")

    summary = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "measurement": measurement,
        "devices": list(instruments),
        "n_eval_classes": int(len(bundle["eval_classes"])),
        "n_samples_total": int(len(bundle["X"])),
        "n_groups_total": int(pd.Series(bundle["groups"]).nunique()),
        "pca_components": int(model_bundle.pca_components),
        "preprocessing_method": preprocessing_method,
        "policy_test_metrics": {
            "purity": test_metrics.purity,
            "coverage": test_metrics.coverage,
            "manual_review_rate": test_metrics.manual_review_rate,
            "worst_domain_performance": test_metrics.worst_domain_performance,
            "contamination_rate": test_metrics.contamination_rate,
            "expected_circularity_proxy": test_metrics.expected_circularity_proxy,
            "realized_circularity_proxy": test_metrics.realized_circularity_proxy,
            "expected_utility": test_metrics.expected_utility,
            "realized_utility": test_metrics.realized_utility,
        },
        "selected_policy": policy_json,
        "policy_fit_prediction_source": policy_fit_prediction_source,
        "probability_calibration_prediction_source": probability_calibration_source,
        "risk_control_targets": policy_json["risk_control_targets"],
        "focus_pairs": policy_json.get("focus_pairs", []),
        "composition_aware_summary": {
            "metrics": composition_aware_results.get("metrics"),
            "devices": composition_aware_results.get("devices"),
            "n_train": composition_aware_results.get("n_train"),
            "n_test": composition_aware_results.get("n_test"),
        },
        "open_set_summary": None if open_set_results is None else open_set_results["summary"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Mean NIR spectra per instrument (standard spectroscopy paper figure)
    _spectra_df = bundle["metadata"].copy().reset_index(drop=True)
    _spectra_df["instrument"] = _spectra_df["instrument"].astype(str)
    _spectra_df["wavelength_axis"] = [wavelengths.tolist() for _ in range(len(_spectra_df))]
    _spectra_df["intensity_axis"] = list(X_raw)
    figure_mean_spectra(_spectra_df, output_dir / "publication_outputs")

    publication_outputs = generate_all_figures(
        breakthrough_dir=output_dir,
        circularity_dir=output_dir,
        device_domain_dir=output_dir / "device_domain_experiments",
        pub_output_dir=output_dir / "publication_outputs",
    )

    return {
        "summary": summary,
        "policy_metrics": test_metrics,
        "policy_json": policy_json,
        "pareto_df": pareto_df,
        "robustness_df": robustness_df,
        "domain_metrics_df": domain_results["metrics_df"],
        "focus_pair_summary_df": policy_opt_outputs.get("focus_pair_summary_df"),
        "risk_control_candidate_df": risk_control_outputs["candidate_df"],
        "risk_control_dangerous_pairs_df": risk_control_outputs["dangerous_pairs_df"],
        "composition_aware_results": composition_aware_results,
        "composition_comparison_df": comparison_df,
        "open_set_results": open_set_results,
        "publication_outputs": publication_outputs,
        "output_dir": str(output_dir),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-device reliability + optimization NIR pipeline.")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preprocessing", type=str, default="snv_savgol1")
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--common-grid-points", type=int, default=256)
    parser.add_argument("--min-common-groups-for-eval", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--policy-focus-min-errors", type=int, default=2)
    parser.add_argument("--policy-max-focus-pairs", type=int, default=4)
    parser.add_argument("--policy-entropy-min", type=float, default=0.02)
    parser.add_argument("--policy-entropy-max", type=float, default=0.60)
    parser.add_argument("--risk-target-purity", type=float, default=0.98)
    parser.add_argument("--risk-min-coverage", type=float, default=0.30)
    parser.add_argument("--risk-max-manual-review-rate", type=float, default=0.70)
    parser.add_argument("--risk-min-worst-domain-performance", type=float, default=0.60)
    parser.add_argument("--disable-open-set", action="store_true")
    parser.add_argument("--open-set-n-unknown-classes", type=int, default=2)
    parser.add_argument("--open-set-min-groups-per-unknown-class", type=int, default=3)
    parser.add_argument("--open-set-score-method", type=str, default="entropy")
    parser.add_argument("--open-set-threshold-selection", type=str, default="known_acceptance")
    parser.add_argument("--open-set-target-known-acceptance", type=float, default=0.95)
    parser.add_argument("--quick", action="store_true")
    # Fix #46: expose instrument selection and calibration method on the CLI so
    # users do not have to edit source code to change these critical parameters.
    parser.add_argument(
        "--instruments", type=str, nargs="+", default=list(DEFAULT_DEVICES),
        help="One or more instrument folder names to include, e.g. --instruments Benchtop_N Handheld_F",
    )
    parser.add_argument(
        "--measurement", type=str, default="Reflectance",
        help="Measurement sub-folder to load spectra from (default: Reflectance).",
    )
    parser.add_argument(
        "--calibration-method", type=str, default="isotonic",
        help="Probability calibration method: 'sigmoid' (Platt) or 'isotonic' (default: isotonic).",
    )
    parser.add_argument(
        "--dg-augmented-copies", type=int, default=5,
        help="Number of augmented copies per sample for domain-generalization training (default: 5).",
    )
    parser.add_argument(
        "--min-groups-for-domain-eval", type=int, default=5,
        help=(
            "Minimum groups per class for device-domain transfer experiments (default: 5). "
            "Classes below this threshold are excluded from cross-device evaluation to avoid "
            "degenerate LODO accuracy from insufficient training data. Must be >= "
            "min-common-groups-for-eval; if lower it is silently raised to that value."
        ),
    )
    args = parser.parse_args()

    pop = 24 if args.quick else args.population_size
    gens = 12 if args.quick else args.generations

    results = run_breakthrough_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        instruments=args.instruments,
        measurement=args.measurement,
        calibration_method=args.calibration_method,
        preprocessing_method=args.preprocessing,
        pca_components=args.pca_components,
        alpha=args.alpha,
        common_grid_points=args.common_grid_points,
        min_common_groups_for_eval=args.min_common_groups_for_eval,
        population_size=pop,
        generations=gens,
        policy_focus_min_errors=args.policy_focus_min_errors,
        policy_max_focus_pairs=args.policy_max_focus_pairs,
        policy_entropy_min=args.policy_entropy_min,
        policy_entropy_max=args.policy_entropy_max,
        risk_target_purity=args.risk_target_purity,
        risk_min_coverage=args.risk_min_coverage,
        risk_max_manual_review_rate=args.risk_max_manual_review_rate,
        risk_min_worst_domain_performance=args.risk_min_worst_domain_performance,
        run_open_set=not args.disable_open_set,
        open_set_n_unknown_classes=args.open_set_n_unknown_classes,
        open_set_min_groups_per_unknown_class=args.open_set_min_groups_per_unknown_class,
        open_set_score_method=args.open_set_score_method,
        open_set_threshold_selection=args.open_set_threshold_selection,
        open_set_target_known_acceptance=args.open_set_target_known_acceptance,
        dg_augmented_copies=args.dg_augmented_copies,
        min_groups_for_domain_eval=args.min_groups_for_domain_eval,
    )
    # Summary is saved to <output_dir>/summary.json


if __name__ == "__main__":
    main()
