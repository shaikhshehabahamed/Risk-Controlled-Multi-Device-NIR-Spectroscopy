from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import preprocessing
from circularity_proxy import (
    build_structured_target_table,
    fractions_to_composition_code,
    infer_component_order,
)
from group_split_baseline import (
    filter_closed_set_classes,
    make_group_safe_holdout_split,
)
from training_matrix import load_benchtop_reflectance_subset, build_training_matrix


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_composition_aware"


@dataclass
class CompositionAwareMetrics:
    label_accuracy: float
    label_balanced_accuracy: float
    label_macro_f1: float
    dominant_accuracy: float
    mono_material_accuracy: float
    n_components_mae: float
    recoverability_mae: float
    sorting_difficulty_mae: float
    fraction_mae: float
    n_train: int
    n_test: int
    n_classes: int
    n_components: int
    preprocessing_method: str
    selection_space: str


class _ConstantPredictor:
    def __init__(self, value: Any):
        self.value = value

    def predict(self, X: np.ndarray):
        n = int(np.asarray(X).shape[0])
        value = self.value
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                return np.tile(value.reshape(1, -1), (n, 1))
            return np.repeat(value[:1], repeats=n, axis=0)
        return np.repeat(value, repeats=n)


def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


class CompositionAwareModel:
    """
    Structured composition predictor for NIR spectra.

    The model predicts a set of composition-aware targets from the same spectra:
    - mono-material vs blend
    - dominant component
    - number of active components
    - fraction vector over the observed component basis
    - recoverability / sorting-difficulty proxies

    A final closed-set composition label is derived by matching the predicted
    fraction vector to the nearest known training composition code.
    """

    def __init__(
        self,
        *,
        preprocessing_method: str = "snv",
        pca_components: Optional[int] = 20,
        alpha: float = 1.0,
        dominant_solver_max_iter: int = 2000,
        min_fraction_for_code: float = 0.01,
    ) -> None:
        self.preprocessing_method = str(preprocessing_method).strip().lower()
        self.pca_components = None if pca_components is None else int(pca_components)
        self.alpha = float(alpha)
        self.dominant_solver_max_iter = int(dominant_solver_max_iter)
        self.min_fraction_for_code = float(min_fraction_for_code)

    def _preprocess_fit(self, X_raw: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        self.preprocessing_state_ = preprocessing.fit_preprocessing_state(
            X_raw,
            wavelengths,
            method=self.preprocessing_method,
        )
        return preprocessing.preprocess_spectra(
            X_raw,
            wavelengths,
            method=self.preprocessing_method,
            preprocessing_state=self.preprocessing_state_,
        )

    def _preprocess_apply(self, X_raw: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        return preprocessing.preprocess_spectra(
            X_raw,
            wavelengths,
            method=self.preprocessing_method,
            preprocessing_state=self.preprocessing_state_,
        )

    def _fit_feature_stack(self, X_proc: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler()
        X_feat = self.scaler_.fit_transform(np.asarray(X_proc, dtype=float))

        max_components = min(X_feat.shape[0], X_feat.shape[1])
        requested = self.pca_components
        if requested is not None and requested > 0 and max_components >= 2:
            n_components = max(1, min(int(requested), int(max_components)))
            self.pca_ = PCA(n_components=n_components, svd_solver="auto", random_state=42)
            X_feat = self.pca_.fit_transform(X_feat)
            self.n_latent_features_ = int(n_components)
        else:
            self.pca_ = None
            self.n_latent_features_ = int(X_feat.shape[1])
        return X_feat

    def _transform_features(self, X_proc: np.ndarray) -> np.ndarray:
        X_feat = self.scaler_.transform(np.asarray(X_proc, dtype=float))
        if self.pca_ is not None:
            X_feat = self.pca_.transform(X_feat)
        return X_feat

    def _fit_or_constant_classifier(self, X_feat: np.ndarray, y: Sequence[Any], *, binary: bool = False):
        y = pd.Series(y)
        unique = y.dropna().astype(str if not binary else int).unique().tolist()
        if len(unique) == 0:
            # Fix #51: all values are NaN — cannot fit or form a constant predictor.
            raise ValueError(
                "_fit_or_constant_classifier received a target array that is "
                "entirely NaN.  Check that the training DataFrame contains valid labels."
            )
        if len(unique) == 1:
            return _ConstantPredictor(unique[0])

        if binary:
            model = LogisticRegression(
                max_iter=self.dominant_solver_max_iter,
                class_weight="balanced",
                random_state=42,
            )
            model.fit(X_feat, y.astype(int).to_numpy())
            return model

        model = RidgeClassifier(alpha=self.alpha)
        model.fit(X_feat, y.astype(str).to_numpy())
        return model

    def _fit_or_constant_regressor(self, X_feat: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        if y.ndim == 1:
            # Fix #16: y[0] raises IndexError on an empty array.
            if y.size == 0:
                raise ValueError(
                    "_fit_or_constant_regressor received an empty target array. "
                    "Check that the training DataFrame is non-empty."
                )
            if np.allclose(y, y[0]):
                return _ConstantPredictor(float(y[0]))
            model = Ridge(alpha=self.alpha)
            model.fit(X_feat, y.astype(float))
            return model

        if y.ndim == 2:
            if np.allclose(y, y[:1]):
                return _ConstantPredictor(np.asarray(y[:1], dtype=float))
            coefs = []
            intercepts = []
            for j in range(y.shape[1]):
                reg = Ridge(alpha=self.alpha)
                reg.fit(X_feat, y[:, j].astype(float))
                coefs.append(np.asarray(reg.coef_, dtype=float))
                intercepts.append(float(reg.intercept_))
            class _MultiTargetRidge:
                def __init__(self, coefs, intercepts):
                    self.coefs = np.vstack(coefs)
                    self.intercepts = np.asarray(intercepts, dtype=float)
                def predict(self, X):
                    X = np.asarray(X, dtype=float)
                    return X @ self.coefs.T + self.intercepts.reshape(1, -1)
            return _MultiTargetRidge(coefs, intercepts)

        raise ValueError(f"Unsupported target dimensionality: {y.ndim}")

    def fit(self, X_raw: np.ndarray, y: Sequence[str], wavelengths: np.ndarray):
        y = pd.Series(y).astype(str).to_numpy()
        self.component_order_ = tuple(infer_component_order(y))
        if not self.component_order_:
            raise ValueError("No composition components could be inferred from y.")

        target_df = build_structured_target_table(y, component_order=self.component_order_)
        self.fraction_columns_ = [f"fraction_{component}" for component in self.component_order_]
        X_proc = self._preprocess_fit(np.asarray(X_raw, dtype=float), np.asarray(wavelengths, dtype=float))
        X_feat = self._fit_feature_stack(X_proc)

        self.mono_material_model_ = self._fit_or_constant_classifier(X_feat, target_df["mono_material"].to_numpy(), binary=True)
        self.dominant_component_model_ = self._fit_or_constant_classifier(X_feat, target_df["dominant_component"].to_numpy(), binary=False)
        self.n_components_model_ = self._fit_or_constant_regressor(X_feat, target_df["n_nonzero_components"].to_numpy(dtype=float))
        self.fraction_model_ = self._fit_or_constant_regressor(X_feat, target_df[self.fraction_columns_].to_numpy(dtype=float))
        self.recoverability_model_ = self._fit_or_constant_regressor(X_feat, target_df["recoverability_proxy"].to_numpy(dtype=float))
        self.sorting_difficulty_model_ = self._fit_or_constant_regressor(X_feat, target_df["sorting_difficulty_proxy"].to_numpy(dtype=float))

        self.known_class_labels_ = sorted(pd.Series(y).astype(str).unique().tolist())
        class_df = build_structured_target_table(self.known_class_labels_, component_order=self.component_order_)
        self.known_fraction_matrix_ = class_df[self.fraction_columns_].to_numpy(dtype=float)
        self.known_dominant_components_ = class_df["dominant_component"].astype(str).to_numpy(dtype=object)
        self.known_n_components_ = class_df["n_nonzero_components"].to_numpy(dtype=int)
        self.component_to_index_ = {component: idx for idx, component in enumerate(self.component_order_)}
        self.wavelengths_ = np.asarray(wavelengths, dtype=float)
        return self

    def _predict_structured_targets(self, X_raw: np.ndarray, wavelengths: np.ndarray) -> pd.DataFrame:
        X_proc = self._preprocess_apply(np.asarray(X_raw, dtype=float), np.asarray(wavelengths, dtype=float))
        X_feat = self._transform_features(X_proc)

        mono_material = np.asarray(self.mono_material_model_.predict(X_feat)).reshape(-1)
        mono_material = (mono_material.astype(float) >= 0.5).astype(int)
        dominant_component = pd.Series(self.dominant_component_model_.predict(X_feat)).astype(str).to_numpy()
        n_components = np.asarray(self.n_components_model_.predict(X_feat), dtype=float).reshape(-1)
        n_components = np.clip(np.rint(n_components).astype(int), 1, len(self.component_order_))

        fraction_pred = np.asarray(self.fraction_model_.predict(X_feat), dtype=float)
        if fraction_pred.ndim == 1:
            fraction_pred = fraction_pred.reshape(-1, len(self.component_order_))
        fraction_pred = np.clip(fraction_pred, 0.0, None)

        recoverability = np.asarray(self.recoverability_model_.predict(X_feat), dtype=float).reshape(-1)
        sorting_difficulty = np.asarray(self.sorting_difficulty_model_.predict(X_feat), dtype=float).reshape(-1)

        rows = []
        for i in range(len(X_feat)):
            fractions = np.asarray(fraction_pred[i], dtype=float).copy()
            if not np.isfinite(fractions).all() or float(fractions.sum()) <= 0.0:
                fractions = np.ones(len(self.component_order_), dtype=float)

            dominant = str(dominant_component[i]).lower()
            dominant_idx = self.component_to_index_.get(dominant)
            if dominant_idx is None:
                dominant_idx = int(np.argmax(fractions))
                dominant = self.component_order_[dominant_idx]
            fractions[dominant_idx] = max(float(fractions[dominant_idx]), float(fractions.max()))

            if int(mono_material[i]) == 1:
                active_idx = np.array([dominant_idx], dtype=int)
            else:
                requested_k = int(max(2, n_components[i]))
                requested_k = min(requested_k, len(self.component_order_))
                active_idx = np.argsort(fractions)[-requested_k:]
                if dominant_idx not in set(active_idx.tolist()):
                    active_idx[0] = dominant_idx
            mask = np.zeros(len(self.component_order_), dtype=bool)
            mask[active_idx] = True
            fractions = np.where(mask, fractions, 0.0)
            if float(fractions.sum()) <= 0.0:
                fractions[dominant_idx] = 1.0
            fractions = fractions / float(fractions.sum())

            structured_code = fractions_to_composition_code(
                fractions,
                component_order=self.component_order_,
                min_fraction=self.min_fraction_for_code,
                drop_zeros=True,
            )
            distances = np.abs(self.known_fraction_matrix_ - fractions.reshape(1, -1)).sum(axis=1)
            dominant_match_bonus = (self.known_dominant_components_ != dominant).astype(float) * 0.05
            n_component_penalty = np.abs(self.known_n_components_ - int(mask.sum())).astype(float) * 0.02
            best_idx = int(np.argmin(distances + dominant_match_bonus + n_component_penalty))

            row = {
                "predicted_label": str(self.known_class_labels_[best_idx]),
                "predicted_structured_code": structured_code,
                "predicted_mono_material": int(mono_material[i]),
                "predicted_dominant_component": dominant,
                "predicted_n_nonzero_components": int(mask.sum()),
                "predicted_recoverability_proxy": float(np.clip(recoverability[i], 0.0, 1.0)),
                "predicted_sorting_difficulty_proxy": float(np.clip(sorting_difficulty[i], 0.0, 1.0)),
                "nearest_known_label_l1_distance": float(distances[best_idx]),
            }
            for j, component in enumerate(self.component_order_):
                row[f"predicted_fraction_{component}"] = float(fractions[j])
            rows.append(row)

        return pd.DataFrame(rows)

    def predict(self, X_raw: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        return self._predict_structured_targets(X_raw, wavelengths)["predicted_label"].astype(str).to_numpy()

    def predict_structured(self, X_raw: np.ndarray, wavelengths: np.ndarray) -> pd.DataFrame:
        return self._predict_structured_targets(X_raw, wavelengths)



def evaluate_composition_aware_holdout(
    *,
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: Sequence[str],
    y_test: Sequence[str],
    wavelengths: np.ndarray,
    preprocessing_method: str = "snv",
    pca_components: Optional[int] = 20,
    alpha: float = 1.0,
) -> dict:
    model = CompositionAwareModel(
        preprocessing_method=preprocessing_method,
        pca_components=pca_components,
        alpha=alpha,
    )
    model.fit(X_train_raw, y_train, wavelengths)
    prediction_df = model.predict_structured(X_test_raw, wavelengths)

    y_true = pd.Series(y_test).astype(str).to_numpy()
    y_pred = prediction_df["predicted_label"].astype(str).to_numpy()
    target_df = build_structured_target_table(y_true, component_order=model.component_order_)

    predicted_fraction_cols = [f"predicted_fraction_{component}" for component in model.component_order_]
    fraction_true = target_df[[f"fraction_{component}" for component in model.component_order_]].to_numpy(dtype=float)
    fraction_pred = prediction_df[predicted_fraction_cols].to_numpy(dtype=float)

    metrics = CompositionAwareMetrics(
        label_accuracy=float(accuracy_score(y_true, y_pred)),
        label_balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        label_macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        dominant_accuracy=float(np.mean(
            prediction_df["predicted_dominant_component"].astype(str).to_numpy()
            == target_df["dominant_component"].astype(str).to_numpy()
        )),
        mono_material_accuracy=float(np.mean(
            prediction_df["predicted_mono_material"].to_numpy(dtype=int)
            == target_df["mono_material"].to_numpy(dtype=int)
        )),
        n_components_mae=float(mean_absolute_error(
            target_df["n_nonzero_components"].to_numpy(dtype=float),
            prediction_df["predicted_n_nonzero_components"].to_numpy(dtype=float),
        )),
        recoverability_mae=float(mean_absolute_error(
            target_df["recoverability_proxy"].to_numpy(dtype=float),
            prediction_df["predicted_recoverability_proxy"].to_numpy(dtype=float),
        )),
        sorting_difficulty_mae=float(mean_absolute_error(
            target_df["sorting_difficulty_proxy"].to_numpy(dtype=float),
            prediction_df["predicted_sorting_difficulty_proxy"].to_numpy(dtype=float),
        )),
        fraction_mae=float(np.mean(np.abs(fraction_true - fraction_pred))),
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        n_classes=int(pd.Series(np.concatenate([pd.Series(y_train).astype(str).to_numpy(), y_true])).nunique()),
        n_components=int(len(model.component_order_)),
        preprocessing_method=str(preprocessing_method),
        selection_space="closed_set_nearest_known_composition",
    )

    prediction_out = pd.concat([
        target_df.add_prefix("true_"),
        prediction_df.reset_index(drop=True),
    ], axis=1)

    return {
        "model": model,
        "prediction_df": prediction_out,
        "metrics": asdict(metrics),
        "component_order": list(model.component_order_),
    }



def run_composition_aware_experiment(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    min_groups_per_class: int = 3,
    preprocessing_method: str = "snv",
    pca_components: Optional[int] = 20,
    alpha: float = 1.0,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
) -> dict:
    # Fix #17: pass loader_script_path through instead of silently deleting it.
    from training_matrix import load_nir_subset  # noqa: PLC0415
    df = load_nir_subset(
        dataset_path=_resolve_local_path(dataset_path),
        loader_script_path=loader_script_path,
        instruments=["Benchtop_N"],
        measurement="Reflectance",
    )
    X_full, y_full, _, groups_full, wavelengths, metadata, _ = build_training_matrix(df, aggregate=None)

    mask, eligible_classes, class_group_summary = filter_closed_set_classes(
        metadata=metadata,
        target_col="composition_code",
        group_col="specimen_group",
        min_groups_per_class=min_groups_per_class,
    )

    X_f = np.asarray(X_full[mask], dtype=float)
    y_f = pd.Series(y_full[mask]).astype(str).to_numpy()
    groups_f = pd.Series(groups_full[mask]).astype(str).to_numpy()
    metadata_f = metadata.loc[mask].reset_index(drop=True)

    train_idx, test_idx = make_group_safe_holdout_split(
        y=y_f,
        groups=groups_f,
        n_splits=2,
        random_state=42,
    )

    outputs = evaluate_composition_aware_holdout(
        X_train_raw=X_f[train_idx],
        X_test_raw=X_f[test_idx],
        y_train=y_f[train_idx],
        y_test=y_f[test_idx],
        wavelengths=np.asarray(wavelengths, dtype=float),
        preprocessing_method=preprocessing_method,
        pca_components=pca_components,
        alpha=alpha,
    )
    prediction_df = pd.concat([metadata_f.iloc[test_idx].reset_index(drop=True), outputs["prediction_df"]], axis=1)

    result = {
        "metrics": outputs["metrics"],
        "component_order": outputs["component_order"],
        "eligible_classes": eligible_classes,
        "n_eligible_classes": int(len(eligible_classes)),
        "n_rows_after_filter": int(mask.sum()),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "prediction_df": prediction_df,
        "class_group_summary": class_group_summary,
    }

    if output_dir is not None:
        output_dir = _resolve_local_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(output_dir / "composition_aware_predictions.csv", index=False)
        class_group_summary.to_csv(output_dir / "class_group_summary.csv", index=False)
        (output_dir / "composition_aware_metrics.json").write_text(
            json.dumps({k: v for k, v in result.items() if k not in {"prediction_df", "class_group_summary"}}, indent=2),
            encoding="utf-8",
        )
        result["output_dir"] = str(output_dir)

    return result



def _safe_domain_metrics(y_true: np.ndarray, y_pred: np.ndarray, domains: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "domain": pd.Series(domains).astype(str).to_numpy(),
        "y_true": pd.Series(y_true).astype(str).to_numpy(),
        "y_pred": pd.Series(y_pred).astype(str).to_numpy(),
    })
    rows = []
    for domain, sub in df.groupby("domain"):
        rows.append({
            "domain": str(domain),
            "n_samples": int(len(sub)),
            "label_accuracy": float(accuracy_score(sub["y_true"], sub["y_pred"])) if len(sub) else 0.0,
            "label_balanced_accuracy": float(balanced_accuracy_score(sub["y_true"], sub["y_pred"])) if len(sub) else 0.0,
            "label_macro_f1": float(f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)) if len(sub) else 0.0,
        })
    return pd.DataFrame(rows)


def run_composition_aware_multidevice_experiment(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    *,
    instruments: Sequence[str] = ("Benchtop_N", "Handheld_F", "Handheld_S"),
    measurement: str = "Reflectance",
    preprocessing_method: str = "snv",
    pca_components: Optional[int] = 20,
    alpha: float = 1.0,
    common_grid_points: int = 128,
    min_common_groups_for_eval: int = 5,
    bundle: Mapping[str, Any] | None = None,
    split_info: Mapping[str, Any] | None = None,
) -> dict:
    from device_domain_experiments import build_multidevice_dataset, make_nested_group_holdout_splits

    if bundle is None:
        bundle = build_multidevice_dataset(
            dataset_path=_resolve_local_path(dataset_path),
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

    X_raw = np.asarray(bundle["X"], dtype=float)
    y = pd.Series(bundle["y"]).astype(str).to_numpy()
    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)

    fit_groups = set(pd.Series(split_info["fit_groups"]).astype(str).tolist())
    test_groups = set(pd.Series(split_info["test_common_groups"]).astype(str).tolist())
    fit_mask = metadata["specimen_group"].astype(str).isin(fit_groups).to_numpy()
    test_mask = metadata["specimen_group"].astype(str).isin(test_groups).to_numpy()
    if not bool(fit_mask.sum()) or not bool(test_mask.sum()):
        raise ValueError("Composition-aware multi-device experiment created an empty fit or test split.")

    outputs = evaluate_composition_aware_holdout(
        X_train_raw=X_raw[fit_mask],
        X_test_raw=X_raw[test_mask],
        y_train=y[fit_mask],
        y_test=y[test_mask],
        wavelengths=wavelengths,
        preprocessing_method=preprocessing_method,
        pca_components=pca_components,
        alpha=alpha,
    )

    prediction_df = pd.concat([
        metadata.loc[test_mask].reset_index(drop=True),
        outputs["prediction_df"].reset_index(drop=True),
    ], axis=1)
    domain_metrics_df = _safe_domain_metrics(
        prediction_df["true_composition_code"].astype(str).to_numpy(),
        prediction_df["predicted_label"].astype(str).to_numpy(),
        prediction_df["instrument"].astype(str).to_numpy(),
    )

    result = {
        "metrics": outputs["metrics"],
        "component_order": outputs["component_order"],
        "prediction_df": prediction_df,
        "domain_metrics_df": domain_metrics_df,
        "n_train": int(fit_mask.sum()),
        "n_test": int(test_mask.sum()),
        "devices": list(instruments),
        "measurement": str(measurement),
    }

    if output_dir is not None:
        outdir = _resolve_local_path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(outdir / "composition_aware_predictions.csv", index=False)
        domain_metrics_df.to_csv(outdir / "composition_aware_domain_metrics.csv", index=False)
        (outdir / "composition_aware_metrics.json").write_text(
            json.dumps({k: v for k, v in result.items() if k not in {"prediction_df", "domain_metrics_df"}}, indent=2),
            encoding="utf-8",
        )
        result["output_dir"] = str(outdir)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grouped composition-aware NIR experiment on the benchtop reflectance subset.")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-groups-per-class", type=int, default=3)
    parser.add_argument("--preprocessing-method", type=str, default="snv")
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    run_composition_aware_experiment(
        dataset_path=args.dataset_path,
        min_groups_per_class=args.min_groups_per_class,
        preprocessing_method=args.preprocessing_method,
        pca_components=args.pca_components,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )
    # Summary saved to <output_dir>/composition_aware_metrics.json


if __name__ == "__main__":
    main()
