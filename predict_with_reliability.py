from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import nir_loader
import preprocessing
import training_matrix
from reliability import apply_abstention_policy, normalize_abstention_policy, predict_with_uncertainty


# reliability.py writes: outputs_reliability/calibrated_reliability_model.joblib
# run_breakthrough_pipeline.py writes: outputs_breakthrough_study/calibrated_reliable_model.joblib
# Default here targets the reliability study output; pass an explicit path for the
# breakthrough pipeline model.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "outputs_reliability" / "calibrated_reliability_model.joblib"
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "outputs_reliability" / "reliable_predictions.csv"


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
            "Only trusted project-local model artifacts are allowed."
        ) from exc
    return path


def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()

def load_model_artifacts(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    model_path = _ensure_trusted_project_file(
        model_path,
        allowed_suffixes=(".joblib", ".pkl"),
        purpose="reliable model bundle",
    )
    return joblib.load(model_path)


def _normalize_artifact_subset_value(value, *, field_name: str) -> list[str]:
    if isinstance(value, str):
        values = [value.strip()] if value.strip() else []
    elif isinstance(value, (list, tuple, set)):
        values = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        values = []
    else:
        raise ValueError(f"Reliable model artifact field '{field_name}' must be a string or list of strings.")
    return values


def _validate_requested_subset(artifacts: dict, *, instrument: str, measurement: str) -> None:
    dataset_subset = artifacts.get("dataset_subset")
    if dataset_subset is None:
        return
    if not isinstance(dataset_subset, dict):
        raise ValueError("Reliable model artifact field 'dataset_subset' must be a mapping when present.")

    trained_instruments = _normalize_artifact_subset_value(dataset_subset.get("instrument"), field_name="dataset_subset.instrument")
    trained_measurements = _normalize_artifact_subset_value(dataset_subset.get("measurement"), field_name="dataset_subset.measurement")

    if trained_instruments and instrument not in set(trained_instruments):
        raise ValueError(
            "Requested instrument is incompatible with the reliable model. "
            f"Requested: {instrument!r}. Model was built for: {sorted(set(trained_instruments))}."
        )
    if trained_measurements and measurement not in set(trained_measurements):
        raise ValueError(
            "Requested measurement is incompatible with the reliable model. "
            f"Requested: {measurement!r}. Model was built for: {sorted(set(trained_measurements))}."
        )


def predict_dataset_with_reliability(
    dataset_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    instrument: str = "Benchtop_N",
    measurement: str = "Reflectance",
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    dataset_path = _resolve_local_path(dataset_path)
    model_path = _resolve_local_path(model_path)
    output_csv = _resolve_local_path(output_csv)

    artifacts = load_model_artifacts(model_path)
    _validate_requested_subset(artifacts, instrument=instrument, measurement=measurement)
    policy = normalize_abstention_policy(artifacts["policy"])
    wavelengths_ref = np.asarray(artifacts["wavelengths"], dtype=float)
    preprocessing_method = str(artifacts["preprocessing_method"])
    preprocessing_state = preprocessing.deserialize_preprocessing_state(artifacts.get("preprocessing_state"))

    df = nir_loader.load_nir_spectra(
        dataset_path=dataset_path,
        instrument_filter=[instrument],
        measurement_filter=[measurement],
    )
    df = training_matrix.resample_dataframe_to_grid(df, wavelengths_ref)
    X = np.vstack(df["intensity_axis"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    X = preprocessing.preprocess_spectra(
        X,
        wavelengths_ref,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )

    pred_df = predict_with_uncertainty(artifacts["calibrated_model"], X)
    accepted = apply_abstention_policy(
        prediction_df=pred_df,
        domains=df["instrument"].astype(str).to_numpy(),
        policy=policy,
    )
    out = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    out["accepted_for_auto_sorting"] = accepted.astype(int)
    out["requires_manual_review"] = (~accepted).astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    summary = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "instrument": instrument,
        "measurement": measurement,
        "n_rows": int(len(out)),
        "n_accepted": int(accepted.sum()),
        "coverage": float(np.mean(accepted)) if len(accepted) else 0.0,
        "output_csv": str(output_csv.resolve()),
    }
    output_csv.with_name(output_csv.stem + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a dataset subset with the calibrated abstaining model.")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--instrument", type=str, default="Benchtop_N")
    parser.add_argument("--measurement", type=str, default="Reflectance")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()
    predict_dataset_with_reliability(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        instrument=args.instrument,
        measurement=args.measurement,
        output_csv=args.output_csv,
    )
    # Predictions saved to <output_csv>; summary saved alongside as <stem>_summary.json
