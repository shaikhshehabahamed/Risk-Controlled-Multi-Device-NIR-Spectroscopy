from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from spectrum_io import parse_and_validate_spectrum_csv, read_csv_spectrum, read_zip_member_spectrum


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "final_model"
DEFAULT_OUTPUT_CSV = DEFAULT_MODEL_DIR / "dataset_predictions.csv"
VALID_MEASUREMENTS = {"Absorbance", "Reflectance"}
INSTRUMENT_DIR_NAMES = {"Benchtop_N", "Handheld_F", "Handheld_S"}


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

def _normalize_dataset_root(dataset_path: str | Path) -> Path:
    dataset_root = _resolve_local_path(dataset_path)

    if dataset_root.is_dir():
        child_dirs = {p.name for p in dataset_root.iterdir() if p.is_dir()}
        if INSTRUMENT_DIR_NAMES.intersection(child_dirs):
            return dataset_root

        nested_root = dataset_root / "NearInfrared"
        if nested_root.is_dir():
            nested_child_dirs = {p.name for p in nested_root.iterdir() if p.is_dir()}
            if INSTRUMENT_DIR_NAMES.intersection(nested_child_dirs):
                return nested_root

    return dataset_root


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


def _coerce_non_empty_string_list(value: Any, *, field_name: str) -> list[str]:
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        values = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError(
            f"model_config.json field '{field_name}' must be a string or a sequence of strings."
        )

    if not values:
        raise ValueError(f"model_config.json field '{field_name}' must not be empty.")
    return values


def _extract_model_class_labels(model) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return []

    values = np.asarray(classes)

    if np.ndim(values) == 0:
        values = [values.item() if hasattr(values, "item") else values]

    normalized = [str(item).strip() for item in values if str(item).strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in normalized:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _validate_prediction_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("model_config.json must contain a JSON object at the top level.")

    dataset_subset = config.get("dataset_subset")
    if not isinstance(dataset_subset, Mapping):
        raise ValueError("model_config.json is missing a valid 'dataset_subset' object.")

    normalized_instruments = _normalize_instrument_selection(
        instrument=_coerce_non_empty_string_list(dataset_subset.get("instrument"), field_name="dataset_subset.instrument"),
        config_instrument=[],
    )
    normalized_measurements = _normalize_measurement_selection(
        measurement=_coerce_non_empty_string_list(dataset_subset.get("measurement"), field_name="dataset_subset.measurement"),
        config_measurement=[],
    )

    preprocessing_value = config.get("preprocessing")
    if not isinstance(preprocessing_value, str) or not preprocessing_value.strip():
        raise ValueError("model_config.json field 'preprocessing' must be a non-empty string.")

    champion_pipeline = config.get("champion_pipeline")
    if not isinstance(champion_pipeline, str) or not champion_pipeline.strip():
        raise ValueError("model_config.json field 'champion_pipeline' must be a non-empty string.")

    eligible_classes: list[str] = []
    class_filter = config.get("class_filter")
    if class_filter is not None:
        if not isinstance(class_filter, Mapping):
            raise ValueError("model_config.json field 'class_filter' must be an object when provided.")
        if "eligible_classes" in class_filter:
            raw_classes = class_filter["eligible_classes"]
            if isinstance(raw_classes, (bytes, bytearray, str)) or not isinstance(raw_classes, Sequence):
                raise ValueError(
                    "model_config.json field 'class_filter.eligible_classes' must be a sequence of class labels when provided."
                )
            eligible_classes = [str(x).strip() for x in raw_classes if str(x).strip()]

    normalized_config = dict(config)
    normalized_config["dataset_subset"] = {
        **dict(dataset_subset),
        "instrument": normalized_instruments,
        "measurement": normalized_measurements,
    }
    normalized_config["preprocessing"] = preprocessing_value.strip()
    normalized_config["champion_pipeline"] = champion_pipeline.strip()

    return {
        "config": normalized_config,
        "eligible_classes": eligible_classes,
    }


def _load_prediction_artifacts(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    model_filename: str = "best_model_full_data.joblib",
) -> dict:
    model_dir = _resolve_local_path(model_dir)

    model_path = _ensure_trusted_project_file(model_dir / model_filename, allowed_suffixes=(".joblib", ".pkl"), purpose="model artifact")
    reference_path = _ensure_trusted_project_file(
        model_dir / "reference_wavelengths.csv",
        allowed_suffixes=(".csv",),
        purpose="reference wavelength artifact",
    )
    config_path = _ensure_trusted_project_file(
        model_dir / "model_config.json",
        allowed_suffixes=(".json",),
        purpose="model configuration artifact",
    )

    model = joblib.load(model_path)

    reference_df = pd.read_csv(reference_path)
    if "wavelength_axis" not in reference_df.columns:
        raise ValueError("reference_wavelengths.csv must contain a 'wavelength_axis' column.")
    ref_wavelengths = reference_df["wavelength_axis"].to_numpy(dtype=float)
    if ref_wavelengths.ndim != 1 or ref_wavelengths.size == 0:
        raise ValueError("reference_wavelengths.csv must contain at least one wavelength value.")
    if not np.isfinite(ref_wavelengths).all():
        raise ValueError("reference_wavelengths.csv contains non-finite wavelength values.")
    if np.any(np.diff(ref_wavelengths) <= 0):
        raise ValueError(
            "reference_wavelengths.csv must contain a strictly increasing 'wavelength_axis' with no duplicates."
        )

    try:
        config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"model_config.json is not valid JSON: {exc}") from exc

    validated = _validate_prediction_config(config_raw)
    model_classes = _extract_model_class_labels(model)
    config_eligible_classes = validated["eligible_classes"]

    warnings: list[str] = []
    if model_classes:
        if config_eligible_classes:
            if set(config_eligible_classes) != set(model_classes):
                warnings.append(
                    "model_config.json class_filter.eligible_classes does not match the fitted model classes; "
                    "prediction QA metrics will use model.classes_ instead."
                )
            effective_eligible_classes = model_classes
        else:
            warnings.append(
                "model_config.json does not include class_filter.eligible_classes; prediction QA metrics "
                "will use model.classes_."
            )
            effective_eligible_classes = model_classes
    else:
        effective_eligible_classes = config_eligible_classes
        if not effective_eligible_classes:
            warnings.append(
                "Neither model.classes_ nor model_config.json class_filter.eligible_classes are available; "
                "in-scope filename QA metrics cannot be computed reliably."
            )

    return {
        "model": model,
        "ref_wavelengths": ref_wavelengths,
        "config": validated["config"],
        "eligible_classes": effective_eligible_classes,
        "config_eligible_classes": config_eligible_classes,
        "model_classes": model_classes,
        "warnings": warnings,
        "model_dir": model_dir,
        "model_filename": model_filename,
    }


def _parse_and_validate_spectrum_csv(csv_source, source_label: str) -> tuple[np.ndarray, np.ndarray]:
    return parse_and_validate_spectrum_csv(csv_source, source_label=source_label)


def _read_csv_spectrum(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    return read_csv_spectrum(csv_path)


def _read_zip_member_spectrum(archive_path: str | Path, member_name: str) -> tuple[np.ndarray, np.ndarray]:
    archive_path = _resolve_local_path(archive_path)
    return read_zip_member_spectrum(archive_path, member_name)


def _safe_decision_output(model, X: np.ndarray) -> tuple[str | None, np.ndarray | None]:
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return "decision_function", np.asarray(scores)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        probs = np.asarray(probs, dtype=float)
        if probs.ndim == 1:
            return "predicted_probability", probs
        if probs.ndim == 2 and probs.shape[1] > 0:
            return "predicted_probability", np.max(probs, axis=1)

    return None, None


def load_single_spectrum(source: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Supported input:
      - standalone CSV path
      - ZIP member path using archive.zip::path/inside/archive.csv
      - ZIP path containing exactly one CSV file
    """
    source_str = str(source)
    if "::" in source_str:
        archive_str, member_name = source_str.split("::", 1)
        if not member_name.strip():
            raise ValueError("ZIP member syntax must include a non-empty member path after '::'.")
        wavelengths, intensities = _read_zip_member_spectrum(archive_str, member_name.strip())
        return wavelengths, intensities, {
            "source_type": "zip_member",
            "archive_path": str(_resolve_local_path(archive_str)),
            "member": member_name.strip(),
        }

    csv_path = _resolve_local_path(source)
    if not csv_path.exists():
        raise FileNotFoundError(f"Spectrum input not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Spectrum input must be a file: {csv_path}")

    if csv_path.suffix.lower() == ".csv":
        wavelengths, intensities = _read_csv_spectrum(csv_path)
        return wavelengths, intensities, {
            "source_type": "csv",
            "archive_path": None,
            "member": None,
        }

    if csv_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(csv_path, "r") as zf:
            csv_members = [name for name in zf.namelist() if (not name.endswith("/")) and name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(
                "ZIP spectrum input must contain exactly one CSV file unless you use archive.zip::member.csv syntax. "
                f"Found {len(csv_members)} CSV file(s) in {csv_path}: {csv_members[:20]}"
            )
        member_name = csv_members[0]
        wavelengths, intensities = _read_zip_member_spectrum(csv_path, member_name)
        return wavelengths, intensities, {
            "source_type": "zip_member",
            "archive_path": str(csv_path),
            "member": member_name,
        }

    raise ValueError(f"Spectrum input must be a CSV file or a ZIP archive: {csv_path}")


def align_to_reference_wavelengths(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    ref_wavelengths: np.ndarray,
) -> tuple[np.ndarray, dict]:
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    ref_wavelengths = np.asarray(ref_wavelengths, dtype=float)

    if wavelengths.ndim != 1 or intensities.ndim != 1:
        raise ValueError("Input wavelengths and intensities must be 1D arrays.")
    if len(wavelengths) != len(intensities):
        raise ValueError("wavelengths and intensities must have the same length.")
    if len(wavelengths) < 2:
        raise ValueError("Spectrum must contain at least 2 points.")

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    intensities = intensities[order]

    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError(
            "Input wavelengths must be strictly increasing after sorting, with no duplicates, before interpolation."
        )

    if np.array_equal(wavelengths, ref_wavelengths):
        return intensities.copy(), {
            "interpolated": False,
            "original_points": int(len(wavelengths)),
            "reference_points": int(len(ref_wavelengths)),
            "source_range": (float(wavelengths.min()), float(wavelengths.max())),
            "reference_range": (float(ref_wavelengths.min()), float(ref_wavelengths.max())),
        }

    src_min, src_max = float(wavelengths.min()), float(wavelengths.max())
    ref_min, ref_max = float(ref_wavelengths.min()), float(ref_wavelengths.max())
    if src_min > ref_min or src_max < ref_max:
        raise ValueError(
            "Input spectrum wavelength range does not fully cover the training reference range. "
            f"Input range: ({src_min}, {src_max}), reference range: ({ref_min}, {ref_max})"
        )

    aligned = np.interp(ref_wavelengths, wavelengths, intensities)
    return aligned, {
        "interpolated": True,
        "original_points": int(len(wavelengths)),
        "reference_points": int(len(ref_wavelengths)),
        "source_range": (src_min, src_max),
        "reference_range": (ref_min, ref_max),
    }


def _parse_filename_metadata(csv_path: Path) -> dict:
    stem = csv_path.stem
    tokens = stem.split("_")

    specimen_group = tokens[0] if len(tokens) > 0 else None
    composition_code = tokens[1] if len(tokens) > 1 else None
    # Fix #24: tokens[-1] equals tokens[0] for single-token filenames and
    # equals composition_code for two-token filenames, making the replicate
    # code meaningless or wrong.  Only assign it when >= 3 tokens are present.
    replicate_code = tokens[-1] if len(tokens) >= 3 else None
    signal_type = str(replicate_code)[-1] if replicate_code else None

    return {
        "filename": csv_path.name,
        "specimen_id_full": stem,
        "specimen_group": specimen_group,
        "composition_code_from_filename": composition_code,
        "replicate_code": replicate_code,
        "signal_type": signal_type,
    }


def _find_dataset_csv_files(
    dataset_path: str | Path,
    instrument: str,
    measurement: str,
) -> list[Path]:
    dataset_root = _normalize_dataset_root(dataset_path)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {dataset_root}")
    if measurement not in VALID_MEASUREMENTS:
        raise ValueError(f"measurement must be one of {sorted(VALID_MEASUREMENTS)}")

    target_root = dataset_root / instrument / measurement
    if not target_root.exists():
        raise FileNotFoundError(f"Expected dataset subset folder not found: {target_root}")

    csv_paths = sorted([p for p in target_root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"])
    if not csv_paths:
        raise ValueError(f"No CSV spectra found under: {target_root}")

    return csv_paths


def _normalize_instrument_selection(
    instrument: str | Sequence[str] | None,
    config_instrument,
) -> list[str]:
    selected = config_instrument if instrument is None else instrument

    if isinstance(selected, str):
        values = [item.strip() for item in selected.split(",") if item.strip()]
    elif isinstance(selected, Sequence):
        values = [str(item).strip() for item in selected if str(item).strip()]
    else:
        raise ValueError(
            "Instrument selection must be a string, a sequence of strings, or None."
        )

    if not values:
        raise ValueError("At least one instrument must be selected for dataset prediction.")

    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduplicated.append(value)
            seen.add(value)

    invalid = [value for value in deduplicated if value not in INSTRUMENT_DIR_NAMES]
    if invalid:
        raise ValueError(
            f"Unsupported instrument(s): {invalid}. Valid options: {sorted(INSTRUMENT_DIR_NAMES)}"
        )

    return deduplicated


def _normalize_measurement_selection(
    measurement: str | Sequence[str] | None,
    config_measurement,
) -> list[str]:
    selected = config_measurement if measurement is None else measurement

    if isinstance(selected, str):
        values = [item.strip() for item in selected.split(",") if item.strip()]
    elif isinstance(selected, Sequence):
        values = [str(item).strip() for item in selected if str(item).strip()]
    else:
        raise ValueError(
            "Measurement selection must be a string, a sequence of strings, or None."
        )

    if not values:
        raise ValueError("At least one measurement must be selected for dataset prediction.")

    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduplicated.append(value)
            seen.add(value)

    invalid = [value for value in deduplicated if value not in VALID_MEASUREMENTS]
    if invalid:
        raise ValueError(
            f"Unsupported measurement(s): {invalid}. Valid options: {sorted(VALID_MEASUREMENTS)}"
        )

    return deduplicated


def _validate_requested_dataset_subset(
    requested_values: Sequence[str],
    trained_values: Sequence[str],
    *,
    field_name: str,
    override_was_provided: bool,
) -> None:
    requested_set = set(str(x).strip() for x in requested_values if str(x).strip())
    trained_set = set(str(x).strip() for x in trained_values if str(x).strip())

    if not requested_set:
        raise ValueError(f"No {field_name} values were selected for prediction.")
    if not trained_set:
        raise ValueError(f"Model configuration does not declare any training-time {field_name} values.")

    incompatible = sorted(requested_set - trained_set)
    if incompatible:
        action = "override" if override_was_provided else "selection"
        raise ValueError(
            f"Requested {field_name} {action} is incompatible with the frozen model. "
            f"Requested: {sorted(requested_set)}. Model was trained on: {sorted(trained_set)}. "
            f"Unsupported values: {incompatible}."
        )




def _apply_required_preprocessing(
    X_aligned: np.ndarray,
    ref_wavelengths: np.ndarray,
    preprocessing_method: str,
    preprocessing_module,
    preprocessing_state: Mapping[str, Any] | None = None,
) -> np.ndarray:
    X_aligned = np.asarray(X_aligned, dtype=float)
    if X_aligned.ndim != 2:
        raise ValueError("X_aligned must be a 2D matrix.")
    return preprocessing_module.preprocess_spectra(
        X_aligned,
        ref_wavelengths,
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )


def predict_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    model_filename: str = "best_model_full_data.joblib",
    instrument: str | Sequence[str] | None = None,
    measurement: str | Sequence[str] | None = None,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    preprocessing_script_path: str | Path = "preprocessing.py",
) -> pd.DataFrame:
    artifacts = _load_prediction_artifacts(model_dir=model_dir, model_filename=model_filename)
    config = artifacts["config"]

    selected_instruments = _normalize_instrument_selection(
        instrument=instrument,
        config_instrument=config["dataset_subset"]["instrument"],
    )
    selected_measurements = _normalize_measurement_selection(
        measurement=measurement,
        config_measurement=config["dataset_subset"]["measurement"],
    )
    _validate_requested_dataset_subset(
        requested_values=selected_instruments,
        trained_values=config["dataset_subset"]["instrument"],
        field_name="instrument",
        override_was_provided=instrument is not None,
    )
    _validate_requested_dataset_subset(
        requested_values=selected_measurements,
        trained_values=config["dataset_subset"]["measurement"],
        field_name="measurement",
        override_was_provided=measurement is not None,
    )
    preprocessing_method = str(config["preprocessing"])
    preprocessing_module = _load_module("preprocessing", preprocessing_script_path)
    preprocessing_state = preprocessing_module.deserialize_preprocessing_state(config.get("preprocessing_state"))

    rows = []
    X_rows = []

    for current_instrument in selected_instruments:
        for current_measurement in selected_measurements:
            csv_paths = _find_dataset_csv_files(
                dataset_path=dataset_path,
                instrument=current_instrument,
                measurement=current_measurement,
            )

            for csv_path in csv_paths:
                wavelengths, intensities, source_info = load_single_spectrum(csv_path)
                aligned_intensities, alignment_info = align_to_reference_wavelengths(
                    wavelengths=wavelengths,
                    intensities=intensities,
                    ref_wavelengths=artifacts["ref_wavelengths"],
                )

                meta = _parse_filename_metadata(csv_path)
                row = {
                    "input_source": str(csv_path.resolve()),
                    "input_type": source_info["source_type"],
                    "member": source_info["member"],
                    "instrument": current_instrument,
                    "measurement": current_measurement,
                    **meta,
                    "alignment_interpolated": bool(alignment_info["interpolated"]),
                    "alignment_original_points": int(alignment_info["original_points"]),
                    "model_file": artifacts["model_filename"],
                    "pipeline": str(config["champion_pipeline"]),
                    "preprocessing_applied": preprocessing_method,
                }

                if "reference_points" in alignment_info:
                    row["alignment_reference_points"] = int(alignment_info["reference_points"])
                if "source_range" in alignment_info:
                    row["alignment_source_range_min"] = float(alignment_info["source_range"][0])
                    row["alignment_source_range_max"] = float(alignment_info["source_range"][1])
                if "reference_range" in alignment_info:
                    row["alignment_reference_range_min"] = float(alignment_info["reference_range"][0])
                    row["alignment_reference_range_max"] = float(alignment_info["reference_range"][1])

                rows.append(row)
                X_rows.append(aligned_intensities)

    if not X_rows:
        raise ValueError(
            "No compatible CSV spectra were found for the requested dataset subset. "
            f"Instruments={selected_instruments}, measurements={selected_measurements}."
        )

    X_aligned = np.vstack(X_rows)
    X_model = _apply_required_preprocessing(
        X_aligned=X_aligned,
        ref_wavelengths=artifacts["ref_wavelengths"],
        preprocessing_method=preprocessing_method,
        preprocessing_module=preprocessing_module,
        preprocessing_state=preprocessing_state,
    )

    model = artifacts["model"]
    preds = model.predict(X_model)
    decision_score_field, decision_scores = _safe_decision_output(model, X_model)

    eligible_class_set = set(artifacts["eligible_classes"])
    in_scope_match_flags = []

    for i, pred in enumerate(preds):
        rows[i]["predicted_label"] = str(pred)
        expected = rows[i]["composition_code_from_filename"]
        in_scope = bool(expected in eligible_class_set)
        rows[i]["expected_label_in_model_classes"] = in_scope

        if in_scope:
            is_match = bool(str(pred) == str(expected))
            rows[i]["prediction_matches_filename_label"] = is_match
            in_scope_match_flags.append(is_match)
        else:
            rows[i]["prediction_matches_filename_label"] = None

        if decision_scores is not None and decision_score_field is not None:
            if np.ndim(decision_scores) == 1:
                rows[i][decision_score_field] = float(decision_scores[i])
            else:
                rows[i][f"{decision_score_field}_max"] = float(np.max(decision_scores[i]))

    df = pd.DataFrame(rows)
    output_csv = _resolve_local_path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    n_in_scope = int(df["expected_label_in_model_classes"].sum())
    n_out_of_scope = int(len(df) - n_in_scope)
    n_in_scope_matches = int(sum(in_scope_match_flags))
    in_scope_match_rate = float(n_in_scope_matches / n_in_scope) if n_in_scope > 0 else None

    summary = {
        "dataset_path": str(_resolve_local_path(dataset_path)),
        "instrument": selected_instruments[0] if len(selected_instruments) == 1 else selected_instruments,
        "measurement": selected_measurements[0] if len(selected_measurements) == 1 else selected_measurements,
        "n_predicted_files": int(len(df)),
        "n_in_scope_files": n_in_scope,
        "n_out_of_scope_files": n_out_of_scope,
        "n_in_scope_filename_matches": n_in_scope_matches,
        "in_scope_filename_match_rate": in_scope_match_rate,
        "output_csv": str(output_csv),
        "pipeline": str(config["champion_pipeline"]),
        "preprocessing_applied": preprocessing_method,
        "model_file": artifacts["model_filename"],
        "eligible_classes_source": (
            "model.classes_" if artifacts["model_classes"] else "model_config.class_filter.eligible_classes"
        ),
        "n_effective_eligible_classes": int(len(eligible_class_set)),
        "warnings": artifacts["warnings"],
    }

    summary_path = output_csv.with_name(output_csv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.attrs["summary"] = summary

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict all compatible CSV spectra in the dataset with the frozen champion model."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="Dataset root directory containing instrument folders such as Benchtop_N, Handheld_F, Handheld_S.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing final_model artifacts.",
    )
    parser.add_argument(
        "--model-filename",
        default="best_model_full_data.joblib",
        help="Model filename inside model_dir.",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Instrument folder(s) to scan. Accepts a single name or a comma-separated list. If omitted, uses model_config.json dataset_subset.instrument.",
    )
    parser.add_argument(
        "--measurement",
        default=None,
        help="Measurement folder(s) to scan. Accepts a single name or a comma-separated list. If omitted, uses model_config.json dataset_subset.measurement.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Where to save the batch prediction CSV.",
    )
    parser.add_argument(
        "--preprocessing-script-path",
        default=str(Path(__file__).resolve().parent / "preprocessing.py"),
        help="Path to preprocessing.py used to apply the exact training-time preprocessing before prediction.",
    )
    args = parser.parse_args()

    predict_dataset(
        dataset_path=args.dataset_path,
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        instrument=args.instrument,
        measurement=args.measurement,
        output_csv=args.output_csv,
        preprocessing_script_path=args.preprocessing_script_path,
    )
    # Predictions saved to <output_csv>; summary saved alongside as <stem>_summary.json
