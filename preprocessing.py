from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import cupy as cp
from cupyx.scipy.signal import savgol_filter as _backend_savgol


def asnumpy(x):
    """Transfer a CuPy array to a NumPy array on the host."""
    return cp.asnumpy(x)


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
STATEFUL_PREPROCESSING_METHODS = {"msc", "msc_savgol1"}
SUPPORTED_PREPROCESSING_METHODS = {
    "raw",
    "snv",
    "msc",
    "savgol1",
    "snv_savgol1",
    "snv_savgol2",
    "msc_savgol1",
}


def _validate_2d_spectra_matrix(X: np.ndarray, *, field_name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{field_name} must be a 2D array. Got shape {X.shape}.")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f"{field_name} must contain at least one sample and one feature. Got shape {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError(f"{field_name} contains non-finite values.")
    return X


def _validate_wavelength_axis(wavelengths: np.ndarray, *, expected_points: int | None = None) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1:
        raise ValueError(f"wavelengths must be a 1D array. Got shape {wavelengths.shape}.")
    if wavelengths.size < 2:
        raise ValueError("wavelengths must contain at least 2 points.")
    if not np.isfinite(wavelengths).all():
        raise ValueError("wavelengths contains non-finite values.")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be strictly increasing and contain no duplicates.")
    if expected_points is not None and wavelengths.size != int(expected_points):
        raise ValueError(
            "The number of spectral features must match the wavelength grid. "
            f"Got expected_points={int(expected_points)} and len(wavelengths)={wavelengths.size}."
        )
    return wavelengths


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



def _load_training_matrix_module(training_matrix_script_path: str | Path):
    training_matrix_script_path = _resolve_local_path(training_matrix_script_path)
    return _load_module_from_path("training_matrix", training_matrix_script_path)



def standard_normal_variate(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X_gpu    = cp.asarray(_validate_2d_spectra_matrix(X), dtype=cp.float64)
    row_mean = X_gpu.mean(axis=1, keepdims=True)
    row_std  = X_gpu.std(axis=1,  keepdims=True)
    row_std  = cp.where(row_std < eps, cp.ones_like(row_std), row_std)
    return asnumpy((X_gpu - row_mean) / row_std)


def multiplicative_scatter_correction(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    X = _validate_2d_spectra_matrix(X)
    if reference is None:
        reference = np.mean(X, axis=0)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if reference.size != X.shape[1]:
        raise ValueError(
            "MSC reference length must match the number of spectral features. "
            f"Got reference length {reference.size} for matrix with {X.shape[1]} features."
        )
    if not np.isfinite(reference).all():
        raise ValueError("MSC reference contains non-finite values.")
    if not np.any(np.abs(reference) > 1e-12):
        raise ValueError(
            "MSC reference vector is effectively zero everywhere; "
            "cannot fit a meaningful scatter correction. "
            "Check that the training spectra contain non-zero intensities."
        )

    X_gpu   = cp.asarray(X, dtype=cp.float64)
    ref_gpu = cp.asarray(reference, dtype=cp.float64)            # (n_features,)
    n_pts   = int(X_gpu.shape[1])

    # Design matrix A  (n_features, 2): [ref | 1]
    A_gpu   = cp.stack([ref_gpu, cp.ones(n_pts, dtype=cp.float64)], axis=1)   # (n_pts, 2)
    # Solve A @ [slope, intercept]^T ≈ X[i]^T  for all i at once
    # Normal equations:  (A^T A) @ coef = A^T @ X^T
    AtA     = A_gpu.T @ A_gpu                                    # (2, 2)
    AtX     = A_gpu.T @ X_gpu.T                                  # (2, n_samples)
    coef    = cp.linalg.solve(AtA, AtX)                          # (2, n_samples)
    slopes  = coef[0]                                            # (n_samples,)
    intercepts = coef[1]                                         # (n_samples,)
    # Guard against near-zero slopes
    slopes  = cp.where(cp.abs(slopes) < 1e-12, cp.ones_like(slopes), slopes)
    out     = (X_gpu - intercepts[None, :].T) / slopes[None, :].T
    return asnumpy(out)



def requires_preprocessing_state(method: str) -> bool:
    method = str(method).strip().lower()
    if method not in SUPPORTED_PREPROCESSING_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    return method in STATEFUL_PREPROCESSING_METHODS



def _normalize_preprocessing_state(preprocessing_state: Mapping[str, Any] | None, method: str) -> dict | None:
    if preprocessing_state is None:
        return None
    if not isinstance(preprocessing_state, Mapping):
        raise ValueError("preprocessing_state must be a mapping when provided.")

    normalized = dict(preprocessing_state)
    state_method = normalized.get("method")
    if state_method is not None and str(state_method) != str(method):
        raise ValueError(
            "preprocessing_state was fit for a different method. "
            f"Requested method={method!r}, state method={state_method!r}."
        )

    reference = normalized.get("msc_reference")
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 1 or reference.size == 0:
            raise ValueError("preprocessing_state['msc_reference'] must be a non-empty 1D vector.")
        reference = reference.reshape(-1)
        if not np.isfinite(reference).all():
            raise ValueError("preprocessing_state['msc_reference'] contains non-finite values.")
        normalized["msc_reference"] = reference
    return normalized



def fit_preprocessing_state(
    X: np.ndarray,
    wavelengths: np.ndarray,
    method: str = "raw",
    sg_window_length: int = 21,
    sg_polyorder: int = 2,
) -> dict:
    X = _validate_2d_spectra_matrix(X)
    wavelengths = _validate_wavelength_axis(wavelengths, expected_points=X.shape[1])
    method = str(method).strip().lower()
    if method not in SUPPORTED_PREPROCESSING_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    state: dict[str, Any] = {
        "method": method,
        "sg_window_length": int(sg_window_length),
        "sg_polyorder": int(sg_polyorder),
        "msc_reference": None,
    }
    if method in STATEFUL_PREPROCESSING_METHODS:
        state["msc_reference"] = np.mean(X, axis=0, dtype=float)
    return state



def serialize_preprocessing_state(preprocessing_state: Mapping[str, Any] | None) -> dict | None:
    normalized = _normalize_preprocessing_state(
        preprocessing_state,
        method=str(preprocessing_state.get("method")) if isinstance(preprocessing_state, Mapping) and preprocessing_state.get("method") is not None else "raw",
    ) if preprocessing_state is not None else None
    if normalized is None:
        return None
    serialized = dict(normalized)
    reference = serialized.get("msc_reference")
    if reference is not None:
        serialized["msc_reference"] = np.asarray(reference, dtype=float).tolist()
    return serialized



def deserialize_preprocessing_state(preprocessing_state: Mapping[str, Any] | None) -> dict | None:
    if preprocessing_state is None:
        return None
    method = preprocessing_state.get("method") if isinstance(preprocessing_state, Mapping) else None
    return _normalize_preprocessing_state(preprocessing_state, method=str(method) if method is not None else "raw")



def savgol_derivative(
    X: np.ndarray,
    wavelengths: np.ndarray,
    window_length: int = 21,
    polyorder: int = 2,
    deriv: int = 1,
) -> np.ndarray:
    X           = _validate_2d_spectra_matrix(X)
    wavelengths = _validate_wavelength_axis(wavelengths, expected_points=X.shape[1])

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder")
    if X.shape[1] < window_length:
        raise ValueError("window_length is larger than number of wavelength points")

    delta  = float(np.median(np.diff(wavelengths)))
    X_gpu  = cp.asarray(X, dtype=cp.float64)
    X_out  = _backend_savgol(
        X_gpu,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        axis=1,
        mode="interp",
    )
    return asnumpy(X_out).astype(float)



def preprocess_spectra(
    X: np.ndarray,
    wavelengths: np.ndarray,
    method: str = "raw",
    sg_window_length: int = 21,
    sg_polyorder: int = 2,
    preprocessing_state: Mapping[str, Any] | None = None,
) -> np.ndarray:
    X = _validate_2d_spectra_matrix(X)
    wavelengths = _validate_wavelength_axis(wavelengths, expected_points=X.shape[1])
    method = str(method).strip().lower()
    if method not in SUPPORTED_PREPROCESSING_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    state = _normalize_preprocessing_state(preprocessing_state, method=method)

    if state is not None:
        if state.get("sg_window_length") is not None:
            sg_window_length = int(state["sg_window_length"])
        if state.get("sg_polyorder") is not None:
            sg_polyorder = int(state["sg_polyorder"])

    if method == "raw":
        return X.copy()
    if method == "snv":
        return standard_normal_variate(X)
    if method == "msc":
        reference = None if state is None else state.get("msc_reference")
        return multiplicative_scatter_correction(X, reference=reference)
    if method == "savgol1":
        return savgol_derivative(X, wavelengths=wavelengths, window_length=sg_window_length, polyorder=sg_polyorder, deriv=1)
    if method == "snv_savgol1":
        return savgol_derivative(standard_normal_variate(X), wavelengths=wavelengths, window_length=sg_window_length, polyorder=sg_polyorder, deriv=1)
    if method == "snv_savgol2":
        return savgol_derivative(standard_normal_variate(X), wavelengths=wavelengths, window_length=sg_window_length, polyorder=sg_polyorder, deriv=2)
    if method == "msc_savgol1":
        reference = None if state is None else state.get("msc_reference")
        return savgol_derivative(multiplicative_scatter_correction(X, reference=reference), wavelengths=wavelengths, window_length=sg_window_length, polyorder=sg_polyorder, deriv=1)

    raise ValueError(f"Unsupported method: {method}")



def fit_and_apply_preprocessing(
    X_fit: np.ndarray,
    X_apply: np.ndarray,
    wavelengths: np.ndarray,
    method: str = "raw",
    sg_window_length: int = 21,
    sg_polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    method = str(method).strip().lower()
    state = fit_preprocessing_state(
        X_fit,
        wavelengths,
        method=method,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
    )
    X_fit_processed = preprocess_spectra(
        X_fit,
        wavelengths,
        method=method,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        preprocessing_state=state,
    )
    X_apply_processed = preprocess_spectra(
        X_apply,
        wavelengths,
        method=method,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        preprocessing_state=state,
    )
    return X_fit_processed, X_apply_processed, state



def summarize_preprocessed_matrix(Xp: np.ndarray, name: str) -> Dict:
    Xp = np.asarray(Xp, dtype=float)
    return {
        "name": name,
        "shape": tuple(Xp.shape),
        "global_min": float(Xp.min()),
        "global_max": float(Xp.max()),
        "global_mean": float(Xp.mean()),
        "global_std": float(Xp.std()),
    }



def build_preprocessing_bundle(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    training_matrix_script_path: str | Path = "training_matrix.py",
    aggregate: Optional[str] = None,
    methods: Optional[list[str]] = None,
    instruments: Optional[Sequence[str]] = None,
    measurement: str = "Reflectance",
    common_classes_only: bool = False,
    common_grid_points: Optional[int] = None,
    fit_full_dataset_methods: bool = False,
):
    training_matrix = _load_training_matrix_module(training_matrix_script_path)

    if instruments is None:
        instruments = ["Benchtop_N"]
    else:
        instruments = list(instruments)

    df = training_matrix.load_nir_subset(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        instruments=instruments,
        measurement=measurement,
    )
    if common_classes_only:
        df, _, _ = training_matrix.keep_common_classes(df, instruments=instruments)
    if common_grid_points is not None:
        target_grid = training_matrix.infer_common_wavelength_grid(
            df,
            instruments=instruments,
            n_points=common_grid_points,
        )
        df = training_matrix.resample_dataframe_to_grid(df, target_grid)

    X, y, y_encoded, groups, wavelengths, metadata, label_map = training_matrix.build_training_matrix(
        df,
        aggregate=aggregate,
    )

    processed: dict[str, np.ndarray] = {}
    preprocessing_states: dict[str, dict | None] = {}
    summaries: list[dict] = []
    requested_methods = [] if methods is None else [str(method).strip().lower() for method in methods]

    # Fix #10: silently writing an empty CSV when no methods are requested is
    # misleading.  Raise instead so callers get immediate feedback.
    if not requested_methods:
        raise ValueError(
            "build_preprocessing_bundle requires at least one method in 'methods'. "
            "Pass e.g. methods=['raw', 'snv', 'snv_savgol1']."
        )

    if not fit_full_dataset_methods:
        blocked_methods = [method for method in requested_methods if requires_preprocessing_state(method)]
        if blocked_methods:
            blocked_str = ", ".join(sorted(blocked_methods))
            raise ValueError(
                "Refusing to fit stateful preprocessing on the full dataset because that can leak evaluation data. "
                f"Blocked method(s): {blocked_str}. Fit preprocessing on the training split and reuse that state for validation/test instead."
            )

    for method in requested_methods:
        state = None
        if fit_full_dataset_methods:
            state = fit_preprocessing_state(X, wavelengths, method=method)
        Xp = preprocess_spectra(X, wavelengths, method=method, preprocessing_state=state)
        processed[method] = Xp
        preprocessing_states[method] = state
        summaries.append(summarize_preprocessed_matrix(Xp, method))

    return {
        "X": X,
        "y": y,
        "y_encoded": y_encoded,
        "groups": groups,
        "wavelengths": wavelengths,
        "metadata": metadata,
        "label_map": label_map,
        "processed": processed,
        "preprocessing_states": preprocessing_states,
        "summary_df": pd.DataFrame(summaries),
    }



def save_preprocessing_bundle(output_dir: str | Path, bundle: dict):
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle["summary_df"].to_csv(output_dir / "preprocessing_summary.csv", index=False)
    bundle["metadata"].to_csv(output_dir / "metadata.csv", index=False)
    bundle["label_map"].to_csv(output_dir / "label_map.csv", index=False)

    np.savez_compressed(
        output_dir / "preprocessing_base_arrays.npz",
        X=bundle["X"],
        y=bundle["y"],
        y_encoded=bundle["y_encoded"],
        groups=bundle["groups"],
        wavelengths=bundle["wavelengths"],
    )

    for method, Xp in bundle.get("processed", {}).items():
        np.savez_compressed(output_dir / f"X_{method}.npz", X=Xp)

    serialized_states = {
        method: serialize_preprocessing_state(state)
        for method, state in bundle.get("preprocessing_states", {}).items()
    }
    with open(output_dir / "preprocessing_states.json", "w", encoding="utf-8") as f:
        json.dump(serialized_states, f, indent=2)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    bundle = build_preprocessing_bundle(
        dataset_path=base_dir,
        loader_script_path=base_dir / "nir_loader.py",
        training_matrix_script_path=base_dir / "training_matrix.py",
        aggregate=None,
        methods=["raw", "snv", "snv_savgol1"],
        fit_full_dataset_methods=False,
    )
    save_preprocessing_bundle(base_dir / "outputs_preprocessing", bundle)
    # All outputs saved to outputs_preprocessing/
