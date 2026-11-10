from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

import cupy as cp


def asnumpy(x):
    """Transfer a CuPy array to a NumPy array on the host."""
    return cp.asnumpy(x)

import preprocessing
from reliability import (
    CalibratedModelBundle,
    apply_abstention_policy,
    evaluate_policy,
    normalize_abstention_policy,
    predict_with_uncertainty,
)



def add_gaussian_noise(X: np.ndarray, std_fraction: float, rng: np.random.Generator) -> np.ndarray:
    X_gpu = cp.asarray(np.asarray(X, dtype=float), dtype=cp.float64)
    sigma = float(std_fraction) * cp.std(X_gpu, axis=1, keepdims=True)
    sigma = cp.where(sigma < 1e-12, cp.full_like(sigma, 1e-12), sigma)
    noise = cp.asarray(rng.normal(0.0, 1.0, size=X_gpu.shape), dtype=cp.float64)
    return asnumpy(X_gpu + sigma * noise)



def add_baseline_drift(X: np.ndarray, wavelengths: np.ndarray, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=float)
    wl_norm = (wl - wl.min()) / (wl.max() - wl.min() + 1e-12)
    slopes = rng.uniform(-magnitude, magnitude, size=(X.shape[0], 1))
    intercepts = rng.uniform(-magnitude, magnitude, size=(X.shape[0], 1))
    X_gpu = cp.asarray(np.asarray(X, dtype=float), dtype=cp.float64)
    wl_gpu = cp.asarray(wl_norm, dtype=cp.float64).reshape(1, -1)
    sl_gpu = cp.asarray(slopes, dtype=cp.float64)
    ic_gpu = cp.asarray(intercepts, dtype=cp.float64)
    drift = ic_gpu + sl_gpu * wl_gpu
    return asnumpy(X_gpu + drift)



def add_multiplicative_scatter(X: np.ndarray, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    factors = rng.uniform(1.0 - magnitude, 1.0 + magnitude, size=(X.shape[0], 1))
    X_gpu = cp.asarray(np.asarray(X, dtype=float), dtype=cp.float64)
    f_gpu = cp.asarray(factors, dtype=cp.float64)
    return asnumpy(X_gpu * f_gpu)



def wavelength_shift_resample(X: np.ndarray, wavelengths: np.ndarray, shift_steps: int) -> np.ndarray:
    """
    Shift the source wavelength grid by `shift_steps * Δλ` then interpolate
    back onto the original grid using the configured array backend.
    """
    wl = np.asarray(wavelengths, dtype=float)
    step = float(np.median(np.diff(wl)))
    shifted = wl + shift_steps * step

    X_gpu = cp.asarray(np.asarray(X, dtype=float), dtype=cp.float64)
    xp = cp.asarray(shifted, dtype=cp.float64)
    xn = cp.asarray(wl, dtype=cp.float64)

    idx = cp.searchsorted(xp, xn, side="right")
    idx = cp.clip(idx, 1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    dx = x1 - x0
    t = (xn - x0) / cp.where(dx < 1e-15, cp.ones_like(dx), dx)

    f0 = X_gpu[:, idx - 1]
    f1 = X_gpu[:, idx]
    out = f0 + (f1 - f0) * t
    left_mask = xn < xp[0]
    right_mask = xn > xp[-1]
    out[:, left_mask] = X_gpu[:, 0:1]
    out[:, right_mask] = X_gpu[:, -1:]
    return asnumpy(out)



def evaluate_robustness_suite(
    model_bundle: CalibratedModelBundle,
    X_test_raw: np.ndarray,
    y_test: Sequence[str],
    domains: Sequence[str],
    wavelengths: np.ndarray,
    thresholds_by_domain: Dict[str, float] | None = None,
    min_margin: float | None = None,
    policy: Mapping[str, Any] | None = None,
    preprocessing_method: str = "snv",
    random_state: int = 42,
) -> pd.DataFrame:
    # For stateful preprocessing methods (msc, msc_savgol1) the training
    # reference vector must be carried inside the bundle. Applying the
    # method with preprocessing_state=None would silently recompute the
    # reference from each perturbed test batch, producing invalid metrics.
    _STATEFUL_METHODS = {"msc", "msc_savgol1"}
    if preprocessing_method.strip().lower() in _STATEFUL_METHODS:
        if model_bundle.preprocessing_state is None:
            raise ValueError(
                f"preprocessing_method={preprocessing_method!r} requires a fitted "
                "preprocessing_state stored in the model bundle, but "
                "CalibratedModelBundle.preprocessing_state is None. "
                "Re-fit the bundle with preprocessing_state set to the state "
                "returned by preprocessing.fit_preprocessing_state() on the "
                "training split."
            )

    if policy is None:
        policy = {
            "thresholds_by_domain": thresholds_by_domain or {},
            "min_margin": 0.0 if min_margin is None else float(min_margin),
        }
    normalized_policy = normalize_abstention_policy(policy)

    rng = np.random.default_rng(random_state)
    y_test = pd.Series(y_test).astype(str).to_numpy()
    domains = pd.Series(domains).astype(str).to_numpy()
    wavelengths = np.asarray(wavelengths, dtype=float)

    perturbations = {
        "clean": X_test_raw,
        "gaussian_noise_0.005": add_gaussian_noise(X_test_raw, 0.005, rng),
        "gaussian_noise_0.010": add_gaussian_noise(X_test_raw, 0.010, rng),
        "baseline_drift_0.02": add_baseline_drift(X_test_raw, wavelengths, 0.02, rng),
        "scatter_0.03": add_multiplicative_scatter(X_test_raw, 0.03, rng),
        "wavelength_shift_plus1": wavelength_shift_resample(X_test_raw, wavelengths, 1),
        "wavelength_shift_minus1": wavelength_shift_resample(X_test_raw, wavelengths, -1),
    }

    rows = []
    for name, Xp_raw in perturbations.items():
        Xp = preprocessing.preprocess_spectra(
            Xp_raw,
            wavelengths,
            method=preprocessing_method,
            preprocessing_state=model_bundle.preprocessing_state,
        )
        pred_df = predict_with_uncertainty(model_bundle, Xp)
        accepted_mask = apply_abstention_policy(
            prediction_df=pred_df,
            domains=domains,
            policy=normalized_policy,
        )
        metrics = evaluate_policy(
            y_true=y_test,
            domains=domains,
            prediction_df=pred_df,
            accepted_mask=accepted_mask,
            thresholds_by_domain=normalized_policy["thresholds_by_domain"],
            min_margin=normalized_policy["min_margin"],
        )
        rows.append(
            {
                "perturbation": name,
                "purity": metrics.purity,
                "coverage": metrics.coverage,
                "manual_review_rate": metrics.manual_review_rate,
                "worst_domain_performance": metrics.worst_domain_performance,
                "accepted_accuracy": metrics.accepted_accuracy,
                "accepted_balanced_accuracy": metrics.accepted_balanced_accuracy,
                "accepted_macro_f1": metrics.accepted_macro_f1,
                "expected_circularity_proxy": metrics.expected_circularity_proxy,
                "realized_circularity_proxy": metrics.realized_circularity_proxy,
                "n_accepted": metrics.n_accepted,
            }
        )

    return pd.DataFrame(rows)
