from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import preprocessing
from reliability import fit_calibrated_model
from robustness_analysis import (
    add_baseline_drift,
    add_gaussian_noise,
    add_multiplicative_scatter,
    wavelength_shift_resample,
)


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DomainGeneralizationConfig:
    n_augmented_copies: int = 5
    gaussian_noise_std: float = 0.0075
    baseline_drift_magnitude: float = 0.02
    scatter_magnitude: float = 0.03
    wavelength_shift_choices: tuple[int, ...] = (-1, 0, 1)
    random_state: int = 42


DEFAULT_DG_CONFIG = DomainGeneralizationConfig()


def _normalize_config(config: DomainGeneralizationConfig | Mapping[str, Any] | None) -> DomainGeneralizationConfig:
    if config is None:
        return DEFAULT_DG_CONFIG
    if isinstance(config, DomainGeneralizationConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be a DomainGeneralizationConfig, mapping, or None.")
    normalized = dict(config)
    if "wavelength_shift_choices" in normalized:
        normalized["wavelength_shift_choices"] = tuple(int(v) for v in normalized["wavelength_shift_choices"])
    return DomainGeneralizationConfig(**normalized)


def config_to_dict(config: DomainGeneralizationConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = _normalize_config(config)
    out = asdict(normalized)
    out["wavelength_shift_choices"] = list(normalized.wavelength_shift_choices)
    return out


def _apply_per_sample_wavelength_shift(
    X_raw: np.ndarray,
    wavelengths: np.ndarray,
    shift_steps: np.ndarray,
) -> np.ndarray:
    X_raw = np.asarray(X_raw, dtype=float)
    shift_steps = np.asarray(shift_steps, dtype=int).reshape(-1)
    if X_raw.ndim != 2:
        raise ValueError(f"X_raw must be a 2D array. Got shape {X_raw.shape}.")
    if X_raw.shape[0] != len(shift_steps):
        raise ValueError(
            "shift_steps length must match the number of spectra. "
            f"Got len(shift_steps)={len(shift_steps)} and n_samples={X_raw.shape[0]}."
        )

    out = np.empty_like(X_raw, dtype=float)
    unique_steps = sorted({int(step) for step in shift_steps.tolist()})
    for step in unique_steps:
        mask = shift_steps == int(step)
        if not np.any(mask):
            continue
        if int(step) == 0:
            out[mask] = X_raw[mask]
        else:
            out[mask] = wavelength_shift_resample(X_raw[mask], wavelengths, int(step))
    return out


def augment_training_spectra(
    X_raw: np.ndarray,
    y: Sequence[str],
    wavelengths: np.ndarray,
    *,
    domains: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    config: DomainGeneralizationConfig | Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    cfg = _normalize_config(config)
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y).astype(str)
    wavelengths = np.asarray(wavelengths, dtype=float)

    if X_raw.ndim != 2:
        raise ValueError(f"X_raw must be a 2D array. Got shape {X_raw.shape}.")
    if len(X_raw) != len(y):
        raise ValueError(f"X_raw and y must have the same length. Got {len(X_raw)} and {len(y)}.")

    domains_arr = None if domains is None else np.asarray(domains).astype(str)
    groups_arr = None if groups is None else np.asarray(groups).astype(str)
    if domains_arr is not None and len(domains_arr) != len(X_raw):
        raise ValueError("domains must have the same length as X_raw when provided.")
    if groups_arr is not None and len(groups_arr) != len(X_raw):
        raise ValueError("groups must have the same length as X_raw when provided.")

    rng = np.random.default_rng(int(cfg.random_state))

    X_chunks: list[np.ndarray] = [X_raw]
    y_chunks: list[np.ndarray] = [y]
    domain_chunks: list[np.ndarray] | None = [domains_arr] if domains_arr is not None else None
    group_chunks: list[np.ndarray] | None = [groups_arr] if groups_arr is not None else None

    for _ in range(int(cfg.n_augmented_copies)):
        Xa = np.asarray(X_raw, dtype=float).copy()
        if float(cfg.scatter_magnitude) > 0.0:
            Xa = add_multiplicative_scatter(Xa, float(cfg.scatter_magnitude), rng)
        if float(cfg.baseline_drift_magnitude) > 0.0:
            Xa = add_baseline_drift(Xa, wavelengths, float(cfg.baseline_drift_magnitude), rng)
        if float(cfg.gaussian_noise_std) > 0.0:
            Xa = add_gaussian_noise(Xa, float(cfg.gaussian_noise_std), rng)
        if cfg.wavelength_shift_choices:
            steps = rng.choice(np.asarray(cfg.wavelength_shift_choices, dtype=int), size=len(Xa), replace=True)
            Xa = _apply_per_sample_wavelength_shift(Xa, wavelengths, steps)

        X_chunks.append(np.asarray(Xa, dtype=float))
        y_chunks.append(y)
        if domain_chunks is not None and domains_arr is not None:
            domain_chunks.append(domains_arr)
        if group_chunks is not None and groups_arr is not None:
            group_chunks.append(groups_arr)

    X_out = np.vstack(X_chunks)
    y_out = np.concatenate(y_chunks)
    permutation = rng.permutation(len(X_out))

    out: dict[str, np.ndarray] = {
        "X_raw": X_out[permutation],
        "y": y_out[permutation],
    }
    if domain_chunks is not None:
        out["domains"] = np.concatenate(domain_chunks)[permutation]
    if group_chunks is not None:
        out["groups"] = np.concatenate(group_chunks)[permutation]
    return out


def fit_domain_generalized_calibrated_model(
    X_train_raw: np.ndarray,
    y_train: Sequence[str],
    wavelengths: np.ndarray,
    *,
    train_domains: Sequence[str] | None = None,
    train_groups: Sequence[str] | None = None,
    X_cal_raw: np.ndarray | None = None,
    y_cal: Sequence[str] | None = None,
    preprocessing_method: str = "snv",
    n_components: int = 20,
    alpha: float = 1.0,
    calibration_method: str = "sigmoid",
    config: DomainGeneralizationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _normalize_config(config)
    augmented = augment_training_spectra(
        X_train_raw,
        y_train,
        wavelengths,
        domains=train_domains,
        groups=train_groups,
        config=cfg,
    )

    # Fix #5 (documentation): preprocessing_state is intentionally fitted on the
    # original (non-augmented) X_train_raw.  Fitting on augmented data would
    # recompute the MSC/SNV reference from perturbed spectra, introducing bias.
    # The state is then applied to the augmented set, which is the correct approach:
    # augmentation should not change the preprocessing reference distribution.
    preprocessing_state = preprocessing.fit_preprocessing_state(
        np.asarray(X_train_raw, dtype=float),
        np.asarray(wavelengths, dtype=float),
        method=preprocessing_method,
    )
    X_train_proc = preprocessing.preprocess_spectra(
        np.asarray(augmented["X_raw"], dtype=float),
        np.asarray(wavelengths, dtype=float),
        method=preprocessing_method,
        preprocessing_state=preprocessing_state,
    )

    X_cal_proc = None
    if X_cal_raw is not None:
        X_cal_proc = preprocessing.preprocess_spectra(
            np.asarray(X_cal_raw, dtype=float),
            np.asarray(wavelengths, dtype=float),
            method=preprocessing_method,
            preprocessing_state=preprocessing_state,
        )

    model_bundle = fit_calibrated_model(
        X_train=X_train_proc,
        y_train=np.asarray(augmented["y"]).astype(str),
        X_cal=X_cal_proc,
        y_cal=None if y_cal is None else np.asarray(y_cal).astype(str),
        n_components=n_components,
        alpha=alpha,
        preprocessing_method=preprocessing_method,
        calibration_method=calibration_method,
        preprocessing_state=preprocessing_state,
    )

    return {
        "model_bundle": model_bundle,
        "preprocessing_state": preprocessing_state,
        "config": config_to_dict(cfg),
        "n_original_train": int(len(np.asarray(X_train_raw))),
        "n_augmented_train": int(len(np.asarray(augmented["X_raw"]))),
        "augmentation_multiplier": float(len(np.asarray(augmented["X_raw"])) / max(len(np.asarray(X_train_raw)), 1)),
    }
