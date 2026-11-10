from __future__ import annotations
import logging

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


_COMPONENT_PATTERN = re.compile(r"([a-z])(\d{1,3})", re.IGNORECASE)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_circularity_proxy"


@dataclass(frozen=True)
class CompositionProxy:
    composition_code: str
    canonical_code: str
    fractions: Dict[str, float]
    dominant_component: str
    dominant_fraction: float
    n_nonzero_components: int
    recoverability_proxy: float
    mono_material_bonus: float
    blend_complexity_penalty: float
    normalized_entropy: float
    effective_n_components: float
    sorting_difficulty_proxy: float


@dataclass(frozen=True)
class CompositionTargets:
    composition_code: str
    canonical_code: str
    component_order: tuple[str, ...]
    fraction_vector: np.ndarray
    mono_material: int
    dominant_component: str
    dominant_index: int
    n_nonzero_components: int
    recoverability_proxy: float
    sorting_difficulty_proxy: float



def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()



def _to_fraction_map(matches: Iterable[tuple[str, str]]) -> Dict[str, float]:
    fractions: Dict[str, float] = {}
    for component, raw in matches:
        key = str(component).lower()
        fractions[key] = fractions.get(key, 0.0) + (float(int(raw)) / 100.0)
    return fractions



def parse_composition_code(
    composition_code: str,
    *,
    validate_total: bool = True,
    total_tolerance: float = 1e-6,
) -> Dict[str, float]:
    code = str(composition_code).strip().lower()
    matches = _COMPONENT_PATTERN.findall(code)
    fractions = _to_fraction_map(matches)
    if not fractions:
        raise ValueError(f"Could not parse composition code: {composition_code}")

    total_fraction = float(sum(fractions.values()))
    if validate_total and abs(total_fraction - 1.0) > float(total_tolerance):
        raise ValueError(
            "Composition code fractions must sum to 1.0. "
            f"Got total={total_fraction:.6f} for {composition_code!r}."
        )
    return fractions



def canonicalize_composition_code(composition_code: str, *, include_zero_components: bool = True) -> str:
    fractions = parse_composition_code(composition_code, validate_total=True)
    items = sorted(fractions.items()) if include_zero_components else sorted((k, v) for k, v in fractions.items() if v > 0)
    if not items:
        raise ValueError(f"Composition code contains no components: {composition_code!r}")
    return "".join(f"{component}{int(round(value * 100)):03d}" for component, value in items)



def composition_to_proxy(composition_code: str) -> CompositionProxy:
    fractions = parse_composition_code(composition_code)
    nonzero = {k: float(v) for k, v in fractions.items() if float(v) > 0.0}
    if not nonzero:
        raise ValueError(
            f"Composition code contains no non-zero fractions: {composition_code}"
        )
    dominant_component = max(nonzero, key=nonzero.get)
    dominant_fraction = float(nonzero[dominant_component])
    n_nonzero_components = int(len(nonzero))

    concentration = float(sum(value ** 2 for value in nonzero.values()))
    entropy = float(-sum(value * np.log(max(value, 1e-12)) for value in nonzero.values()))
    entropy_norm = float(entropy / np.log(n_nonzero_components)) if n_nonzero_components > 1 else 0.0
    effective_n = float(np.exp(entropy))

    # These quantities remain explicit and reproducible for downstream policy studies:
    # - recoverability_proxy: concentration / purity proxy (Herfindahl index)
    # - mono_material_bonus: dominant fraction retained for compatibility with prior analyses
    # - blend_complexity_penalty: inverse effective component count to penalize complex blends
    recoverability_proxy = concentration
    mono_material_bonus = dominant_fraction
    blend_complexity_penalty = float(1.0 / effective_n)
    sorting_difficulty_proxy = float(1.0 - recoverability_proxy)

    return CompositionProxy(
        composition_code=str(composition_code),
        canonical_code=canonicalize_composition_code(composition_code),
        fractions=fractions,
        dominant_component=dominant_component,
        dominant_fraction=dominant_fraction,
        n_nonzero_components=n_nonzero_components,
        recoverability_proxy=recoverability_proxy,
        mono_material_bonus=mono_material_bonus,
        blend_complexity_penalty=blend_complexity_penalty,
        normalized_entropy=entropy_norm,
        effective_n_components=effective_n,
        sorting_difficulty_proxy=sorting_difficulty_proxy,
    )



def infer_component_order(class_labels: Sequence[str] | pd.Series | np.ndarray) -> list[str]:
    labels = pd.Series(class_labels).dropna().astype(str)
    if labels.empty:
        return []

    components = sorted({component for label in labels.tolist() for component in parse_composition_code(label).keys()})
    return list(components)


def _normalize_fraction_mapping(
    fractions: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    component_order: Sequence[str],
) -> Dict[str, float]:
    component_order = [str(component).lower() for component in component_order]
    if not component_order:
        raise ValueError("component_order must contain at least one component.")

    if isinstance(fractions, Mapping):
        normalized = {str(key).lower(): float(value) for key, value in fractions.items()}
    else:
        values = np.asarray(fractions, dtype=float).reshape(-1)
        if len(values) != len(component_order):
            raise ValueError(
                "Fraction sequence length must match component_order. "
                f"Got {len(values)} values for {len(component_order)} components."
            )
        normalized = {component_order[i]: float(values[i]) for i in range(len(component_order))}

    for component in component_order:
        normalized.setdefault(component, 0.0)
    return normalized


def _largest_remainder_integer_percentages(values: np.ndarray, total: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=int)

    clipped = np.clip(values, 0.0, None)
    if float(clipped.sum()) <= 0.0:
        out = np.zeros_like(clipped, dtype=int)
        out[0] = int(total)
        return out

    scaled = clipped / float(clipped.sum()) * float(total)
    base = np.floor(scaled).astype(int)
    remainder = int(total - base.sum())
    if remainder > 0:
        order = np.argsort(-(scaled - base))
        base[order[:remainder]] += 1
    # Fix #32: remainder < 0 is mathematically impossible because floor(x) <= x
    # for all x, so sum(floor(scaled)) <= sum(scaled) = total, meaning
    # remainder = total - sum(floor) >= 0 always.  The elif branch was dead code.
    return base


def fractions_to_composition_code(
    fractions: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    component_order: Sequence[str] | None = None,
    min_fraction: float = 0.0,
    drop_zeros: bool = True,
) -> str:
    if component_order is None:
        if not isinstance(fractions, Mapping):
            raise ValueError("component_order is required when fractions is not a mapping.")
        component_order = sorted(str(key).lower() for key in fractions.keys())

    normalized = _normalize_fraction_mapping(fractions, component_order=component_order)
    keys = [str(component).lower() for component in component_order]
    values = np.asarray([normalized[key] for key in keys], dtype=float)
    values = np.where(values >= float(min_fraction), values, 0.0)

    integer_percentages = _largest_remainder_integer_percentages(values, total=100)
    if drop_zeros:
        kept = [(keys[i], integer_percentages[i]) for i in range(len(keys)) if integer_percentages[i] > 0]
    else:
        kept = [(keys[i], integer_percentages[i]) for i in range(len(keys))]

    # Fix #45: _largest_remainder_integer_percentages guarantees sum == 100 with at
    # least one nonzero entry, so `kept` is never empty when drop_zeros=True.
    # The "if not kept" fallback was unreachable dead code and has been removed.
    assert kept, "Internal error: kept is empty after _largest_remainder_integer_percentages"

    code = "".join(f"{component}{percentage:03d}" for component, percentage in kept)
    return canonicalize_composition_code(code, include_zero_components=False)


def composition_to_targets(
    composition_code: str,
    *,
    component_order: Sequence[str] | None = None,
) -> CompositionTargets:
    proxy = composition_to_proxy(composition_code)
    if component_order is None:
        component_order = sorted(proxy.fractions)
    component_order = tuple(str(component).lower() for component in component_order)
    if not component_order:
        raise ValueError("component_order must contain at least one component.")

    fraction_vector = np.asarray([float(proxy.fractions.get(component, 0.0)) for component in component_order], dtype=float)
    if np.all(fraction_vector <= 0.0):
        raise ValueError(f"Composition code contains no mass on the requested component_order: {composition_code!r}")

    dominant_index = int(np.argmax(fraction_vector))
    return CompositionTargets(
        composition_code=str(composition_code),
        canonical_code=proxy.canonical_code,
        component_order=component_order,
        fraction_vector=fraction_vector,
        mono_material=int(proxy.n_nonzero_components == 1),
        dominant_component=str(component_order[dominant_index]),
        dominant_index=dominant_index,
        n_nonzero_components=int(proxy.n_nonzero_components),
        recoverability_proxy=float(proxy.recoverability_proxy),
        sorting_difficulty_proxy=float(proxy.sorting_difficulty_proxy),
    )


def build_structured_target_table(
    class_labels: Sequence[str] | pd.Series | np.ndarray,
    *,
    component_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    labels = pd.Series(class_labels).dropna().astype(str).reset_index(drop=True)
    if labels.empty:
        base_cols = [
            "composition_code",
            "canonical_code",
            "mono_material",
            "dominant_component",
            "dominant_index",
            "n_nonzero_components",
            "recoverability_proxy",
            "sorting_difficulty_proxy",
        ]
        return pd.DataFrame(columns=base_cols)

    if component_order is None:
        component_order = infer_component_order(labels)
    component_order = [str(component).lower() for component in component_order]

    rows = []
    for label in labels.tolist():
        target = composition_to_targets(label, component_order=component_order)
        row = {
            "composition_code": target.composition_code,
            "canonical_code": target.canonical_code,
            "mono_material": int(target.mono_material),
            "dominant_component": target.dominant_component,
            "dominant_index": int(target.dominant_index),
            "n_nonzero_components": int(target.n_nonzero_components),
            "recoverability_proxy": float(target.recoverability_proxy),
            "sorting_difficulty_proxy": float(target.sorting_difficulty_proxy),
        }
        for idx, component in enumerate(component_order):
            row[f"fraction_{component}"] = float(target.fraction_vector[idx])
        rows.append(row)

    return pd.DataFrame(rows)


def build_proxy_table(class_labels: Sequence[str] | pd.Series | np.ndarray) -> pd.DataFrame:
    labels = pd.Series(class_labels).dropna().astype(str)
    if labels.empty:
        return pd.DataFrame(
            columns=[
                "composition_code",
                "canonical_code",
                "dominant_component",
                "dominant_fraction",
                "n_nonzero_components",
                "recoverability_proxy",
                "mono_material_bonus",
                "blend_complexity_penalty",
                "normalized_entropy",
                "effective_n_components",
                "sorting_difficulty_proxy",
            ]
        )

    rows = []
    all_components = sorted({component for label in labels.unique().tolist() for component in parse_composition_code(label).keys()})
    for label in sorted(labels.unique().tolist()):
        proxy = composition_to_proxy(label)
        row = {
            "composition_code": proxy.composition_code,
            "canonical_code": proxy.canonical_code,
            "dominant_component": proxy.dominant_component,
            "dominant_fraction": proxy.dominant_fraction,
            "n_nonzero_components": proxy.n_nonzero_components,
            "recoverability_proxy": proxy.recoverability_proxy,
            "mono_material_bonus": proxy.mono_material_bonus,
            "blend_complexity_penalty": proxy.blend_complexity_penalty,
            "normalized_entropy": proxy.normalized_entropy,
            "effective_n_components": proxy.effective_n_components,
            "sorting_difficulty_proxy": proxy.sorting_difficulty_proxy,
        }
        for component in all_components:
            row[f"fraction_{component}"] = float(proxy.fractions.get(component, 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("composition_code").reset_index(drop=True)



def summarize_proxy_table(proxy_table: pd.DataFrame) -> dict:
    if proxy_table.empty:
        return {
            "n_classes": 0,
            "mean_recoverability_proxy": 0.0,
            "mean_sorting_difficulty_proxy": 0.0,
            "mono_material_classes": 0,
            "dominant_components": {},
        }

    summary = {
        "n_classes": int(len(proxy_table)),
        "mean_recoverability_proxy": float(proxy_table["recoverability_proxy"].mean()),
        "median_recoverability_proxy": float(proxy_table["recoverability_proxy"].median()),
        "mean_sorting_difficulty_proxy": float(proxy_table["sorting_difficulty_proxy"].mean()),
        "mono_material_classes": int((proxy_table["n_nonzero_components"] == 1).sum()),
        "blend_classes": int((proxy_table["n_nonzero_components"] > 1).sum()),
        "dominant_components": proxy_table["dominant_component"].value_counts().sort_index().to_dict(),
    }
    return summary



def build_proxy_table_from_dataset(
    dataset_path: str | Path,
    *,
    instrument_filter: Sequence[str] | None = None,
    measurement_filter: Sequence[str] | None = None,
    source_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    import nir_loader

    df = nir_loader.load_nir_spectra(
        dataset_path=_resolve_local_path(dataset_path),
        instrument_filter=list(instrument_filter) if instrument_filter is not None else None,
        measurement_filter=list(measurement_filter) if measurement_filter is not None else None,
        source_filter=list(source_filter) if source_filter is not None else None,
    )
    return build_proxy_table(df["composition_code"].dropna().astype(str).unique().tolist())



def save_proxy_artifacts(
    proxy_table: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    csv_name: str = "circularity_proxy_table.csv",
    summary_name: str = "circularity_proxy_summary.json",
) -> dict:
    output_dir = _resolve_local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    summary_path = output_dir / summary_name
    proxy_table.to_csv(csv_path, index=False)
    summary = summarize_proxy_table(proxy_table)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Fix #25: import publication_outputs lazily to avoid loading matplotlib
    # unconditionally when save_proxy_artifacts is called without figure generation.
    from publication_outputs import figure_circularity_proxy  # noqa: PLC0415
    figure_circularity_proxy(proxy_table, output_dir)

    return {
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        **summary,
    }



def _require_equal_length_1d_arrays(**arrays) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}
    for name, values in arrays.items():
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}.")
        normalized[name] = arr
        lengths[name] = int(arr.shape[0])

    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Input arrays must have the same length; got {details}.")
    return normalized



def expected_circularity_capture(
    predicted_labels: np.ndarray,
    acceptance_mask: np.ndarray,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    if confidence is None:
        confidence = np.ones(len(np.asarray(predicted_labels)), dtype=float)
    arrays = _require_equal_length_1d_arrays(
        predicted_labels=np.asarray(predicted_labels, dtype=object),
        acceptance_mask=np.asarray(acceptance_mask, dtype=bool),
        confidence=np.asarray(confidence, dtype=float),
    )
    predicted_labels = arrays["predicted_labels"]
    acceptance_mask = arrays["acceptance_mask"]
    confidence = arrays["confidence"]

    # Fix #23: cache proxy lookups — O(n_unique) parses instead of O(n_samples).
    _proxy_cache: dict[str, float] = {}
    out = np.zeros(len(predicted_labels), dtype=float)
    for i, label in enumerate(predicted_labels):
        label_str = str(label)
        if label_str not in _proxy_cache:
            _proxy_cache[label_str] = composition_to_proxy(label_str).recoverability_proxy
        out[i] = float(acceptance_mask[i]) * float(confidence[i]) * _proxy_cache[label_str]
    return out



def realized_circularity_capture(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    acceptance_mask: np.ndarray,
) -> np.ndarray:
    arrays = _require_equal_length_1d_arrays(
        true_labels=np.asarray(true_labels, dtype=object),
        predicted_labels=np.asarray(predicted_labels, dtype=object),
        acceptance_mask=np.asarray(acceptance_mask, dtype=bool),
    )
    true_labels = arrays["true_labels"]
    predicted_labels = arrays["predicted_labels"]
    acceptance_mask = arrays["acceptance_mask"]

    # Fix #23: cache proxy lookups — O(n_unique) parses instead of O(n_samples).
    _proxy_cache: dict[str, float] = {}
    out = np.zeros(len(true_labels), dtype=float)
    for i, true_label in enumerate(true_labels):
        label_str = str(true_label)
        if label_str not in _proxy_cache:
            _proxy_cache[label_str] = composition_to_proxy(label_str).recoverability_proxy
        is_accepted_and_correct = bool(acceptance_mask[i]) and (label_str == str(predicted_labels[i]))
        out[i] = float(is_accepted_and_correct) * _proxy_cache[label_str]
    return out



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build circularity proxy tables from explicit labels or the local NIR dataset.")
    parser.add_argument("--dataset-path", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--measurement", type=str, nargs="*", default=None)
    parser.add_argument("--instrument", type=str, nargs="*", default=None)
    parser.add_argument("--source", type=str, nargs="*", default=None)
    parser.add_argument("--class-label", type=str, nargs="*", default=None)
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    if args.class_label:
        proxy_table = build_proxy_table(args.class_label)
    else:
        proxy_table = build_proxy_table_from_dataset(
            dataset_path=args.dataset_path,
            instrument_filter=args.instrument,
            measurement_filter=args.measurement,
            source_filter=args.source,
        )
    save_proxy_artifacts(proxy_table, output_dir=args.output_dir)
    # Summary saved to <output_dir>/proxy_summary.json


if __name__ == "__main__":
    main()
