from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

import cupy as cp


def asnumpy(x):
    """Transfer a CuPy array to a NumPy array on the host."""
    return cp.asnumpy(x)


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent


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

def _load_loader_module(loader_script_path: str | Path):
    loader_script_path = _resolve_local_path(loader_script_path)
    return _load_module_from_path("nir_loader", loader_script_path)


def load_nir_subset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
    instruments: Optional[Sequence[str]] = None,
    measurement: Optional[str] = None,
    sources: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    mod = _load_loader_module(loader_script_path)
    measurement_filter = [measurement] if measurement is not None else None
    instrument_filter = list(instruments) if instruments is not None else None
    source_filter = list(sources) if sources is not None else None
    df = mod.load_nir_spectra(
        dataset_path=dataset_path,
        instrument_filter=instrument_filter,
        measurement_filter=measurement_filter,
        source_filter=source_filter,
    )
    return df.reset_index(drop=True)


def load_benchtop_reflectance_subset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    loader_script_path: str | Path = "nir_loader.py",
) -> pd.DataFrame:
    return load_nir_subset(
        dataset_path=dataset_path,
        loader_script_path=loader_script_path,
        instruments=["Benchtop_N"],
        measurement="Reflectance",
    )


def _get_axis_for_instrument(df: pd.DataFrame, instrument: str) -> np.ndarray:
    sub = df[df["instrument"] == instrument]
    if sub.empty:
        raise ValueError(f"No rows found for instrument={instrument}")
    axis = np.asarray(sub.iloc[0]["wavelength_axis"], dtype=float)
    for i, candidate in enumerate(sub["wavelength_axis"]):
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != axis.shape or not np.allclose(candidate, axis):
            raise ValueError(
                f"Inconsistent wavelength axis found for instrument={instrument} at row offset {i}"
            )
    return axis


def infer_common_wavelength_grid(
    df: pd.DataFrame,
    instruments: Optional[Sequence[str]] = None,
    n_points: Optional[int] = None,
) -> np.ndarray:
    if instruments is None:
        instruments = sorted(df["instrument"].dropna().astype(str).unique().tolist())
    if not instruments:
        raise ValueError("At least one instrument is required to infer a common wavelength grid.")

    axes = [_get_axis_for_instrument(df, instrument) for instrument in instruments]
    overlap_min = max(float(axis.min()) for axis in axes)
    overlap_max = min(float(axis.max()) for axis in axes)
    if overlap_min >= overlap_max:
        raise ValueError(
            f"No overlapping wavelength range across instruments {list(instruments)}."
        )

    if n_points is None:
        counts = []
        for axis in axes:
            count = int(np.sum((axis >= overlap_min) & (axis <= overlap_max)))
            counts.append(count)
        n_points = int(min(counts))

    if n_points < 8:
        raise ValueError(f"n_points must be >= 8, got {n_points}")

    return np.linspace(overlap_min, overlap_max, int(n_points), dtype=float)


def resample_spectrum_to_grid(
    wavelengths: Sequence[float],
    intensities: Sequence[float],
    target_grid: Sequence[float],
) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    target_grid = np.asarray(target_grid, dtype=float)

    if wavelengths.ndim != 1 or intensities.ndim != 1:
        raise ValueError("wavelengths and intensities must be 1D arrays.")
    if len(wavelengths) != len(intensities):
        raise ValueError("wavelengths and intensities must have the same length.")
    if len(wavelengths) < 2:
        raise ValueError("Each spectrum must contain at least two points.")

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    intensities = intensities[order]

    if target_grid.min() < wavelengths.min() or target_grid.max() > wavelengths.max():
        raise ValueError(
            "Target grid lies outside the available wavelength range for this spectrum. "
            f"Target=({target_grid.min()}, {target_grid.max()}), spectrum=({wavelengths.min()}, {wavelengths.max()})"
        )

    wl_gpu  = cp.asarray(wavelengths, dtype=cp.float64)
    fp_gpu  = cp.asarray(intensities, dtype=cp.float64)
    tg_gpu  = cp.asarray(target_grid, dtype=cp.float64)
    return asnumpy(cp.interp(tg_gpu, wl_gpu, fp_gpu)).astype(float)


def _gpu_batch_interp(
    xp_shared: np.ndarray,
    fp_matrix: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """
    Vectorised linear interpolation on GPU for a batch of spectra that all
    share the same source wavelength grid.

    Parameters
    ----------
    xp_shared : (n_src_pts,)            – shared source wavelength grid
    fp_matrix : (n_spectra, n_src_pts)  – source intensity values
    x_new     : (n_tgt_pts,)            – target wavelength grid

    Returns
    -------
    out : np.ndarray  shape (n_spectra, n_tgt_pts)
    """
    # Fix #34: this function silently linear-extrapolates when x_new falls outside
    # the range of xp_shared.  All callers in resample_dataframe_to_grid perform an
    # explicit out-of-range check before calling here, so extrapolation should never
    # occur in normal use.  Do NOT call this function directly without a prior range
    # check.
    xp = cp.asarray(xp_shared, dtype=cp.float64)          # (n_src_pts,)
    fp = cp.asarray(fp_matrix, dtype=cp.float64)       # (n_spectra, n_src_pts)
    xn = cp.asarray(x_new, dtype=cp.float64)           # (n_tgt_pts,)

    # searchsorted: for each target point find the right bin in the source grid
    idx = cp.searchsorted(xp, xn, side="right")        # (n_tgt_pts,)
    idx = cp.clip(idx, 1, len(xp) - 1)

    x0 = xp[idx - 1]                                   # (n_tgt_pts,)
    x1 = xp[idx]
    dx = x1 - x0
    # Slope for every spectrum × every target point: (n_spectra, n_tgt_pts)
    f0 = fp[:, idx - 1]
    f1 = fp[:, idx]
    t  = (xn - x0) / cp.where(dx < 1e-15, cp.ones_like(dx), dx)
    out = f0 + (f1 - f0) * t
    return asnumpy(out)


def resample_dataframe_to_grid(
    df: pd.DataFrame,
    target_grid: Sequence[float],
) -> pd.DataFrame:
    """
    Resample all spectra in *df* to *target_grid* using GPU linear interpolation.
    When all spectra share the same source wavelength grid (the common case) a
    single batched GPU pass is used.  When grids differ, spectra are grouped by
    their unique source grid and each group is processed as a GPU batch.
    """
    target_grid     = np.asarray(target_grid, dtype=float)
    target_grid_lst = target_grid.tolist()
    original        = df.copy().reset_index(drop=True)

    wl_arrays  = [np.asarray(r, dtype=float) for r in original["wavelength_axis"]]
    int_arrays = [np.asarray(r, dtype=float) for r in original["intensity_axis"]]

    # Check if all source grids are identical (the typical case)
    shapes_equal = all(a.shape == wl_arrays[0].shape for a in wl_arrays)
    grids_equal  = shapes_equal and all(
        np.array_equal(a, wl_arrays[0]) for a in wl_arrays
    )

    resampled: list = [None] * len(original)

    if grids_equal and len(wl_arrays) > 0:
        # Single batched GPU pass
        src_min = float(wl_arrays[0].min())
        src_max = float(wl_arrays[0].max())
        if target_grid.min() < src_min or target_grid.max() > src_max:
            raise ValueError(
                "Target grid lies outside the available wavelength range. "
                f"Target=({target_grid.min()}, {target_grid.max()}), "
                f"spectrum=({src_min}, {src_max})"
            )
        fp_matrix  = np.vstack(int_arrays)
        batch_out  = _gpu_batch_interp(wl_arrays[0], fp_matrix, target_grid)
        resampled  = [batch_out[i].tolist() for i in range(len(original))]
    else:
        # Heterogeneous grids – group by unique source grid and GPU-batch each group
        from collections import defaultdict
        grid_to_indices: dict = defaultdict(list)
        for idx, wl in enumerate(wl_arrays):
            grid_key = wl.tobytes()
            grid_to_indices[grid_key].append(idx)

        for grid_key, indices in grid_to_indices.items():
            src_wl = wl_arrays[indices[0]]
            src_min = float(src_wl.min())
            src_max = float(src_wl.max())
            if target_grid.min() < src_min or target_grid.max() > src_max:
                raise ValueError(
                    "Target grid lies outside the available wavelength range. "
                    f"Target=({target_grid.min()}, {target_grid.max()}), "
                    f"spectrum=({src_min}, {src_max})"
                )
            fp_matrix = np.vstack([int_arrays[i] for i in indices])
            batch_out = _gpu_batch_interp(src_wl, fp_matrix, target_grid)
            for local_i, global_i in enumerate(indices):
                resampled[global_i] = batch_out[local_i].tolist()

    out = original.copy()
    out["wavelength_axis"] = [list(target_grid_lst) for _ in range(len(out))]
    out["intensity_axis"]  = resampled
    out["n_points"]        = int(len(target_grid))
    out["resampled_to_common_grid"] = True
    return out



def keep_common_classes(
    df: pd.DataFrame,
    instruments: Optional[Sequence[str]] = None,
    target_col: str = "composition_code",
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    if instruments is None:
        instruments = sorted(df["instrument"].dropna().astype(str).unique().tolist())

    class_sets = []
    for instrument in instruments:
        sub = df[df["instrument"] == instrument]
        class_sets.append(set(sub[target_col].astype(str).unique().tolist()))

    if not class_sets:
        raise ValueError("No instruments available when filtering common classes.")

    common_classes = sorted(set.intersection(*class_sets))
    filtered = df[df[target_col].astype(str).isin(common_classes)].reset_index(drop=True)

    summary_rows = []
    all_classes = sorted(df[target_col].astype(str).unique().tolist())
    for class_label in all_classes:
        row = {target_col: class_label, "is_common": class_label in common_classes}
        for instrument in instruments:
            count = int(
                df[(df["instrument"] == instrument) & (df[target_col].astype(str) == class_label)]["specimen_group"].nunique()
            )
            row[f"n_groups_{instrument}"] = count
        summary_rows.append(row)

    return filtered, common_classes, pd.DataFrame(summary_rows)


def keep_groups_present_in_all_instruments(
    df: pd.DataFrame,
    instruments: Optional[Sequence[str]] = None,
    group_col: str = "specimen_group",
) -> tuple[pd.DataFrame, list[str]]:
    if instruments is None:
        instruments = sorted(df["instrument"].dropna().astype(str).unique().tolist())
    group_sets = [set(df[df["instrument"] == instrument][group_col].astype(str).unique().tolist()) for instrument in instruments]
    common_groups = sorted(set.intersection(*group_sets)) if group_sets else []
    filtered = df[df[group_col].astype(str).isin(common_groups)].reset_index(drop=True)
    return filtered, common_groups


def summarize_common_class_groups(
    df: pd.DataFrame,
    instruments: Optional[Sequence[str]] = None,
    target_col: str = "composition_code",
    group_col: str = "specimen_group",
) -> pd.DataFrame:
    if instruments is None:
        instruments = sorted(df["instrument"].dropna().astype(str).unique().tolist())

    rows = []
    for class_label, class_df in df.groupby(target_col):
        row = {target_col: str(class_label)}
        per_instrument_counts = []
        for instrument in instruments:
            count = int(
                class_df[class_df["instrument"] == instrument][group_col].astype(str).nunique()
            )
            row[f"n_groups_{instrument}"] = count
            per_instrument_counts.append(count)
        row["min_groups_across_instruments"] = int(min(per_instrument_counts)) if per_instrument_counts else 0
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["min_groups_across_instruments", target_col],
        ascending=[True, True],
    ).reset_index(drop=True)


def filter_classes_by_min_groups_across_instruments(
    df: pd.DataFrame,
    min_groups_per_class: int,
    instruments: Optional[Sequence[str]] = None,
    target_col: str = "composition_code",
    group_col: str = "specimen_group",
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    summary = summarize_common_class_groups(
        df=df,
        instruments=instruments,
        target_col=target_col,
        group_col=group_col,
    )
    eligible = summary.loc[
        summary["min_groups_across_instruments"] >= int(min_groups_per_class),
        target_col,
    ].astype(str).tolist()
    filtered = df[df[target_col].astype(str).isin(eligible)].reset_index(drop=True)
    return filtered, eligible, summary


def _derive_specimen_aggregate_id(metadata: pd.DataFrame) -> pd.Series:
    if "specimen_id_full" not in metadata.columns:
        raise ValueError(
            "Aggregation 'mean_by_specimen' requires the 'specimen_id_full' column in metadata."
        )

    if "replicate_code" not in metadata.columns:
        return metadata["specimen_id_full"].astype(str).rename("specimen_aggregate_id")

    aggregate_ids = []
    for specimen_id_full, replicate_code in zip(
        metadata["specimen_id_full"].astype(str),
        metadata["replicate_code"].fillna("").astype(str),
    ):
        suffix = f"_{replicate_code}" if replicate_code else ""
        if suffix and specimen_id_full.endswith(suffix):
            aggregate_ids.append(specimen_id_full[: -len(suffix)])
        else:
            aggregate_ids.append(specimen_id_full)

    return pd.Series(aggregate_ids, index=metadata.index, name="specimen_aggregate_id")


def build_training_matrix(
    df: pd.DataFrame,
    target_col: str = "composition_code",
    group_col: str = "specimen_group",
    aggregate: Optional[Literal["mean_by_specimen"]] = None,
):
    required_cols = [
        "wavelength_axis",
        "intensity_axis",
        target_col,
        group_col,
        "source",
        "replicate_code",
        "instrument",
        "measurement",
        "filename",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("Input dataframe is empty after filtering/resampling.")

    first_axis = np.asarray(df.loc[0, "wavelength_axis"], dtype=float)
    for i, axis in enumerate(df["wavelength_axis"]):
        axis = np.asarray(axis, dtype=float)
        if axis.shape != first_axis.shape or not np.allclose(axis, first_axis):
            raise ValueError(f"Inconsistent wavelength axis found at row {i}")

    wavelengths = first_axis
    X = np.vstack(df["intensity_axis"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in intensity matrix")

    metadata_cols = [
        "instrument", "measurement", "source", "filename",
        "specimen_id_full", "specimen_group", "composition_code",
        "replicate_code", "signal_type", "n_points",
    ]
    metadata = df[[c for c in metadata_cols if c in df.columns]].copy()

    if aggregate == "mean_by_specimen":
        metadata = metadata.copy()
        metadata["specimen_aggregate_id"] = _derive_specimen_aggregate_id(metadata)
        grouping_cols = [
            c for c in ["instrument", "measurement", "source", "specimen_aggregate_id"]
            if c in metadata.columns
        ]

        agg_rows = []
        for _, idx in metadata.groupby(grouping_cols, sort=False).groups.items():
            idx = np.array(list(idx), dtype=int)
            X_mean = X[idx].mean(axis=0)

            first_meta = metadata.iloc[idx[0]].copy()
            for consistency_col in [target_col, group_col, "specimen_aggregate_id"]:
                if consistency_col in metadata.columns:
                    values = metadata.iloc[idx][consistency_col].astype(str).unique().tolist()
                    if len(values) != 1:
                        raise ValueError(
                            "Aggregation 'mean_by_specimen' found inconsistent values within a specimen group "
                            f"for column '{consistency_col}': {values}"
                        )

            first_meta["specimen_id_full"] = str(first_meta["specimen_aggregate_id"])
            first_meta["replicate_code"] = "AGG"
            first_meta["n_aggregated_spectra"] = int(len(idx))
            first_meta["aggregation"] = "mean_by_specimen"
            agg_rows.append((X_mean, first_meta))

        # Fix #52: guard against an empty agg_rows list (e.g. all-NaN groupby keys)
        # before np.vstack, which raises ValueError on an empty sequence.
        if not agg_rows:
            raise ValueError(
                "build_training_matrix with aggregate='mean_by_specimen' produced "
                "zero aggregated specimens.  Check that the grouping columns "
                f"({grouping_cols}) contain non-null values."
            )
        X = np.vstack([row[0] for row in agg_rows])
        metadata = pd.DataFrame([row[1] for row in agg_rows]).reset_index(drop=True)

    y = metadata[target_col].astype(str).to_numpy()
    groups = metadata[group_col].astype(str).to_numpy()

    unique_labels = sorted(pd.Series(y).unique().tolist())
    label_map = pd.DataFrame({
        "class_index": np.arange(len(unique_labels), dtype=int),
        "class_label": unique_labels,
    })
    label_to_index = dict(zip(label_map["class_label"], label_map["class_index"]))
    y_encoded = np.array([label_to_index[label] for label in y], dtype=int)

    return X, y, y_encoded, groups, wavelengths, metadata, label_map


def summarize_matrix(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    wavelengths: np.ndarray,
    metadata: pd.DataFrame,
) -> dict:
    class_counts = pd.Series(y).value_counts().sort_values(ascending=False)
    return {
        "X_shape": tuple(X.shape),
        "n_classes": int(pd.Series(y).nunique()),
        "n_groups": int(pd.Series(groups).nunique()),
        "wavelength_range": (float(wavelengths.min()), float(wavelengths.max())),
        "top_10_class_counts": class_counts.head(10).to_dict(),
        "source_counts": metadata["source"].value_counts().to_dict(),
        "instrument_counts": metadata["instrument"].value_counts().to_dict(),
    }


def save_experiment_bundle(
    output_dir: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    y_encoded: np.ndarray,
    groups: np.ndarray,
    wavelengths: np.ndarray,
    metadata: pd.DataFrame,
    label_map: pd.DataFrame,
    prefix: str = "benchtop_reflectance",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / f"{prefix}_experiment.npz",
        X=X,
        y=y,
        y_encoded=y_encoded,
        groups=groups,
        wavelengths=wavelengths,
    )
    metadata.to_csv(output_dir / f"{prefix}_metadata.csv", index=False)
    label_map.to_csv(output_dir / f"{prefix}_label_map.csv", index=False)


if __name__ == "__main__":
    import json as _json
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs_training_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_benchtop_reflectance_subset(
        dataset_path=base_dir,
        loader_script_path=base_dir / "nir_loader.py",
    )

    X, y, y_encoded, groups, wavelengths, metadata, label_map = build_training_matrix(df)
    spectrum_summary = summarize_matrix(X, y, groups, wavelengths, metadata)

    X_mean, y_mean, y_mean_encoded, groups_mean, wavelengths_mean, metadata_mean, label_map_mean = build_training_matrix(
        df,
        aggregate="mean_by_specimen",
    )
    specimen_summary = summarize_matrix(X_mean, y_mean, groups_mean, wavelengths_mean, metadata_mean)

    save_experiment_bundle(
        output_dir=out_dir,
        X=X,
        y=y,
        y_encoded=y_encoded,
        groups=groups,
        wavelengths=wavelengths,
        metadata=metadata,
        label_map=label_map,
    )

    multi_df = load_nir_subset(
        dataset_path=base_dir,
        loader_script_path=base_dir / "nir_loader.py",
        instruments=["Benchtop_N", "Handheld_F", "Handheld_S"],
        measurement="Reflectance",
    )
    common_df, common_classes, _ = keep_common_classes(multi_df)
    grid = infer_common_wavelength_grid(common_df, n_points=128)
    common_resampled = resample_dataframe_to_grid(common_df, grid)
    Xc, yc, _, gc, wc, mc, _ = build_training_matrix(common_resampled)
    multidevice_summary = summarize_matrix(Xc, yc, gc, wc, mc)
    multidevice_summary["n_common_classes"] = int(len(common_classes))

    # Save all summaries to a single JSON — no stdout output
    all_summaries = {
        "spectrum_level": spectrum_summary,
        "specimen_mean": specimen_summary,
        "multidevice_common_grid": multidevice_summary,
    }
    (out_dir / "matrix_summaries.json").write_text(
        _json.dumps(all_summaries, indent=2, default=str), encoding="utf-8"
    )
    # All outputs saved to outputs_training_matrix/
