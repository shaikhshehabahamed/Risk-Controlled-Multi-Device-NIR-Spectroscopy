from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from spectrum_io import read_spectrum_dataframe


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent
INSTRUMENT_DIR_NAMES = {"Benchtop_N", "Handheld_F", "Handheld_S"}


def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()



def _extract_metadata_from_filename(filename: str) -> dict:
    stem = Path(filename).stem
    tokens = stem.split("_")

    specimen_id_full = stem
    specimen_group = tokens[0] if len(tokens) > 0 else None
    composition_code = tokens[1] if len(tokens) > 1 else None
    # Fix #7: tokens[-1] == tokens[0] for single-token filenames and equals
    # composition_code for two-token filenames — both are wrong for replicate_code.
    # Only assign a replicate code when at least 3 tokens are present.
    replicate_code = tokens[-1] if len(tokens) >= 3 else None
    signal_type = str(replicate_code)[-1] if replicate_code else None

    return {
        "filename": Path(filename).name,
        "specimen_id_full": specimen_id_full,
        "specimen_group": specimen_group,
        "composition_code": composition_code,
        "replicate_code": replicate_code,
        "signal_type": signal_type,
    }



def _read_spectrum_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = _resolve_local_path(csv_path)
    return read_spectrum_dataframe(csv_path, source_label=str(csv_path))



def _normalize_dataset_root(dataset_path: str | Path) -> Path:
    dataset_path = _resolve_local_path(dataset_path)

    if dataset_path.is_dir():
        child_dirs = {p.name for p in dataset_path.iterdir() if p.is_dir()}
        if INSTRUMENT_DIR_NAMES.intersection(child_dirs):
            return dataset_path

        nested_root = dataset_path / "NearInfrared"
        if nested_root.is_dir():
            nested_child_dirs = {p.name for p in nested_root.iterdir() if p.is_dir()}
            if INSTRUMENT_DIR_NAMES.intersection(nested_child_dirs):
                return nested_root.resolve()

    return dataset_path



def _append_spectrum_row(
    rows: list[dict],
    instrument: str,
    measurement: str,
    source: str,
    csv_path: Path,
) -> None:
    spectrum = _read_spectrum_csv(csv_path)
    rows.append(
        {
            "instrument": instrument,
            "measurement": measurement,
            "source": source,
            **_extract_metadata_from_filename(csv_path.name),
            "wavelength_axis": spectrum["axis"].tolist(),
            "intensity_axis": spectrum["value"].tolist(),
            "n_points": len(spectrum),
        }
    )



def _load_from_extracted_directory(
    dataset_root: Path,
    instrument_filter: Optional[List[str]],
    measurement_filter: Optional[List[str]],
    source_filter: Optional[List[str]],
) -> list[dict]:
    rows: list[dict] = []

    instrument_dirs = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir() and p.name in INSTRUMENT_DIR_NAMES],
        key=lambda p: p.name,
    )

    for instrument_dir in instrument_dirs:
        instrument = instrument_dir.name
        if instrument_filter and instrument not in instrument_filter:
            continue

        measurement_dirs = sorted(
            [p for p in instrument_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        for measurement_dir in measurement_dirs:
            measurement = measurement_dir.name
            if measurement_filter and measurement not in measurement_filter:
                continue

            source_dirs = sorted(
                [p for p in measurement_dir.iterdir() if p.is_dir()],
                key=lambda p: p.name,
            )
            for source_dir in source_dirs:
                source = source_dir.name
                if source_filter and source not in source_filter:
                    continue

                csv_files = sorted(
                    [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
                )
                for csv_path in csv_files:
                    _append_spectrum_row(
                        rows=rows,
                        instrument=instrument,
                        measurement=measurement,
                        source=source,
                        csv_path=csv_path,
                    )

    return rows



def load_nir_spectra(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    instrument_filter: Optional[List[str]] = None,
    measurement_filter: Optional[List[str]] = None,
    source_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load the NIR dataset into one tidy DataFrame.

    dataset_path must be the extracted dataset root that contains Benchtop_N,
    Handheld_F, and Handheld_S, or a parent directory containing NearInfrared/
    with those instrument folders.
    """
    dataset_path = _normalize_dataset_root(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(
            "Dataset path must be a directory containing instrument folders; "
            f"received file path: {dataset_path}"
        )

    rows = _load_from_extracted_directory(
        dataset_root=dataset_path,
        instrument_filter=instrument_filter,
        measurement_filter=measurement_filter,
        source_filter=source_filter,
    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No spectra were loaded from: {dataset_path}")
    return df



def summarize_loaded_data(df: pd.DataFrame) -> dict:
    return {
        "n_spectra": int(len(df)),
        "n_unique_specimens": int(df["specimen_group"].nunique()),
        "instruments": sorted(df["instrument"].dropna().unique().tolist()),
        "measurements": sorted(df["measurement"].dropna().unique().tolist()),
        "sources": sorted(df["source"].dropna().unique().tolist()),
        "points_per_instrument": (
            df.groupby("instrument")["n_points"]
            .agg(["min", "max", "median"])
            .reset_index()
            .to_dict(orient="records")
        ),
    }


if __name__ == "__main__":
    import json as _json
    _out = Path(__file__).resolve().parent / "outputs_nir_loader"
    _out.mkdir(parents=True, exist_ok=True)
    df = load_nir_spectra()
    summary = summarize_loaded_data(df)
    (_out / "nir_loader_summary.json").write_text(
        _json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    # Summary saved to outputs_nir_loader/nir_loader_summary.json
