from __future__ import annotations

from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd


def read_spectrum_dataframe(csv_source, source_label: str) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_source, header=None)
    if raw_df.empty:
        raise ValueError(f"Spectrum CSV is empty: {source_label}")
    if raw_df.shape[1] != 2:
        raise ValueError(
            f"Spectrum CSV must contain exactly 2 columns (wavelength, intensity); "
            f"found {raw_df.shape[1]} columns in {source_label}"
        )

    raw_df.columns = ["axis", "value"]
    raw_df["axis"] = pd.to_numeric(raw_df["axis"], errors="coerce")
    raw_df["value"] = pd.to_numeric(raw_df["value"], errors="coerce")

    invalid_rows = raw_df[raw_df[["axis", "value"]].isna().any(axis=1)]
    if not invalid_rows.empty:
        raise ValueError(
            f"Spectrum CSV contains non-numeric or missing wavelength/intensity values in "
            f"{len(invalid_rows)} row(s): {source_label}"
        )

    raw_df = raw_df.sort_values("axis", kind="mergesort").reset_index(drop=True)
    wavelengths = raw_df["axis"].to_numpy(dtype=float)
    intensities = raw_df["value"].to_numpy(dtype=float)

    if len(wavelengths) < 2:
        raise ValueError(f"Spectrum CSV must contain at least 2 valid points: {source_label}")
    if not np.isfinite(wavelengths).all() or not np.isfinite(intensities).all():
        raise ValueError(f"Spectrum CSV contains non-finite values: {source_label}")

    duplicate_mask = raw_df["axis"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_values = raw_df.loc[duplicate_mask, "axis"].tolist()
        preview = sorted({float(x) for x in duplicate_values})[:10]
        raise ValueError(
            "Spectrum CSV contains duplicate wavelength values, which would make interpolation ambiguous. "
            f"Found {int(duplicate_mask.sum())} duplicate row(s) in {source_label}. Example wavelengths: {preview}"
        )
    # Fix #6: after the duplicate check above, the wavelength array is guaranteed to
    # have no duplicates and (post-mergesort) is strictly increasing.  The two checks
    # below are therefore unreachable dead code and have been removed.

    return raw_df


def parse_and_validate_spectrum_csv(csv_source, source_label: str) -> tuple[np.ndarray, np.ndarray]:
    raw_df = read_spectrum_dataframe(csv_source, source_label=source_label)
    return (
        raw_df["axis"].to_numpy(dtype=float),
        raw_df["value"].to_numpy(dtype=float),
    )


def read_csv_spectrum(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    return parse_and_validate_spectrum_csv(csv_path, source_label=str(csv_path))


def read_zip_member_spectrum(archive_path: str | Path, member_name: str) -> tuple[np.ndarray, np.ndarray]:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Spectrum archive not found: {archive_path}")
    if archive_path.suffix.lower() != ".zip":
        raise ValueError(f"Archive input must be a .zip file: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if member_name not in names:
            raise FileNotFoundError(
                f"CSV member '{member_name}' not found in archive {archive_path}. Available members: {names[:20]}"
            )
        with zf.open(member_name, "r") as fh:
            data = io.BytesIO(fh.read())
    return parse_and_validate_spectrum_csv(data, source_label=f"{archive_path}::{member_name}")
