"""
src/pipeline/cleaning/loading.py

Loads and concatenates the raw SRI yearly CSVs (`data/clean/SRI_Vehiculos_
Nuevos_*.csv`), normalizing schema differences across years (process-date
column name AND format, canton column name, vehicle-code column name) into
one common frame: `fecha`, `canton`, `clase`, `codigo_vehiculo`.

This logic already existed, duplicated inline, in
`scripts/loading/episode_feasibility.py`. Pulled out here so every consumer
(the feature pipeline, the feasibility script, scenarios.py later) shares one
tested implementation instead of copies that can drift apart.

IMPORTANT -- the FECHA PROCESO column header lies about its own format in
most years. Verified against the real 2018-2026 exports:

    Year       Header claims        Actual values found
    2018       (MM/DD/AA)           D/M/YYYY H:MM        <- day-first, not MM/DD; has time
    2019       (MM/DD/AA)           D/M/YYYY HH:MM:SS     <- day-first, not MM/DD; has time
    2020       (DD/MM/AA)           D-Mon-YY (Spanish)    <- text month, not numeric
                                     (Sept, 4 letters, for September specifically)
    2021-2023  (DD/MM/AA)           D-Mon-YY (Spanish)    <- text month, not numeric
    2024       (DD/MM/AA)           D/M/YYYY              <- 4-digit year, not 2-digit
    2025-2026  (DD/MM/AAAA)         D/M/YYYY              <- header is correct here

Proof that every numeric year is day-first regardless of the header: in
2018/2019/2024, the SECOND slash-separated field is never > 12 while the
FIRST field is > 12 in 60-67% of rows -- the first field can only be the
day. So `_parse_fecha()` below never branches on the header text; it always
tries day-first numeric first, then a Spanish D-Mon-YY fallback, and reports
whatever still fails to parse instead of silently coercing it to NaT.

Do not "fix" this by deriving the format from the header again -- that
was tried, and it silently dropped 100% of 2018-2024 (2.4M+ rows) via
`errors="coerce"`, because pandas' strict `format=` parsing has zero
tolerance for a mismatched format, unlike the lenient dateutil fallback
that was here originally and is easy to mistake for "it's parsing fine".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CANTON_COLUMN_CANDIDATES = ["CANTÓN", "CANTON", "Codigo Canton"]
CLASS_COLUMN_CANDIDATES = ["CLASE", "Clase"]
CODE_COLUMN_CANDIDATES = [
    "CÓDIGO DE VEHÍCULO",
    "CODIGO DE VEHICULO",
    "CODIGO_VEHICULO",
    "CODIGO VEHICULO",  # 2018 export omits "DE"
]

SPANISH_MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "sept": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
}


def _locate_column(columns: list[str], candidates: list[str]) -> str | None:
    upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def _locate_date_column(columns: list[str]) -> str | None:
    """Any column whose name contains "FECHA PROCESO" -- the parenthetical
    format suffix is not trustworthy (see module docstring), so it isn't
    used to choose between candidates, only to confirm we found the date
    column at all."""
    for c in columns:
        if "FECHA PROCESO" in c.upper():
            return c
    return None


def _parse_fecha(raw: pd.Series) -> pd.Series:
    """Parse FECHA PROCESO values, trying both real formats found in the
    data (never the header's claimed format -- see module docstring)."""
    s = raw.astype(str).str.strip()
    s = s.str.split(" ").str[0]  # drop time-of-day; not needed for weekly grouping

    # Pattern A: numeric D/M/YYYY, day-first.
    parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")

    # Pattern B: D-Mon-YY, Spanish month abbreviation (3 or 4 letters).
    missing = parsed.isna()
    if missing.any():
        extracted = s[missing].str.extract(r"^(\d{1,2})-([A-Za-z]{3,4})-(\d{2})$")
        day = pd.to_numeric(extracted[0], errors="coerce")
        month = extracted[1].str.lower().map(SPANISH_MONTHS)
        year = pd.to_numeric(extracted[2], errors="coerce") + 2000
        text_parsed = pd.to_datetime(dict(year=year, month=month, day=day), errors="coerce")
        parsed.loc[missing] = text_parsed

    return parsed


def load_year_frame(csv_path: Path) -> pd.DataFrame | None:
    """Load one year's CSV into a normalized (fecha, canton, clase, codigo_vehiculo) frame.

    Returns None if the file has no recognizable process-date column at all
    (2017: month-only schema, no daily process date -- see 03_data.md).
    """
    head = pd.read_csv(csv_path, sep=";", encoding="latin1", nrows=0)
    cols = head.columns.tolist()

    date_col = _locate_date_column(cols)
    if not date_col:
        return None

    canton_col = _locate_column(cols, CANTON_COLUMN_CANDIDATES)
    class_col = _locate_column(cols, CLASS_COLUMN_CANDIDATES)
    code_col = _locate_column(cols, CODE_COLUMN_CANDIDATES)
    if not class_col:
        raise ValueError(f"{csv_path.name}: no CLASE/Clase column found")
    if not code_col:
        # No fallback: deduplicate_by_vehicle_code() drops rows by matching
        # `codigo_vehiculo`, and pandas treats multiple NaN entries as
        # duplicates of EACH OTHER -- an all-NaN column would silently
        # collapse this entire year to ~1 row. A code column exists under
        # some name in every 2018-2026 export; a year missing it needs a
        # human to add its exact name to CODE_COLUMN_CANDIDATES.
        raise ValueError(
            f"{csv_path.name}: no vehicle-code column found among "
            f"{CODE_COLUMN_CANDIDATES} -- add this year's exact column name "
            "before loading it (see deduplication.py docstring)."
        )

    usecols = [date_col, class_col, code_col]
    usecols += [canton_col] if canton_col else []
    df = pd.read_csv(csv_path, sep=";", encoding="latin1", usecols=usecols)
    df = df.rename(columns={date_col: "fecha", class_col: "clase", code_col: "codigo_vehiculo"})
    if canton_col:
        df = df.rename(columns={canton_col: "canton"})
    else:
        df["canton"] = pd.NA

    df["fecha"] = _parse_fecha(df["fecha"])
    n_before = len(df)
    df = df.dropna(subset=["fecha"])
    n_dropped = n_before - len(df)
    if n_dropped:
        pct = 100 * n_dropped / n_before
        if pct > 0.5:  # a handful of genuinely malformed rows is plausible; more is not
            raise ValueError(
                f"{csv_path.name}: {n_dropped:,}/{n_before:,} rows ({pct:.1f}%) "
                "failed date parsing -- this is high enough to suggest a new, "
                "unhandled date format rather than a few bad rows. Inspect "
                "before trusting the rest of this file."
            )

    df["clase"] = df["clase"].astype(str).str.strip().str.upper()
    df["source_year"] = int(csv_path.stem.split("_")[-1])
    df["uid"] = df["source_year"].astype(str) + "_" + df.index.astype(str)

    return df.reset_index(drop=True)


def load_all_years(data_dir: Path, years: list[int] | None = None) -> tuple[pd.DataFrame, list[int]]:
    """Concatenate every available year into one frame.

    Returns (frame, skipped_years).
    """
    files = sorted(Path(data_dir).glob("SRI_Vehiculos_Nuevos_*.csv"))
    if years:
        files = [f for f in files if any(str(y) in f.name for y in years)]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    parts, skipped = [], []
    for csv_path in files:
        year = int(csv_path.stem.split("_")[-1])
        frame = load_year_frame(csv_path)
        if frame is None:
            skipped.append(year)
            continue
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["fecha", "canton", "clase", "source_year", "codigo_vehiculo", "uid"]
    )
    return combined, skipped
