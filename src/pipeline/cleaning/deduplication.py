"""
src/pipeline/cleaning/deduplication.py

Handles a duplication pattern distinct from exact-row duplicates (already
removed per docs/deduplication_workflow.md, reports/data_deduplication_summary.md):
the same physical vehicle (`CÓDIGO DE VEHÍCULO`) appears in multiple rows
with a different `FECHA PROCESO`, everything else identical. Confirmed on
the real 2026 export: 164,503 rows / 143,852 unique vehicle codes, and 9,395
of those vehicles land in two *different* ISO weeks across their duplicate
rows -- if not collapsed before weekly grouping, the same vehicle would
appear as needing transport in two separate weekly episodes.

Policy: keep the row with the EARLIEST `fecha` (FECHA PROCESO) per vehicle
code -- interpreted as the vehicle's first processing event; later rows for
the same code are treated as reprocessing/corrections of that same event,
not new vehicles. Team decision (2026-07-17). Pass `keep="last"` to reverse
this if that interpretation changes later.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DedupReport:
    total_rows: int
    unique_vehicles: int
    rows_removed: int
    vehicles_spanning_multiple_weeks: int  # measured before dedup, for visibility

    @property
    def removed_pct(self) -> float:
        return 100.0 * self.rows_removed / self.total_rows if self.total_rows else 0.0


def deduplicate_by_vehicle_code(
    df: pd.DataFrame, code_col: str = "codigo_vehiculo", date_col: str = "fecha", keep: str = "first"
) -> tuple[pd.DataFrame, DedupReport]:
    """One row per `code_col`, keeping the `keep` ("first"/"last") by `date_col`."""
    total = len(df)

    # Guard: pandas' drop_duplicates() treats multiple NaN entries in the
    # subset column as duplicates of EACH OTHER, not as distinct missing
    # values. If `code_col` were ever all-NaN (e.g. loading.py's column
    # match silently failed upstream), this would collapse the whole frame
    # to ~1 row instead of raising. Fail loudly instead.
    n_missing_code = df[code_col].isna().sum()
    if n_missing_code:
        raise ValueError(
            f"{n_missing_code:,} of {total:,} rows have a missing "
            f"'{code_col}' -- refusing to deduplicate, since pandas would "
            "silently treat all of them as duplicates of each other. "
            "Fix the upstream column mapping (see loading.py) first."
        )

    # Measure cross-week impact before collapsing, for the report.
    if "fecha" in df.columns:
        iso_week = df[date_col].dt.isocalendar()["week"]
        spanning = (
            df.assign(_iso_week=iso_week).groupby(code_col)["_iso_week"].nunique() > 1
        ).sum()
    else:
        spanning = 0

    sorted_df = df.sort_values(date_col)
    dedup = sorted_df.drop_duplicates(subset=[code_col], keep=keep).sort_index()

    report = DedupReport(
        total_rows=total,
        unique_vehicles=len(dedup),
        rows_removed=total - len(dedup),
        vehicles_spanning_multiple_weeks=int(spanning),
    )
    return dedup, report
