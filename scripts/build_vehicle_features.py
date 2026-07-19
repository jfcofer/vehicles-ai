#!/usr/bin/env python3
"""Build the CU-enriched, in-scope vehicle feature dataset.

data/clean/SRI_Vehiculos_Nuevos_*.csv (all classes, raw-ish)
        -> filter to in-scope classes (config/vehicle_classes.yaml)
        -> add CU, iso_year, iso_week
        -> data/features/vehicles_in_scope.parquet
        -> reports/.../08_feature_coverage.md  (auto-generated, like 06_feasibility.md)

Usage (from repo root):
    python3 scripts/build_vehicle_features.py
    python3 scripts/build_vehicle_features.py --years 2023 2024
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.cleaning.deduplication import deduplicate_by_vehicle_code
from src.pipeline.cleaning.loading import load_all_years
from src.pipeline.transformation.derived_fields import VehicleClassConfig, build_features

DATA_DIR = REPO_ROOT / "data" / "clean"
CONFIG_PATH = REPO_ROOT / "config" / "vehicle_classes.yaml"
OUTPUT_PATH = REPO_ROOT / "data" / "features" / "vehicles_in_scope.parquet"
REPORT_PATH = (
    REPO_ROOT / "reports" / "03_proposals" / "fleet_routing" / "08_feature_coverage.md"
)


def write_report(dedup_report, report, skipped_years: list[int], out_path: Path) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Vehicle Feature Coverage",
        "",
        "> **Auto-generated.** Reproduce with:",
        "> ```bash",
        "> python3 scripts/build_vehicle_features.py",
        "> ```",
        "",
        f"**Generated:** {generated}  ",
        f"**Output:** `{out_path.relative_to(REPO_ROOT)}`  ",
        f"**Skipped years (no process-date column):** {', '.join(map(str, skipped_years)) or 'none'}",
        "",
        "---",
        "",
        "## Vehicle-code deduplication",
        "",
        "Same `CÓDIGO DE VEHÍCULO` can appear in multiple rows with only "
        "`FECHA PROCESO` differing (reprocessing) -- distinct from the exact-row "
        "duplicates already removed per `docs/deduplication_workflow.md`. "
        "One row is kept per vehicle (earliest `fecha`).",
        "",
        "| | |",
        "|---|---|",
        f"| Rows before | {dedup_report.total_rows:,} |",
        f"| Unique vehicles after | {dedup_report.unique_vehicles:,} |",
        f"| Rows removed | {dedup_report.rows_removed:,} ({dedup_report.removed_pct:.1f}%) |",
        f"| Vehicles that spanned 2+ different ISO weeks pre-dedup | {dedup_report.vehicles_spanning_multiple_weeks:,} |",
        "",
        "---",
        "",
        "## Scope filter",
        "",
        f"| | Rows | % |",
        f"|---|------|---|",
        f"| Total (all SRI classes, post vehicle-dedup) | {report.total_rows:,} | 100.0% |",
        f"| Kept (in-scope classes) | {report.kept_rows:,} | {report.kept_pct:.1f}% |",
        f"| Dropped (out-of-scope classes) | {report.dropped_rows:,} | {100 - report.kept_pct:.1f}% |",
        "",
        "### Dropped, by class",
        "",
        "| Clase | Rows dropped |",
        "|-------|--------------|",
    ]
    for clase, n in sorted(report.dropped_by_class.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {clase} | {n:,} |")

    if report.unrecognized_classes:
        lines += [
            "",
            "### ⚠️ Unrecognized classes (not in config at all — needs review)",
            "",
            "| Clase | Rows |",
            "|-------|------|",
        ]
        for clase, n in sorted(report.unrecognized_classes.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {clase} | {n:,} |")
    else:
        lines += ["", "No unrecognized classes — config/vehicle_classes.yaml covers 100% of the raw CLASE catalog."]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*", help="Years to include (default: all)")
    args = parser.parse_args()

    config = VehicleClassConfig.from_yaml(str(CONFIG_PATH))
    raw, skipped_years = load_all_years(DATA_DIR, args.years)
    raw, dedup_report = deduplicate_by_vehicle_code(raw, keep="first")
    features, report = build_features(raw, config)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(write_report(dedup_report, report, skipped_years, OUTPUT_PATH), encoding="utf-8")

    print(f"Dedup: {dedup_report.total_rows:,} -> {dedup_report.unique_vehicles:,} unique vehicles "
          f"({dedup_report.rows_removed:,} rows removed, {dedup_report.vehicles_spanning_multiple_weeks:,} spanned 2+ weeks)")
    print(f"Wrote {OUTPUT_PATH} ({report.kept_rows:,} rows, {report.kept_pct:.1f}% of post-dedup)")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
