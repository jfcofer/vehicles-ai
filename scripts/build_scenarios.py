#!/usr/bin/env python3
"""Build and label all weekly-canton episodes.

data/features/vehicles_in_scope.parquet
        -> group by (iso_year, iso_week, canton), drop N<5
        -> subsample + synthetic fleet + labeler (see src/loading/scenarios.py)
        -> data/episodes/episodes.parquet          (one row per episode)
        -> data/episodes/episode_vehicles.parquet  (one row per vehicle per episode)
        -> reports/.../09_scenarios_coverage.md

Usage (from repo root):
    python3 scripts/build_scenarios.py                # full run (~35k episodios, ~30 min)
    python3 scripts/build_scenarios.py --limit 200     # quick local test
    python3 scripts/build_scenarios.py --years 2025 2026
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.loading.scenarios import FLOOR_N, MAX_N, build_all_episodes

FEATURES_PATH = REPO_ROOT / "data" / "features" / "vehicles_in_scope.parquet"
EPISODES_PATH = REPO_ROOT / "data" / "episodes" / "episodes.parquet"
VEHICLES_PATH = REPO_ROOT / "data" / "episodes" / "episode_vehicles.parquet"
REPORT_PATH = REPO_ROOT / "reports" / "03_proposals" / "fleet_routing" / "09_scenarios_coverage.md"


def write_report(episodes_df: pd.DataFrame, summary, elapsed_s: float) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = len(episodes_df)
    trivial = int((episodes_df["n_deferred"] == 0).sum()) if n else 0
    non_optimal = int((~episodes_df["optimal"]).sum()) if n else 0

    lines = [
        "# Scenarios Coverage",
        "",
        "> **Auto-generated.** Reproduce with:",
        "> ```bash",
        "> python3 scripts/build_scenarios.py",
        "> ```",
        "",
        f"**Generated:** {generated}  ",
        f"**Elapsed:** {elapsed_s:.1f}s  ",
        f"**Floor (min N kept):** {FLOOR_N}  ",
        f"**Max N per episode (subsample cap):** {MAX_N}",
        "",
        "---",
        "",
        "## Episode universe",
        "",
        "| | |",
        "|---|---|",
        f"| Grupos semana-cantón totales | {summary.n_groups_total:,} |",
        f"| Excluidos por piso (N<{FLOOR_N}) | {summary.n_below_floor:,} |",
        f"| Episodios construidos y etiquetados | {summary.n_episodes_built:,} |",
        "",
        "## Resultado del labeler",
        "",
        "| | |",
        "|---|---|",
        f"| Filas en episode_vehicles.parquet | {int(episodes_df['n_sampled'].sum()) if n else 0:,} |",
        f"| Episodios triviales (nadie deferido) | {trivial:,} ({100*trivial/n:.1f}%)" if n else "| Episodios triviales | 0 |",
        f"| Episodios no-óptimos (time_budget agotado) | {non_optimal:,} |",
        f"| search_time_ms promedio | {episodes_df['search_time_ms'].mean():.1f}" if n else "| search_time_ms promedio | - |",
        f"| search_time_ms p99 | {episodes_df['search_time_ms'].quantile(0.99):.1f}" if n else "| search_time_ms p99 | - |",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max episodes to build (testing)")
    parser.add_argument("--years", type=int, nargs="*", help="Restrict to these iso_year values")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} no existe -- corran scripts/build_vehicle_features.py primero.")
        raise SystemExit(1)

    df = pd.read_parquet(FEATURES_PATH)
    if args.years:
        df = df[df["iso_year"].isin(args.years)]

    t0 = time.perf_counter()
    episodes_df, vehicles_df, summary = build_all_episodes(df, limit=args.limit)
    elapsed = time.perf_counter() - t0

    EPISODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(EPISODES_PATH, index=False)
    vehicles_df.to_parquet(VEHICLES_PATH, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(write_report(episodes_df, summary, elapsed), encoding="utf-8")

    print(f"Grupos totales: {summary.n_groups_total:,}  bajo el piso: {summary.n_below_floor:,}  "
          f"episodios construidos: {summary.n_episodes_built:,}  ({elapsed:.1f}s)")
    print(f"Wrote {EPISODES_PATH}")
    print(f"Wrote {VEHICLES_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
