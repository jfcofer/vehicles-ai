"""
src/pipeline/transformation/derived_fields.py

Turns a normalized (fecha, canton, clase, uid) frame -- see
src/pipeline/cleaning/loading.py -- into the feature-ready table the loading
labeler and scenario builder consume: adds `cu`, `iso_year`, `iso_week`, and
filters out-of-scope classes (documented, with a coverage report -- not
silently dropped, see 02_scope.md / 03_data.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yaml


@dataclass
class VehicleClassConfig:
    cu_by_class: dict[str, float]
    out_of_scope: list[str]

    @classmethod
    def from_yaml(cls, path: str) -> "VehicleClassConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cu_by_class = {c: v["cu"] for c, v in raw["in_scope"].items()}
        return cls(cu_by_class=cu_by_class, out_of_scope=raw.get("out_of_scope", []))

    @property
    def in_scope_classes(self) -> list[str]:
        return list(self.cu_by_class.keys())


@dataclass
class FilterReport:
    total_rows: int
    kept_rows: int
    dropped_rows: int
    dropped_by_class: dict[str, int]
    unrecognized_classes: dict[str, int]  # present in data, not in config at all

    @property
    def kept_pct(self) -> float:
        return 100.0 * self.kept_rows / self.total_rows if self.total_rows else 0.0


def filter_in_scope(
    df: pd.DataFrame, config: VehicleClassConfig
) -> tuple[pd.DataFrame, FilterReport]:
    """Keep only rows whose `clase` is in the in-scope list.

    Also separately reports classes present in the data that are neither in
    `in_scope` nor `out_of_scope` in the config -- a sign the config is stale
    relative to the real SRI CLASE catalog and needs a human look, rather
    than being silently swept into "dropped".
    """
    total = len(df)
    known = set(config.in_scope_classes) | set(config.out_of_scope)
    unrecognized = df.loc[~df["clase"].isin(known), "clase"].value_counts().to_dict()

    mask = df["clase"].isin(config.in_scope_classes)
    kept = df.loc[mask].copy()
    dropped_by_class = df.loc[~mask, "clase"].value_counts().to_dict()

    report = FilterReport(
        total_rows=total,
        kept_rows=len(kept),
        dropped_rows=total - len(kept),
        dropped_by_class=dropped_by_class,
        unrecognized_classes=unrecognized,
    )
    return kept, report


def add_cu(df: pd.DataFrame, config: VehicleClassConfig) -> pd.DataFrame:
    """Add the `cu` column via the class -> CU config mapping."""
    df = df.copy()
    df["cu"] = df["clase"].map(config.cu_by_class)
    if df["cu"].isna().any():
        bad = df.loc[df["cu"].isna(), "clase"].unique().tolist()
        raise ValueError(
            f"CU missing for classes present after filtering: {bad} "
            "-- filter_in_scope() should have removed these already."
        )
    return df


def add_iso_week(df: pd.DataFrame) -> pd.DataFrame:
    """Add `iso_year` / `iso_week` from `fecha`, for weekly episode grouping."""
    df = df.copy()
    iso = df["fecha"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)
    return df


def build_features(df: pd.DataFrame, config: VehicleClassConfig) -> tuple[pd.DataFrame, FilterReport]:
    """Full derived-fields step: filter -> CU -> ISO week."""
    kept, report = filter_in_scope(df, config)
    kept = add_cu(kept, config)
    kept = add_iso_week(kept)
    cols = ["uid", "codigo_vehiculo", "canton", "clase", "cu", "fecha", "iso_year", "iso_week", "source_year"]
    return kept[cols].reset_index(drop=True), report
