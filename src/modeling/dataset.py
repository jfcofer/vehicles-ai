"""src/modeling/dataset.py

Carga y particionado del dataset de entrenamiento del asignador.

Dos decisiones no negociables viven aquí.

1. **El join es obligatorio.** `episode_vehicles.parquet` contiene el vehículo y
   su etiqueta, pero **no las capacidades de la flota**: ésas están en
   `episodes.parquet` (`n_trucks`, `truck_capacities`). Un modelo entrenado sólo
   con la tabla de vehículos no puede saber si hay uno o cuatro camiones ni de
   qué tamaño, y por tanto no puede predecir la asignación mejor que al azar.

2. **La partición es por episodio, nunca por fila.** Todos los vehículos de un
   episodio comparten la misma flota, el mismo manifiesto y el mismo contexto
   agregado. Repartir filas individualmente pone casi la misma información a
   ambos lados de la partición y produce métricas infladas que no sobreviven a
   datos nuevos.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EPISODE_COLUMNS = [
    "episode_id",
    "iso_year",
    "iso_week",
    "canton",
    "n_sampled",
    "n_trucks",
    "truck_capacities",
    "n_loaded",
    "n_deferred",
    "cu_utilized",
    "optimal",
]
VEHICLE_COLUMNS = ["episode_id", "uid", "clase", "cu", "canton", "truck", "loaded"]

# 2017 no tiene columna FECHA PROCESO en el CSV del SRI y `load_all_years` lo
# descarta -- ver "Skipped years" en reports/.../08_feature_coverage.md. La
# cobertura real del dataset es 2018-2026, no 2017-2026.
DEFAULT_TRAIN_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
DEFAULT_VAL_YEARS = (2025,)
DEFAULT_TEST_YEARS = (2026,)


@dataclass(frozen=True)
class SplitSummary:
    """Cifras por partición, para el reporte y para detectar particiones vacías."""

    name: str
    n_episodes: int
    n_rows: int
    n_deferred_rows: int
    years: tuple[int, ...]

    @property
    def deferred_pct(self) -> float:
        return 100.0 * self.n_deferred_rows / self.n_rows if self.n_rows else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n_episodes": self.n_episodes,
            "n_rows": self.n_rows,
            "n_deferred_rows": self.n_deferred_rows,
            "deferred_pct": round(self.deferred_pct, 4),
            "years": list(self.years),
        }


def _require_columns(df: pd.DataFrame, expected: list[str], source: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{source} no tiene las columnas requeridas: {missing}")


def load_episode_tables(
    episodes_path: str | Path,
    vehicles_path: str | Path,
) -> pd.DataFrame:
    """Une las dos tablas por `episode_id` y devuelve una fila por vehículo.

    Cada fila resultante lleva ya la flota completa del episodio, que es lo que
    faltaba para que la asignación fuese aprendible.
    """
    episodes = pd.read_parquet(episodes_path)
    vehicles = pd.read_parquet(vehicles_path)

    _require_columns(episodes, EPISODE_COLUMNS, str(episodes_path))
    _require_columns(vehicles, VEHICLE_COLUMNS, str(vehicles_path))

    if episodes["episode_id"].duplicated().any():
        raise ValueError("episodes.parquet tiene episode_id duplicados")

    joined = vehicles.merge(
        episodes[EPISODE_COLUMNS],
        on="episode_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_episode"),
    )

    orphans = len(vehicles) - len(joined)
    if orphans:
        raise ValueError(
            f"{orphans} filas de episode_vehicles.parquet no encontraron su "
            "episodio. Las dos tablas provienen de corridas distintas de "
            "scripts/build_scenarios.py; regenerar ambas."
        )

    return joined


def drop_non_optimal(joined: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina episodios donde el maestro agotó su presupuesto de tiempo.

    `assign_vehicles` devuelve `optimal=False` cuando corta la búsqueda antes de
    probar la optimalidad. Esas etiquetas son la mejor solución encontrada, no la
    óptima, así que no sirven ni como objetivo de entrenamiento ni como referencia
    de evaluación.
    """
    mask = joined["optimal"].astype(bool)
    dropped = int(joined.loc[~mask, "episode_id"].nunique())
    return joined.loc[mask].reset_index(drop=True), dropped


def split_by_time(
    joined: pd.DataFrame,
    train_years: tuple[int, ...] = DEFAULT_TRAIN_YEARS,
    val_years: tuple[int, ...] = DEFAULT_VAL_YEARS,
    test_years: tuple[int, ...] = DEFAULT_TEST_YEARS,
) -> dict[str, pd.DataFrame]:
    """Partición temporal. Un episodio pertenece a un único año, así que la
    integridad del episodio es automática.

    Es la partición honesta para este problema: el modelo se entrena con el
    pasado y se evalúa con semanas que nunca vio, igual que en operación.
    """
    overlap = (
        set(train_years) & set(val_years)
        or set(train_years) & set(test_years)
        or set(val_years) & set(test_years)
    )
    if overlap:
        raise ValueError(f"Los años se solapan entre particiones: {sorted(overlap)}")

    splits = {
        "train": joined[joined["iso_year"].isin(train_years)],
        "val": joined[joined["iso_year"].isin(val_years)],
        "test": joined[joined["iso_year"].isin(test_years)],
    }
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def split_by_episode_hash(
    joined: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    salt: str = "juan-mlp",
) -> dict[str, pd.DataFrame]:
    """Partición agrupada por episodio, para cuando la temporal no aplica.

    La muestra de humo (`build_scenarios.py --limit 200`) cae entera en la semana
    2018-W02, así que una partición por año dejaría validación y prueba vacías.
    Esta alternativa reparte **episodios completos** con un hash estable -- no
    `hash()`, que Python aleatoriza por proceso -- de modo que la misma corrida
    da siempre la misma partición y ningún episodio se parte en dos.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"Las fracciones deben sumar 1.0, suman {sum(fractions)}")

    def bucket(episode_id: str) -> str:
        digest = hashlib.md5(f"{salt}:{episode_id}".encode()).hexdigest()
        u = int(digest[:8], 16) / 0xFFFFFFFF
        if u < fractions[0]:
            return "train"
        if u < fractions[0] + fractions[1]:
            return "val"
        return "test"

    assigned = joined["episode_id"].map(bucket)
    return {
        name: joined.loc[assigned == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }


def summarize_splits(splits: dict[str, pd.DataFrame]) -> list[SplitSummary]:
    summaries = []
    for name, df in splits.items():
        summaries.append(
            SplitSummary(
                name=name,
                n_episodes=int(df["episode_id"].nunique()) if len(df) else 0,
                n_rows=len(df),
                n_deferred_rows=int((~df["loaded"].astype(bool)).sum()) if len(df) else 0,
                years=tuple(sorted(df["iso_year"].unique().tolist())) if len(df) else (),
            )
        )
    return summaries


def assert_no_episode_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Falla ruidosamente si un episodio aparece en dos particiones."""
    seen: dict[str, str] = {}
    for name, df in splits.items():
        for episode_id in df["episode_id"].unique():
            previous = seen.get(episode_id)
            if previous is not None:
                raise AssertionError(
                    f"El episodio {episode_id!r} está en '{previous}' y en '{name}'."
                )
            seen[episode_id] = name
