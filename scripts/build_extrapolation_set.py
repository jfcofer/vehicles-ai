#!/usr/bin/env python3
"""Genera conjuntos de prueba con flotas MÁS GRANDES que las del entrenamiento.

El generador de episodios usa `N_TRUCKS_RANGE = (1, 4)`, así que el modelo nunca
ve más de cuatro camiones. La arquitectura por pares **puede** atender cualquier
`n` con los mismos pesos, pero eso es una propiedad del diseño, no un resultado
medido. Este script produce la evidencia.

Toma los manifiestos que el modelo ya iba a evaluar (mismos vehículos, mismas
clases, mismos CU), les cambia sólo la flota, y **reetiqueta con el maestro
exacto** `src/loading/labeler.assign_vehicles` -- barato para N<=20. Así la
comparación aísla una única variable: la cantidad de camiones.

No modifica `src/loading/scenarios.py` ni el dataset principal.

Uso (desde la raíz del repositorio):
    uv run python scripts/build_extrapolation_set.py --n-trucks 5 6
    uv run python scripts/build_extrapolation_set.py --n-trucks 8 10 --cap-mode constant-total
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.loading.labeler import Vehicle, assign_vehicles  # noqa: E402
from src.loading.scenarios import CAP_RANGE, N_TRUCKS_RANGE  # noqa: E402

DEFAULT_EPISODES_DIR = REPO_ROOT / "data" / "episodes"
TRAIN_MAX_TRUCKS = N_TRUCKS_RANGE[1]


def extrapolation_seed(episode_id: str, n_trucks: int) -> int:
    """Semilla estable e independiente de PYTHONHASHSEED, como en scenarios.py."""
    key = f"extrapolation:{episode_id}:{n_trucks}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def make_fleet(rng: random.Random, n_trucks: int, cap_mode: str) -> list[float]:
    """`same`: idéntica distribución de capacidad que el entrenamiento; la
    capacidad total crece con `n`, así que el aumento de camiones es la única
    variable nueva.

    `constant-total`: la capacidad total se mantiene en la banda que el modelo
    vio (la de una flota de 4), repartida entre más camiones. Es el escenario
    difícil: más contenedores para el mismo espacio.
    """
    low, high = CAP_RANGE
    if cap_mode == "constant-total":
        factor = TRAIN_MAX_TRUCKS / n_trucks
        low, high = low * factor, high * factor
    return [round(rng.uniform(low, high), 2) for _ in range(n_trucks)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--n-trucks", type=int, nargs="+", required=True)
    parser.add_argument("--cap-mode", choices=("same", "constant-total"), default="same")
    parser.add_argument("--years", type=int, nargs="*", default=[2026])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    episodes = pd.read_parquet(args.episodes_dir / "episodes.parquet")
    vehicles = pd.read_parquet(args.episodes_dir / "episode_vehicles.parquet")

    keep = episodes[episodes["iso_year"].isin(args.years)]
    if args.limit:
        keep = keep.head(args.limit)
    vehicles = vehicles[vehicles["episode_id"].isin(set(keep["episode_id"]))]

    tag = f"extrap_{'_'.join(map(str, args.n_trucks))}_{args.cap_mode.replace('-', '')}"
    out_dir = args.out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_records, vehicle_records = [], []
    meta = keep.set_index("episode_id")

    for episode_id, group in vehicles.groupby("episode_id", sort=True):
        rng = random.Random(extrapolation_seed(episode_id, args.n_trucks[0]))
        n_trucks = rng.choice(args.n_trucks)
        fleet = make_fleet(rng, n_trucks, args.cap_mode)

        veh = [Vehicle(uid=r.uid, clase=r.clase, cu=r.cu) for r in group.itertuples()]
        result = assign_vehicles(veh, fleet, time_budget_s=5.0, seed=rng.randrange(2**31))

        row = meta.loc[episode_id]
        episode_records.append(
            {
                "episode_id": episode_id,
                "iso_year": int(row["iso_year"]),
                "iso_week": int(row["iso_week"]),
                "canton": int(row["canton"]),
                "n_sampled": len(group),
                "n_trucks": len(fleet),
                "truck_capacities": fleet,
                "n_loaded": result.n_loaded,
                "n_deferred": result.n_deferred,
                "cu_utilized": result.cu_utilized,
                "optimal": result.optimal,
            }
        )
        for r in group.itertuples():
            truck = result.assignment[r.uid]
            vehicle_records.append(
                {
                    "episode_id": episode_id,
                    "uid": r.uid,
                    "clase": r.clase,
                    "cu": r.cu,
                    "canton": r.canton,
                    "truck": truck,
                    "loaded": truck != "SIN_CAMION",
                }
            )

    ep_df = pd.DataFrame(episode_records)
    ve_df = pd.DataFrame(vehicle_records)
    ep_df.to_parquet(out_dir / "episodes.parquet", index=False)
    ve_df.to_parquet(out_dir / "episode_vehicles.parquet", index=False)

    non_optimal = int((~ep_df["optimal"]).sum())
    print(f"Camiones: {sorted(args.n_trucks)}  modo de capacidad: {args.cap_mode}")
    print(f"Episodios: {len(ep_df):,}  filas: {len(ve_df):,}  no-óptimos: {non_optimal}")
    print(f"Capacidad total media: {ep_df['truck_capacities'].apply(sum).mean():.2f} CU")
    print(
        f"Diferidos: {int(ep_df['n_deferred'].sum()):,} "
        f"({100 * (~ve_df['loaded']).mean():.2f}% de las filas)"
    )
    print(f"Escrito en {out_dir}")


if __name__ == "__main__":
    main()
