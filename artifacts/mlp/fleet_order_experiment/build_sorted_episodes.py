#!/usr/bin/env python3
"""Genera el dataset con la flota ORDENADA antes de etiquetar, a un directorio aparte.

Es la verificación del parche propuesto para `src/loading/scenarios.py`
(ver `docs/tarea4/06_canonicalizacion_y_etiquetado.md`). En vez de aplicar el
parche al repositorio -- lo que invalidaría los modelos ya entrenados por el
equipo -- se monkeypatchea `generate_fleet` aquí y se mide el efecto.

**No toca `data/episodes/` ni el reporte autogenerado.**

    uv run python artifacts/mlp/fleet_order_experiment/build_sorted_episodes.py \
        --order asc --out /tmp/episodes_asc
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.loading import scenarios  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--order", choices=["desc", "asc"], default="asc")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

reverse = args.order == "desc"


def generate_fleet_sorted(rng: random.Random) -> list[float]:
    """Idéntico al original salvo el `sorted` final.

    Consume exactamente las mismas extracciones del RNG y en el mismo orden, así
    que el resto del episodio -- el submuestreo previo y la semilla del
    etiquetador posterior -- no cambia en absoluto. Eso es lo que hace que el
    diff contra el dataset actual quede acotado a la columna `truck`.
    """
    n_trucks = rng.randint(*scenarios.N_TRUCKS_RANGE)
    caps = [round(rng.uniform(*scenarios.CAP_RANGE), 1) for _ in range(n_trucks)]
    return sorted(caps, reverse=reverse)


scenarios.generate_fleet = generate_fleet_sorted

df = pd.read_parquet(REPO_ROOT / "data" / "features" / "vehicles_in_scope.parquet")
t0 = time.perf_counter()
episodes_df, vehicles_df, summary = scenarios.build_all_episodes(df, limit=args.limit)
elapsed = time.perf_counter() - t0

args.out.mkdir(parents=True, exist_ok=True)
episodes_df.to_parquet(args.out / "episodes.parquet", index=False)
vehicles_df.to_parquet(args.out / "episode_vehicles.parquet", index=False)

print(
    f"orden={args.order}  episodios={summary.n_episodes_built:,}  "
    f"filas={len(vehicles_df):,}  ({elapsed:.1f}s)"
)
print(f"Escrito en {args.out}")
