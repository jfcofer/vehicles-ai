#!/usr/bin/env python3
"""¿Cuánta exactitud es alcanzable, como máximo, sobre estas etiquetas?

Motivación
----------
La exactitud cruda del MLP se queda en ~0.53 y la concordancia por clase en
~0.55. Antes de culpar al modelo hay que preguntarse si la etiqueta es
predecible en absoluto.

El maestro exacto recibe la flota en el orden en que la escupió el generador
aleatorio, y su programación dinámica llena el camión de índice 0 tan lleno como
puede antes de pasar al siguiente (`labeler.py:188-213`, con `range(max_x, -1, -1)`
y comparación estricta). Ese índice 0 es un camión de capacidad *aleatoria*.
Además, dentro de una clase reparte los cupos con un `random.shuffle` sembrado.

Este script mide el techo empírico: se le presenta al maestro **la misma flota
permutada** -- una situación operativamente idéntica, mismo conjunto de camiones,
mismas capacidades -- y se compara su nueva respuesta con la original, ambas
canonicalizadas por capacidad. Lo que el maestro no reproduce de sí mismo, ningún
modelo puede predecirlo: es ruido de desempate, no señal.

Uso (desde la raíz del repositorio):
    uv run python scripts/teacher_self_agreement.py --years 2026
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.loading.labeler import Vehicle, assign_vehicles  # noqa: E402
from src.modeling.canonicalization import canonical_target_index, canonicalize_fleet  # noqa: E402

DEFAULT_EPISODES_DIR = REPO_ROOT / "data" / "episodes"
CLASSES = ["AUTOMOVIL", "CAMIONETA", "JEEP", "MOTOCICLETA"]


def class_agreement(a: np.ndarray, b: np.ndarray, classes: np.ndarray, n_slots: int) -> float:
    """1 - distancia de variación total entre dos planes, por (camión, clase)."""
    left = np.zeros((n_slots, len(CLASSES)), dtype=int)
    right = np.zeros((n_slots, len(CLASSES)), dtype=int)
    np.add.at(left, (a, classes), 1)
    np.add.at(right, (b, classes), 1)
    return 1.0 - float(np.abs(left - right).sum()) / (2.0 * len(a))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "mlp" / "teacher_self_agreement.json",
        help="Ruta del JSON de salida. Explícita a propósito: derivarla de --episodes-dir "
        "escribe fuera de artifacts/ cuando se apunta a un conjunto de extrapolación.",
    )
    parser.add_argument("--years", type=int, nargs="*", default=[2026])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    episodes = pd.read_parquet(args.episodes_dir / "episodes.parquet")
    vehicles = pd.read_parquet(args.episodes_dir / "episode_vehicles.parquet")

    keep = episodes[episodes["iso_year"].isin(args.years)]
    if args.limit:
        keep = keep.head(args.limit)
    meta = keep.set_index("episode_id")
    vehicles = vehicles[vehicles["episode_id"].isin(set(keep["episode_id"]))]

    rng = random.Random(args.seed)
    raw_scores, class_scores, identical = [], [], 0
    loaded_deltas, cu_deltas = [], []

    for episode_id, group in vehicles.groupby("episode_id", sort=True):
        row = meta.loc[episode_id]
        original_caps = list(row["truck_capacities"])
        if len(original_caps) < 2:
            continue  # con un solo camión no hay permutación posible

        order = list(range(len(original_caps)))
        rng.shuffle(order)
        permuted_caps = [original_caps[i] for i in order]

        veh = [Vehicle(uid=r.uid, clase=r.clase, cu=r.cu) for r in group.itertuples()]
        redo = assign_vehicles(veh, permuted_caps, time_budget_s=5.0, seed=rng.randrange(2**31))

        rows = group.sort_values("uid")
        fleet_original = canonicalize_fleet(original_caps)
        fleet_permuted = canonicalize_fleet(permuted_caps)

        a = np.array([canonical_target_index(t, fleet_original) for t in rows["truck"]], dtype=int)
        b = np.array(
            [canonical_target_index(redo.assignment[uid], fleet_permuted) for uid in rows["uid"]],
            dtype=int,
        )
        classes = np.array([CLASSES.index(c) for c in rows["clase"]], dtype=int)

        raw_scores.append(float((a == b).mean()))
        score = class_agreement(a, b, classes, len(original_caps) + 1)
        class_scores.append(score)
        identical += int(score == 1.0)
        loaded_deltas.append(abs(int(row["n_loaded"]) - redo.n_loaded))
        cu_deltas.append(abs(float(row["cu_utilized"]) - redo.cu_utilized))

    raw = np.asarray(raw_scores)
    cls = np.asarray(class_scores)
    payload = {
        "years": args.years,
        "n_episodes_compared": int(len(raw)),
        "teacher_raw_self_accuracy_mean": float(raw.mean()),
        "teacher_class_level_self_agreement_mean": float(cls.mean()),
        "episodes_reproduced_identically_pct": float(100.0 * identical / len(raw)),
        "n_loaded_absolute_delta_mean": float(np.mean(loaded_deltas)),
        "cu_utilized_absolute_delta_mean": float(np.mean(cu_deltas)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Episodios comparados (n_camiones >= 2): {len(raw):,}")
    print()
    print("TECHO EMPÍRICO -- el maestro contra sí mismo, misma flota en otro orden:")
    print(f"  Exactitud cruda reproducida        {raw.mean():.4f}")
    print(f"  Concordancia por clase             {cls.mean():.4f}")
    print(f"  Episodios reproducidos idénticos   {100 * identical / len(raw):.2f}%")
    print()
    print("CONTROL -- lo que sí es determinista (el objetivo real):")
    print(f"  |Δ vehículos cargados| medio       {np.mean(loaded_deltas):.4f}")
    print(f"  |Δ CU aprovechada| medio           {np.mean(cu_deltas):.4f}")
    print(f"\nEscrito en {args.out}")


if __name__ == "__main__":
    main()
