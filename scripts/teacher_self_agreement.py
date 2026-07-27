#!/usr/bin/env python3
"""¿Cuánto de la etiqueta depende del orden en que llegó la flota?

Motivación
----------
La exactitud cruda del MLP se queda en ~0.53 y la concordancia por clase en
~0.55. Antes de culpar al modelo hay que preguntarse cuánto de la etiqueta es
reproducible siquiera por quien la generó.

(El techo *absoluto* de exactitud se calcula aparte, en forma cerrada, con
`scripts/label_ceiling.py`. Lo que mide este script es distinto: cuánta de la
discrepancia viene del orden de la flota y desaparece al fijarlo.)

El maestro exacto recibe la flota en el orden en que la escupió el generador
aleatorio, y su programación dinámica llena el camión de índice 0 tan lleno como
puede antes de pasar al siguiente (`labeler.py:188-213`, con `range(max_x, -1, -1)`
y comparación estricta). Ese índice 0 es un camión de capacidad *aleatoria*.
Además, dentro de una clase reparte los cupos con un `random.shuffle` sembrado.

Este script mide cuánto se reproduce el maestro a sí mismo: se le presenta **la
misma flota permutada** -- una situación operativamente idéntica, mismo conjunto
de camiones, mismas capacidades -- y se compara su nueva respuesta con la
original, ambas canonicalizadas por capacidad.

`--fleet-order` es el experimento decisivo
------------------------------------------
Canonicalizar la salida sólo arregla el **nombre** del camión. No arregla el
**plan**: con la flota [6,0 · 4,0] la PD llena el camión de índice 0 (el grande) y
con [4,0 · 6,0] llena el chico, y los dos planes resultantes son igual de óptimos
pero no se corresponden por ningún renombramiento.

Con `--fleet-order desc|asc` la flota se ordena **antes** de etiquetar, de modo
que cualquier permutación produce exactamente la misma entrada al maestro. Lo que
quede de discrepancia es entonces sólo el reparto intra-clase (`labeler.py:221-229`),
que es el único ruido verdaderamente irreducible.

Uso (desde la raíz del repositorio):
    uv run python scripts/teacher_self_agreement.py --years 2026
    uv run python scripts/teacher_self_agreement.py --years 2026 --fleet-order desc
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
    parser.add_argument(
        "--fleet-order",
        choices=["as-is", "desc", "asc"],
        default="as-is",
        help="`as-is` reproduce el comportamiento actual del pipeline. `desc`/`asc` "
        "ordenan la flota ANTES de etiquetar, que es el arreglo propuesto para "
        "src/loading/scenarios.py -- ver el docstring del módulo.",
    )
    args = parser.parse_args()

    def order_fleet(caps: list[float]) -> list[float]:
        if args.fleet_order == "desc":
            return sorted(caps, reverse=True)
        if args.fleet_order == "asc":
            return sorted(caps)
        return list(caps)

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
    label_counts: dict[int, int] = {}

    for episode_id, group in vehicles.groupby("episode_id", sort=True):
        row = meta.loc[episode_id]
        original_caps = list(row["truck_capacities"])
        if len(original_caps) < 2:
            continue  # con un solo camión no hay permutación posible

        order = list(range(len(original_caps)))
        rng.shuffle(order)
        permuted_caps = [original_caps[i] for i in order]

        veh = [Vehicle(uid=r.uid, clase=r.clase, cu=r.cu) for r in group.itertuples()]
        rows = group.sort_values("uid")

        # Lado de referencia. Con `as-is` son las etiquetas ya guardadas en el
        # parquet; con un orden fijo hay que reetiquetar para comparar peras con
        # peras, porque el parquet se generó con la flota desordenada.
        if args.fleet_order == "as-is":
            reference_caps = original_caps
            reference_labels = list(rows["truck"])
            reference_loaded = int(row["n_loaded"])
            reference_cu = float(row["cu_utilized"])
        else:
            reference_caps = order_fleet(original_caps)
            ref = assign_vehicles(veh, reference_caps, time_budget_s=5.0, seed=rng.randrange(2**31))
            reference_labels = [ref.assignment[uid] for uid in rows["uid"]]
            reference_loaded, reference_cu = ref.n_loaded, ref.cu_utilized

        redo_caps = order_fleet(permuted_caps)
        redo = assign_vehicles(veh, redo_caps, time_budget_s=5.0, seed=rng.randrange(2**31))

        fleet_original = canonicalize_fleet(reference_caps)
        fleet_permuted = canonicalize_fleet(redo_caps)

        a = np.array(
            [canonical_target_index(t, fleet_original) for t in reference_labels], dtype=int
        )
        b = np.array(
            [canonical_target_index(redo.assignment[uid], fleet_permuted) for uid in rows["uid"]],
            dtype=int,
        )
        classes = np.array([CLASSES.index(c) for c in rows["clase"]], dtype=int)
        for label in a:
            label_counts[int(label)] = label_counts.get(int(label), 0) + 1

        raw_scores.append(float((a == b).mean()))
        score = class_agreement(a, b, classes, len(original_caps) + 1)
        class_scores.append(score)
        identical += int(score == 1.0)
        loaded_deltas.append(abs(reference_loaded - redo.n_loaded))
        cu_deltas.append(abs(reference_cu - redo.cu_utilized))

    raw = np.asarray(raw_scores)
    cls = np.asarray(class_scores)
    total_labels = sum(label_counts.values())
    payload = {
        "years": args.years,
        "fleet_order": args.fleet_order,
        "label_distribution_pct": {
            ("SIN_CAMION" if k == 0 else f"CAMION_{k}"): round(100.0 * v / total_labels, 4)
            for k, v in sorted(label_counts.items())
        },
        "n_episodes_compared": int(len(raw)),
        "teacher_raw_self_accuracy_mean": float(raw.mean()),
        "teacher_class_level_self_agreement_mean": float(cls.mean()),
        "episodes_reproduced_identically_pct": float(100.0 * identical / len(raw)),
        "n_loaded_absolute_delta_mean": float(np.mean(loaded_deltas)),
        "cu_utilized_absolute_delta_mean": float(np.mean(cu_deltas)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Episodios comparados (n_camiones >= 2): {len(raw):,}   orden: {args.fleet_order}")
    print()
    print("Reparto de etiquetas canónicas del lado de referencia:")
    for label, pct in payload["label_distribution_pct"].items():
        print(f"  {label:<12} {pct:6.2f} %")
    print()
    print("EL MAESTRO CONTRA SÍ MISMO, misma flota en otro orden:")
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
