#!/usr/bin/env python3
"""¿Cuál es la exactitud cruda máxima que cualquiera de los cinco modelos puede sacar?

Motivación
----------
`scripts/teacher_self_agreement.py` mide cuánto reproduce el maestro de sí mismo
(0,3983 sobre 2026) y se venía usando como techo. Pero esa cifra **no es un techo
estricto**: predecir la moda de una variable aleatoria coincide con una muestra
más a menudo de lo que dos muestras independientes coinciden entre sí. Es decir,
el techo real está por encima de 0,3983 y no sabíamos dónde.

Este script lo calcula en forma cerrada, sin entrenar nada y sin modelo, leyendo
sólo las etiquetas ya generadas. Dos vehículos de la misma clase en el mismo
episodio tienen features idénticas; de ahí salen dos cotas superiores exactas
(ver `src/modeling/metrics.py:label_ceilings` para la derivación):

    cota A -- modelo determinista por vehículo (argmax)
    cota B -- pipeline con decodificador, que reparte la clase pero no sabe
              *cuál* vehículo concreto diferir

Sirve para los cinco modelos del grupo, no sólo para el MLP: la cota no depende
de la arquitectura, sólo de las etiquetas.

Uso (desde la raíz del repositorio):
    uv run python scripts/label_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.modeling.canonicalization import canonical_target_index, canonicalize_fleet  # noqa: E402
from src.modeling.dataset import (  # noqa: E402
    drop_non_optimal,
    load_episode_tables,
    split_by_time,
)
from src.modeling.metrics import label_ceilings  # noqa: E402

DEFAULT_EPISODES_DIR = REPO_ROOT / "data" / "episodes"
CLASSES = ["AUTOMOVIL", "CAMIONETA", "JEEP", "MOTOCICLETA"]


def ceilings_for(df: pd.DataFrame, canonical: bool) -> dict:
    """Cotas sobre una partición ya unida (`load_episode_tables`).

    `canonical=False` deja las etiquetas tal como salieron del maestro; sirve
    para comprobar empíricamente que la canonicalización **no mueve el techo**.
    """
    episode_index, _ = pd.factorize(df["episode_id"], sort=True)
    class_index = pd.Categorical(df["clase"], categories=CLASSES).codes
    if (class_index < 0).any():
        unknown = sorted(set(df["clase"]) - set(CLASSES))
        raise ValueError(f"Clases no contempladas en CLASSES: {unknown}")

    targets = np.empty(len(df), dtype=np.int64)
    for _, rows in df.groupby("episode_id", sort=False):
        caps = list(rows["truck_capacities"].iloc[0])
        fleet = canonicalize_fleet(caps)
        positions = rows.index.to_numpy()
        if canonical:
            targets[positions] = [canonical_target_index(t, fleet) for t in rows["truck"]]
        else:
            targets[positions] = [
                0 if t == "SIN_CAMION" else int(t.removeprefix("CAMION_")) for t in rows["truck"]
            ]

    return label_ceilings(
        target_index=targets,
        class_index=class_index.astype(np.int64),
        episode_index=episode_index.astype(np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "artifacts" / "mlp" / "label_ceilings.json"
    )
    parser.add_argument(
        "--measured-accuracy",
        type=float,
        default=None,
        help="Exactitud cruda medida de un modelo, para contrastarla con el techo. "
        "Por defecto se lee de artifacts/mlp/metrics.json si existe.",
    )
    args = parser.parse_args()

    joined = load_episode_tables(
        args.episodes_dir / "episodes.parquet",
        args.episodes_dir / "episode_vehicles.parquet",
    )
    joined, dropped = drop_non_optimal(joined)

    payload: dict = {
        "n_episodes_total": int(joined["episode_id"].nunique()),
        "n_rows_total": int(len(joined)),
        "n_episodes_dropped_non_optimal": dropped,
        "global": ceilings_for(joined.reset_index(drop=True), canonical=True),
        "global_sin_canonicalizar": ceilings_for(joined.reset_index(drop=True), canonical=False),
        "por_particion": {},
    }
    for name, part in split_by_time(joined).items():
        if len(part):
            payload["por_particion"][name] = ceilings_for(
                part.reset_index(drop=True), canonical=True
            )

    measured = args.measured_accuracy
    if measured is None:
        metrics_path = args.out.parent / "metrics.json"
        if metrics_path.exists():
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            measured = saved.get("model", {}).get("test", {}).get("raw_assignment_accuracy")
    payload["exactitud_medida_mlp_prueba"] = measured

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    g = payload["global"]
    print(f"Episodios: {payload['n_episodes_total']:,}   Filas: {payload['n_rows_total']:,}")
    print()
    print("TECHO EXACTO de exactitud cruda (dataset completo, etiquetas canónicas):")
    print(f"  Cota A -- modelo determinista por vehículo   {g['ceiling_argmax_micro']:.4f}")
    print(f"  Cota B -- pipeline con decodificador         {g['ceiling_decoder_micro']:.4f}")
    print()
    print("CONTROL -- las mismas cotas SIN canonicalizar (deben ser idénticas):")
    ng = payload["global_sin_canonicalizar"]
    print(
        f"  Cota A   {ng['ceiling_argmax_micro']:.4f}"
        f"     Cota B   {ng['ceiling_decoder_micro']:.4f}"
    )
    print()
    for name, part in payload["por_particion"].items():
        print(
            f"  {name:<6} A={part['ceiling_argmax_micro']:.4f}  "
            f"B={part['ceiling_decoder_micro']:.4f}  "
            f"({part['n_episodes']:,} episodios)"
        )
    if measured is not None:
        print()
        print(f"Exactitud cruda medida del MLP (prueba 2026): {measured:.4f}")
    print(f"\nEscrito en {args.out}")


if __name__ == "__main__":
    main()
