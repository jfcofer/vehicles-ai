"""src/modeling/canonicalization.py

Reordena la flota de cada episodio por capacidad descendente y remapea las
etiquetas `CAMION_k` del maestro en consecuencia.

Por qué esto es necesario
-------------------------
`src/loading/scenarios.py:generate_fleet()` produce las capacidades en **orden
aleatorio**, y `src/loading/labeler.py:solve()` recorre los camiones por índice,
llenando el camión 0 tan lleno como pueda antes de pasar al 1. La consecuencia
medible en la muestra de 200 episodios es un reparto degenerado de etiquetas::

    CAMION_1  1966
    CAMION_2   640
    CAMION_3   106
    CAMION_4     3

`CAMION_1` no significa "el camión grande" ni "el primero de la ruta": significa
"el que salió primero del generador aleatorio". Dos episodios con flotas
idénticas salvo el orden producen etiquetas distintas para la misma decisión
operativa. Un clasificador no puede aprender esa distinción porque no está en sus
entradas -- sólo puede memorizar.

Tras canonicalizar, `CAMION_1` es siempre el camión de mayor capacidad, y la
etiqueta pasa a ser una función determinista de una característica que el modelo
sí observa. Las asignaciones resultantes son **operativamente idénticas**: sólo
cambia el nombre del camión, no qué vehículo viaja con cuál.

Nota de alcance: esto NO reetiqueta ni re-ejecuta el maestro. Es una permutación
de nombres aplicada aguas abajo, reversible con `inverse_label_map`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFER_LABEL = "SIN_CAMION"


def truck_label(index_zero_based: int) -> str:
    """`0 -> "CAMION_1"`, coincidiendo con la convención del maestro."""
    return f"CAMION_{index_zero_based + 1}"


def parse_truck_label(label: str) -> int | None:
    """`"CAMION_3" -> 2`; `"SIN_CAMION" -> None`."""
    if label == DEFER_LABEL:
        return None
    if not label.startswith("CAMION_"):
        raise ValueError(f"Etiqueta de camión no reconocida: {label!r}")
    return int(label.removeprefix("CAMION_")) - 1


@dataclass(frozen=True)
class CanonicalFleet:
    """Resultado de canonicalizar una flota.

    `capacities` queda ordenada de mayor a menor. `label_map` traduce la etiqueta
    del maestro a la etiqueta canónica; `order[p]` es el índice original del
    camión que ocupa la posición canónica `p`.
    """

    capacities: tuple[float, ...]
    order: tuple[int, ...]
    label_map: dict[str, str]

    @property
    def inverse_label_map(self) -> dict[str, str]:
        return {new: old for old, new in self.label_map.items()}

    @property
    def n_trucks(self) -> int:
        return len(self.capacities)


def canonicalize_fleet(capacities: Sequence[float]) -> CanonicalFleet:
    """Ordena por capacidad descendente; desempata por el índice original.

    El desempate por índice mantiene la operación determinista y estable: dos
    camiones de igual capacidad son intercambiables para el problema, así que
    cualquier criterio fijo sirve mientras sea siempre el mismo.
    """
    caps = [float(c) for c in capacities]
    order = sorted(range(len(caps)), key=lambda i: (-caps[i], i))

    label_map = {DEFER_LABEL: DEFER_LABEL}
    for new_pos, old_idx in enumerate(order):
        label_map[truck_label(old_idx)] = truck_label(new_pos)

    return CanonicalFleet(
        capacities=tuple(caps[i] for i in order),
        order=tuple(order),
        label_map=label_map,
    )


def canonical_target_index(label: str, fleet: CanonicalFleet) -> int:
    """Índice objetivo para la salida del modelo.

    `0` es `SIN_CAMION`; `1..n` son los camiones en orden canónico. Poner el
    diferimiento en la posición 0 -- y no al final -- hace que el índice objetivo
    sea independiente del relleno (*padding*) del lote: con 2 o con 8 camiones,
    `SIN_CAMION` sigue siendo 0 y `CAMION_1` sigue siendo 1.
    """
    canonical = fleet.label_map[label]
    truck_idx = parse_truck_label(canonical)
    if truck_idx is None:
        return 0
    if truck_idx >= fleet.n_trucks:
        raise ValueError(
            f"La etiqueta {label!r} apunta al camión {truck_idx} pero la flota "
            f"sólo tiene {fleet.n_trucks}."
        )
    return truck_idx + 1
