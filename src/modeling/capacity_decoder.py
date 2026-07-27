"""src/modeling/capacity_decoder.py

Convierte las puntuaciones del MLP en un plan de distribución **factible**.

Por qué hace falta
------------------
El modelo puntúa cada vehículo por separado. Nada le impide preferir el mismo
camión para tres vehículos de 3.0 CU cuando ese camión tiene 6.0: cada decisión
aislada es razonable y la suma es inválida. El softmax no conoce la restricción
acoplada `sum(cu) <= capacidad`.

El decoder recorre los vehículos en un orden, prueba los camiones en el orden de
preferencia del modelo y asigna al primero que quepa. La invariante es dura: al
salir, ningún camión excede su capacidad. Ésa es la garantía que el clasificador
por sí solo no puede dar.

Sobre el orden y el diferimiento voluntario
-------------------------------------------
El maestro optimiza de forma lexicográfica: primero **cuántos** vehículos carga,
después cuántas unidades de almacenamiento aprovecha. Por eso las políticas
`model` y `count` nunca difieren un vehículo que quepa: bajo ese objetivo, cargar
más es siempre mejor, y honrar un `SIN_CAMION` predicho sólo puede empeorar la
métrica principal.

`respect_defer` sí lo honra, y existe para poder **medir** ese costo en lugar de
afirmarlo. La cabeza de diferimiento se entrena igual -- forma parte del softmax
-- y su margen se usa para ordenar y como diagnóstico.

Una segunda pasada de reparación sería inútil: la capacidad restante sólo
decrece, así que un vehículo que no cupo en el paso *t* tampoco cabe en *t+1*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DEFERRED = -1
_TOL = 1e-9

Policy = Literal["model", "count", "respect_defer"]
POLICIES: tuple[Policy, ...] = ("model", "count", "respect_defer")


@dataclass(frozen=True)
class DecodedEpisode:
    """Plan factible para un episodio.

    `assignment[i]` es el índice canónico del camión (0 = el de mayor capacidad)
    o `DEFERRED` si el vehículo queda diferido.
    """

    assignment: np.ndarray
    truck_loads: np.ndarray
    capacities: np.ndarray

    @property
    def n_loaded(self) -> int:
        return int((self.assignment != DEFERRED).sum())

    @property
    def n_deferred(self) -> int:
        return int((self.assignment == DEFERRED).sum())

    @property
    def cu_loaded(self) -> float:
        return float(self.truck_loads.sum())

    @property
    def utilization(self) -> float:
        total = float(self.capacities.sum())
        return self.cu_loaded / total if total > 0 else 0.0

    @property
    def max_overflow(self) -> float:
        """Cuánto se excedió el camión más sobrecargado. Debe ser 0."""
        return float(np.max(self.truck_loads - self.capacities, initial=0.0))

    @property
    def is_feasible(self) -> bool:
        return self.max_overflow <= _TOL


def split_logits(logits: np.ndarray, n_trucks: int) -> tuple[np.ndarray, np.ndarray]:
    """`(V, 1+T) -> (pair_logits (V,T), defer_logits (V,))`, índice 0 = diferir."""
    if logits.ndim != 2:
        raise ValueError(f"Se esperaban logits (V, 1+T), llegó {logits.shape}")
    if logits.shape[1] < n_trucks + 1:
        raise ValueError(
            f"Los logits tienen {logits.shape[1]} columnas pero la flota tiene "
            f"{n_trucks} camiones (se esperaban al menos {n_trucks + 1})."
        )
    return logits[:, 1 : n_trucks + 1], logits[:, 0]


def _vehicle_order(policy: Policy, cu: np.ndarray, margin: np.ndarray) -> np.ndarray:
    if policy == "count":
        # CU ascendente maximiza el conteo; el margen del modelo desempata.
        return np.lexsort((-margin, cu))
    return np.argsort(-margin, kind="stable")


def decode_episode(
    logits: np.ndarray,
    cu: np.ndarray,
    capacities: np.ndarray,
    policy: Policy = "count",
) -> DecodedEpisode:
    """Decodifica un episodio garantizando factibilidad."""
    if policy not in POLICIES:
        raise ValueError(f"Política desconocida: {policy!r}. Opciones: {POLICIES}")

    capacities = np.asarray(capacities, dtype=np.float64)
    cu = np.asarray(cu, dtype=np.float64)
    n_trucks = len(capacities)
    n_vehicles = len(cu)

    assignment = np.full(n_vehicles, DEFERRED, dtype=np.int32)
    remaining = capacities.copy()

    if n_trucks == 0 or n_vehicles == 0:
        return DecodedEpisode(assignment, np.zeros(n_trucks), capacities)

    pair_logits, defer_logits = split_logits(logits, n_trucks)
    margin = pair_logits.max(axis=1) - defer_logits

    for i in _vehicle_order(policy, cu, margin):
        if policy == "respect_defer" and margin[i] < 0:
            continue
        for j in np.argsort(-pair_logits[i], kind="stable"):
            if cu[i] <= remaining[j] + _TOL:
                assignment[i] = j
                remaining[j] -= cu[i]
                break

    decoded = DecodedEpisode(assignment, capacities - remaining, capacities)
    # Invariante dura: si esto falla, ninguna otra métrica importa.
    assert decoded.is_feasible, f"Capacidad excedida en {decoded.max_overflow:.6f} CU"
    return decoded


def greedy_first_fit_decreasing(cu: np.ndarray, capacities: np.ndarray) -> DecodedEpisode:
    """Línea base sin modelo: vehículo más grande primero, primer camión que quepa.

    Es la heurística que el reporte (Sec. I) describe como el enfoque manual
    habitual, y la que produce resultados subóptimos en el caso de estudio de
    `05_evaluation.md`.
    """
    capacities = np.asarray(capacities, dtype=np.float64)
    cu = np.asarray(cu, dtype=np.float64)
    assignment = np.full(len(cu), DEFERRED, dtype=np.int32)
    remaining = capacities.copy()

    for i in np.argsort(-cu, kind="stable"):
        for j in range(len(capacities)):
            if cu[i] <= remaining[j] + _TOL:
                assignment[i] = j
                remaining[j] -= cu[i]
                break

    decoded = DecodedEpisode(assignment, capacities - remaining, capacities)
    assert decoded.is_feasible
    return decoded
