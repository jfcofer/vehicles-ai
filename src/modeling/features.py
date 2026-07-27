"""src/modeling/features.py

Construye los tensores que consume el MLP compartido por par (vehículo, camión).

Esquema
-------
Tres bloques, que el modelo recombina en dos cabezas::

    vehículo (6) : cu, one-hot(clase) x4, n_misma_clase
    camión   (3) : capacidad, capacidad/capacidad_total, capacidad/cu_total
    contexto(10) : n_vehiculos, n_camiones, cu_total, capacidad_total,
                   deficit_capacidad, ratio_utilizacion, conteo x4 por clase

    par     = vehículo ⊕ camión ⊕ contexto  -> 19
    diferir = vehículo ⊕ contexto           ->  16

Qué NO entra, y por qué
-----------------------
`uid` / `codigo_vehiculo`
    Identificadores. Sólo permiten memorizar.

`truck_id` (el número del camión)
    Es exactamente la etiqueta arbitraria que `canonicalization.py` elimina.
    Reintroducirla como entrada devolvería el problema por la puerta de atrás.

Posición del vehículo dentro de su clase
    `src/loading/labeler.py:224-229` baraja los vehículos de una misma clase con
    una semilla antes de repartir los cupos. Esa posición es ruido puro: dos
    vehículos con features idénticas reciben etiquetas distintas según el sorteo.
    Sí se conserva `n_misma_clase`, que es un conteo determinista.

Rango del camión por capacidad
    Tras canonicalizar, el rango es una función determinista de la capacidad, que
    el modelo ya recibe. Incluirlo re-introduciría identidad de posición y
    rompería la generalización a flotas más grandes que las vistas.

`canton`
    El maestro exacto lo ignora: `labeler.py:139-145` agrupa por clase, y el
    cantón no participa en la restricción de capacidad. Como entrada sólo puede
    aprender ruido o identidad de episodio. Se mide en ablación, no en la v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.modeling.canonicalization import canonical_target_index, canonicalize_fleet

VEHICLE_BLOCK = "vehicle"
TRUCK_BLOCK = "truck"
CONTEXT_BLOCK = "context"

_EPS = 1e-9


def feature_names(classes: list[str]) -> dict[str, list[str]]:
    """Nombres de columna por bloque, en el mismo orden que los tensores."""
    return {
        VEHICLE_BLOCK: ["cu", *[f"clase_{c.lower()}" for c in classes], "n_misma_clase"],
        TRUCK_BLOCK: ["capacidad", "capacidad_rel_flota", "capacidad_rel_demanda"],
        CONTEXT_BLOCK: [
            "n_vehiculos",
            "n_camiones",
            "cu_total",
            "capacidad_total",
            "deficit_capacidad",
            "ratio_utilizacion",
            *[f"conteo_{c.lower()}" for c in classes],
        ],
    }


def _scale_mask(classes: list[str]) -> dict[str, np.ndarray]:
    """Qué columnas se estandarizan. Los one-hot se dejan intactos."""
    n = len(classes)
    vehicle = np.array([True] + [False] * n + [True])
    return {
        VEHICLE_BLOCK: vehicle,
        TRUCK_BLOCK: np.ones(3, dtype=bool),
        CONTEXT_BLOCK: np.ones(6 + n, dtype=bool),
    }


@dataclass(frozen=True)
class EpisodeTensors:
    """Un episodio, ya canonicalizado y sin relleno."""

    episode_id: str
    vehicle: np.ndarray  # (V, Dv)
    truck: np.ndarray  # (T, Dt)
    context: np.ndarray  # (Dg,)
    target: np.ndarray  # (V,)  0 = SIN_CAMION, 1..T = camión canónico
    cu: np.ndarray  # (V,)
    class_index: np.ndarray  # (V,)  posición en `classes`, para métricas por clase
    capacities: np.ndarray  # (T,)  orden canónico (descendente)
    teacher_n_loaded: int
    teacher_cu_utilized: float

    @property
    def n_vehicles(self) -> int:
        return self.vehicle.shape[0]

    @property
    def n_trucks(self) -> int:
        return self.truck.shape[0]


def build_episode_tensors(episode_rows: pd.DataFrame, classes: list[str]) -> EpisodeTensors:
    """Construye los tensores de un episodio a partir de sus filas unidas.

    `episode_rows` son todas las filas del join de `dataset.load_episode_tables`
    que comparten `episode_id`. Los vehículos se ordenan por `uid` para que el
    resultado sea determinista corrida tras corrida.
    """
    rows = episode_rows.sort_values("uid").reset_index(drop=True)
    first = rows.iloc[0]

    fleet = canonicalize_fleet(list(first["truck_capacities"]))
    capacities = np.asarray(fleet.capacities, dtype=np.float64)

    cu = rows["cu"].to_numpy(dtype=np.float64)
    clase = rows["clase"].to_numpy()

    n_vehicles = len(rows)
    total_cu = float(cu.sum())
    total_capacity = float(capacities.sum())
    class_counts = np.array([(clase == c).sum() for c in classes], dtype=np.float64)
    same_class_count = np.array([class_counts[classes.index(c)] for c in clase])

    onehot = np.stack([(clase == c).astype(np.float64) for c in classes], axis=1)
    vehicle = np.concatenate([cu[:, None], onehot, same_class_count[:, None]], axis=1)

    truck = np.stack(
        [
            capacities,
            capacities / (total_capacity + _EPS),
            capacities / (total_cu + _EPS),
        ],
        axis=1,
    )

    context = np.concatenate(
        [
            np.array(
                [
                    n_vehicles,
                    len(capacities),
                    total_cu,
                    total_capacity,
                    total_cu - total_capacity,
                    total_cu / (total_capacity + _EPS),
                ],
                dtype=np.float64,
            ),
            class_counts,
        ]
    )

    target = np.array(
        [canonical_target_index(label, fleet) for label in rows["truck"]], dtype=np.int32
    )

    return EpisodeTensors(
        episode_id=str(first["episode_id"]),
        vehicle=vehicle,
        truck=truck,
        context=context,
        target=target,
        cu=cu,
        class_index=np.array([classes.index(c) for c in clase], dtype=np.int32),
        capacities=capacities,
        teacher_n_loaded=int(first["n_loaded"]),
        teacher_cu_utilized=float(first["cu_utilized"]),
    )


def build_all_episodes(joined: pd.DataFrame, classes: list[str]) -> list[EpisodeTensors]:
    """Un `EpisodeTensors` por episodio, en orden estable de `episode_id`."""
    return [
        build_episode_tensors(group, classes)
        for _, group in joined.groupby("episode_id", sort=True)
    ]


@dataclass
class BlockScaler:
    """Estandarización por bloque, ajustada **sólo con entrenamiento**.

    Se guarda en `feature_schema.json` para que evaluación e inferencia usen
    exactamente la misma transformación que el entrenamiento.
    """

    mean: dict[str, np.ndarray] = field(default_factory=dict)
    std: dict[str, np.ndarray] = field(default_factory=dict)
    mask: dict[str, np.ndarray] = field(default_factory=dict)
    names: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def fit(cls, episodes: list[EpisodeTensors], classes: list[str]) -> BlockScaler:
        stacked = {
            VEHICLE_BLOCK: np.concatenate([e.vehicle for e in episodes], axis=0),
            TRUCK_BLOCK: np.concatenate([e.truck for e in episodes], axis=0),
            CONTEXT_BLOCK: np.stack([e.context for e in episodes], axis=0),
        }
        mask = _scale_mask(classes)
        scaler = cls(mask=mask, names=feature_names(classes))
        for block, values in stacked.items():
            scaler.mean[block] = values.mean(axis=0)
            std = values.std(axis=0)
            # Una columna constante en entrenamiento no aporta señal; dividir por
            # su desviación (0) produciría NaN silenciosos.
            scaler.std[block] = np.where(std < _EPS, 1.0, std)
        return scaler

    def transform(self, block: str, values: np.ndarray) -> np.ndarray:
        mask = self.mask[block]
        out = values.astype(np.float64, copy=True)
        out[..., mask] = (out[..., mask] - self.mean[block][mask]) / self.std[block][mask]
        return out

    def to_dict(self) -> dict:
        return {
            block: {
                "names": self.names[block],
                "mean": self.mean[block].tolist(),
                "std": self.std[block].tolist(),
                "standardized": self.mask[block].tolist(),
            }
            for block in (VEHICLE_BLOCK, TRUCK_BLOCK, CONTEXT_BLOCK)
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BlockScaler:
        scaler = cls()
        for block, spec in payload.items():
            scaler.names[block] = spec["names"]
            scaler.mean[block] = np.asarray(spec["mean"], dtype=np.float64)
            scaler.std[block] = np.asarray(spec["std"], dtype=np.float64)
            scaler.mask[block] = np.asarray(spec["standardized"], dtype=bool)
        return scaler


@dataclass(frozen=True)
class ModelArrays:
    """Tensores rectangulares listos para `model.fit` / `model.predict`.

    El relleno hasta `max_trucks` es sólo para poder apilar episodios con flotas
    de distinto tamaño en un mismo lote. **No es un límite de la arquitectura**:
    el modelo declara su eje de camiones como `None`, así que en inferencia acepta
    cualquier `T` con los mismos pesos.
    """

    pair: np.ndarray  # (N, T, 19)
    defer: np.ndarray  # (N, 16)
    mask_bias: np.ndarray  # (N, T)   0 en camión real, -1e9 en relleno
    target: np.ndarray  # (N,)
    episode_index: np.ndarray  # (N,)  posición en la lista de episodios
    episode_ids: list[str]

    @property
    def max_trucks(self) -> int:
        return self.pair.shape[1]


MASK_NEG_INF = -1e9


def build_model_arrays(
    episodes: list[EpisodeTensors],
    scaler: BlockScaler,
    max_trucks: int | None = None,
) -> ModelArrays:
    """Apila los episodios en tensores rectangulares con relleno enmascarado."""
    if not episodes:
        raise ValueError("No hay episodios que apilar.")

    max_t = max_trucks or max(e.n_trucks for e in episodes)
    pair_dim = episodes[0].vehicle.shape[1] + episodes[0].truck.shape[1] + len(episodes[0].context)
    defer_dim = episodes[0].vehicle.shape[1] + len(episodes[0].context)
    n_rows = sum(e.n_vehicles for e in episodes)

    pair = np.zeros((n_rows, max_t, pair_dim), dtype=np.float32)
    defer = np.zeros((n_rows, defer_dim), dtype=np.float32)
    mask_bias = np.full((n_rows, max_t), MASK_NEG_INF, dtype=np.float32)
    target = np.zeros(n_rows, dtype=np.int32)
    episode_index = np.zeros(n_rows, dtype=np.int32)

    cursor = 0
    for ep_i, ep in enumerate(episodes):
        if ep.n_trucks > max_t:
            raise ValueError(
                f"El episodio {ep.episode_id} tiene {ep.n_trucks} camiones y el "
                f"relleno es de {max_t}."
            )
        v = scaler.transform(VEHICLE_BLOCK, ep.vehicle)
        t = scaler.transform(TRUCK_BLOCK, ep.truck)
        g = scaler.transform(CONTEXT_BLOCK, ep.context[None, :])[0]

        n_v, n_t = ep.n_vehicles, ep.n_trucks
        rows = slice(cursor, cursor + n_v)

        # (V, T, Dv+Dt+Dg): el vehículo y el contexto se repiten por cada camión.
        pair[rows, :n_t, :] = np.concatenate(
            [
                np.broadcast_to(v[:, None, :], (n_v, n_t, v.shape[1])),
                np.broadcast_to(t[None, :, :], (n_v, n_t, t.shape[1])),
                np.broadcast_to(g[None, None, :], (n_v, n_t, g.shape[0])),
            ],
            axis=2,
        ).astype(np.float32)

        defer[rows] = np.concatenate(
            [v, np.broadcast_to(g[None, :], (n_v, g.shape[0]))], axis=1
        ).astype(np.float32)

        mask_bias[rows, :n_t] = 0.0
        target[rows] = ep.target
        episode_index[rows] = ep_i
        cursor += n_v

    return ModelArrays(
        pair=pair,
        defer=defer,
        mask_bias=mask_bias,
        target=target,
        episode_index=episode_index,
        episode_ids=[e.episode_id for e in episodes],
    )


def as_model_inputs(arrays: ModelArrays) -> dict[str, np.ndarray]:
    return {
        "pair_features": arrays.pair,
        "defer_features": arrays.defer,
        "mask_bias": arrays.mask_bias,
    }


def balanced_sample_weights(target: np.ndarray) -> np.ndarray:
    """Peso por fila que compensa el desbalance cargado / diferido.

    El reporte (Sec. VI-A) menciona un factor de 33 para `SIN_CAMION`, pero ese
    número corresponde al reparto de otro modelo. Aquí se **mide** sobre la
    partición de entrenamiento real: en la muestra de 200 episodios los diferidos
    son 179 de 2.894 filas (6,2%), no 1 de 33.
    """
    deferred = target == 0
    n_deferred = int(deferred.sum())
    n_loaded = int((~deferred).sum())
    if n_deferred == 0 or n_loaded == 0:
        return np.ones_like(target, dtype=np.float32)

    total = n_deferred + n_loaded
    w_deferred = total / (2.0 * n_deferred)
    w_loaded = total / (2.0 * n_loaded)
    return np.where(deferred, w_deferred, w_loaded).astype(np.float32)
