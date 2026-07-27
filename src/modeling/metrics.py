"""src/modeling/metrics.py

Métricas a nivel de episodio, contra el maestro exacto.

Por qué no basta la exactitud de asignación
-------------------------------------------
Dos planes pueden ser operativamente idénticos y discrepar en la etiqueta de
cada vehículo: si dos camiones tienen la misma capacidad, intercambiarlos no
cambia nada para el operador, pero la exactitud cruda lo cuenta como error.
Al revés, un plan con exactitud alta puede exceder la capacidad y ser inservible.

Por eso el orden de reporte es el del dominio:

1. tasa de violación de capacidad -- debe ser 0, o nada más importa;
2. brecha de vehículos cargados frente al maestro (objetivo primario);
3. brecha de CU aprovechada (objetivo secundario);
4. vehículos diferidos;
5. F1 macro y matriz de confusión;
6. latencia de inferencia;
7. exactitud cruda, como diagnóstico.

El maestro ya dejó `n_loaded` y `cu_utilized` por episodio en `episodes.parquet`,
así que la verdad de terreno no cuesta nada recalcularla.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.modeling.capacity_decoder import (
    DEFERRED,
    DecodedEpisode,
    Policy,
    decode_episode,
    greedy_first_fit_decreasing,
)
from src.modeling.features import EpisodeTensors, ModelArrays

_TOL = 1e-9


@dataclass(frozen=True)
class EpisodeResult:
    """Resultado de un episodio: lo que hizo el plan frente a lo que era óptimo."""

    episode_id: str
    n_vehicles: int
    n_trucks: int
    total_capacity: float
    model_n_loaded: int
    teacher_n_loaded: int
    model_cu: float
    teacher_cu: float
    max_overflow: float
    predicted_index: np.ndarray  # (V,) 0 = SIN_CAMION, 1..T = camión canónico
    target_index: np.ndarray  # (V,) idem, del maestro
    class_index: np.ndarray  # (V,) clase de cada vehículo
    n_classes: int

    @property
    def loaded_gap(self) -> int:
        """Vehículos que el maestro cargó y el plan no. Nunca debería ser < 0."""
        return self.teacher_n_loaded - self.model_n_loaded

    @property
    def cu_gap(self) -> float:
        return self.teacher_cu - self.model_cu

    @property
    def matches_teacher_count(self) -> bool:
        return self.loaded_gap == 0

    @property
    def is_feasible(self) -> bool:
        return self.max_overflow <= _TOL


def _to_target_index(decoded: DecodedEpisode) -> np.ndarray:
    """`-1 -> 0` (diferido); `j -> j+1` (camión canónico j)."""
    return np.where(decoded.assignment == DEFERRED, 0, decoded.assignment + 1).astype(np.int32)


def episode_logits(logits: np.ndarray, rows: np.ndarray, n_trucks: int) -> np.ndarray:
    """Recorta los logits de un episodio a sus camiones reales.

    Las columnas de relleno vienen con `-1e9` sumado; recortarlas evita que un
    argsort las considere siquiera.
    """
    return logits[rows][:, : n_trucks + 1]


def build_result(
    episode: EpisodeTensors, decoded: DecodedEpisode, target: np.ndarray, n_classes: int
) -> EpisodeResult:
    return EpisodeResult(
        class_index=episode.class_index,
        n_classes=n_classes,
        episode_id=episode.episode_id,
        n_vehicles=episode.n_vehicles,
        n_trucks=episode.n_trucks,
        total_capacity=float(episode.capacities.sum()),
        model_n_loaded=decoded.n_loaded,
        teacher_n_loaded=episode.teacher_n_loaded,
        model_cu=decoded.cu_loaded,
        teacher_cu=episode.teacher_cu_utilized,
        max_overflow=decoded.max_overflow,
        predicted_index=_to_target_index(decoded),
        target_index=target,
    )


def evaluate_model(
    episodes: list[EpisodeTensors],
    arrays: ModelArrays,
    logits: np.ndarray,
    policy: Policy = "count",
    n_classes: int = 4,
) -> list[EpisodeResult]:
    """Decodifica y evalúa cada episodio con las puntuaciones del modelo."""
    results = []
    for ep_i, episode in enumerate(episodes):
        rows = np.flatnonzero(arrays.episode_index == ep_i)
        decoded = decode_episode(
            episode_logits(logits, rows, episode.n_trucks),
            cu=episode.cu,
            capacities=episode.capacities,
            policy=policy,
        )
        results.append(build_result(episode, decoded, arrays.target[rows], n_classes))
    return results


def evaluate_greedy(
    episodes: list[EpisodeTensors], arrays: ModelArrays, n_classes: int = 4
) -> list[EpisodeResult]:
    """Línea base sin modelo: primer ajuste, vehículo más grande primero."""
    results = []
    for ep_i, episode in enumerate(episodes):
        rows = np.flatnonzero(arrays.episode_index == ep_i)
        decoded = greedy_first_fit_decreasing(episode.cu, episode.capacities)
        results.append(build_result(episode, decoded, arrays.target[rows], n_classes))
    return results


def class_level_agreement(results: list[EpisodeResult]) -> dict:
    """Concordancia con el maestro **invariante a qué vehículo concreto** viaja.

    El maestro resuelve el problema en conteos por clase y sólo al final reparte
    los vehículos individuales de cada clase con un `random.shuffle` sembrado
    (`labeler.py:224-229`). Dos vehículos de la misma clase tienen exactamente
    las mismas features y CU, así que cuál de ellos recibe el cupo es una moneda
    al aire que **ningún modelo puede predecir**. La exactitud cruda castiga esa
    moneda al aire y por eso subestima la calidad del plan.

    Esta métrica compara lo único que el maestro sí determinó: cuántos vehículos
    de cada clase van a cada camión. Se reporta como `1 - distancia de variación
    total`, así que 1.0 significa que el plan es indistinguible del óptimo a
    nivel de decisión real.
    """
    agreements, exact = [], 0
    for r in results:
        n_slots = r.n_trucks + 1
        pred = np.zeros((n_slots, r.n_classes), dtype=int)
        true = np.zeros((n_slots, r.n_classes), dtype=int)
        np.add.at(pred, (r.predicted_index, r.class_index), 1)
        np.add.at(true, (r.target_index, r.class_index), 1)

        n = r.n_vehicles
        tv = float(np.abs(pred - true).sum()) / (2.0 * n) if n else 0.0
        agreements.append(1.0 - tv)
        exact += int(tv == 0.0)

    a = np.asarray(agreements)
    return {
        "class_level_agreement_mean": float(a.mean()),
        "episodes_identical_to_teacher_pct": float(100.0 * exact / len(results)),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def confusion(results: list[EpisodeResult], n_labels: int) -> list[list[int]]:
    """Matriz de confusión sobre índices canónicos (fila = maestro)."""
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    for r in results:
        for t, p in zip(r.target_index, r.predicted_index, strict=True):
            if t < n_labels and p < n_labels:
                matrix[t, p] += 1
    return matrix.tolist()


def aggregate(results: list[EpisodeResult], n_labels: int) -> dict:
    """Resumen listo para `metrics.json` y para el reporte."""
    if not results:
        raise ValueError("No hay resultados que agregar.")

    y_true = np.concatenate([r.target_index for r in results])
    y_pred = np.concatenate([r.predicted_index for r in results])
    accuracy, macro_f1 = _macro_f1(y_true, y_pred)

    loaded_gap = np.array([r.loaded_gap for r in results], dtype=float)
    cu_gap = np.array([r.cu_gap for r in results], dtype=float)
    teacher_loaded = np.array([r.teacher_n_loaded for r in results], dtype=float)
    teacher_cu = np.array([r.teacher_cu for r in results], dtype=float)
    model_cu = np.array([r.model_cu for r in results], dtype=float)
    capacity = np.array([r.total_capacity for r in results], dtype=float)
    overflow = np.array([r.max_overflow for r in results], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_gap = np.where(teacher_loaded > 0, loaded_gap / teacher_loaded, 0.0)

    return {
        # 1. Lo primero que hay que mirar.
        "capacity_violation_rate": float((overflow > _TOL).mean()),
        "max_overflow_cu": float(overflow.max()),
        # 2. Objetivo primario del maestro.
        "loaded_gap_mean": float(loaded_gap.mean()),
        "loaded_gap_max": int(loaded_gap.max()),
        "episodes_matching_teacher_count_pct": float(100.0 * (loaded_gap == 0).mean()),
        "optimality_gap_loaded_pct": float(100.0 * rel_gap.mean()),
        # 3. Objetivo secundario.
        "cu_gap_mean": float(cu_gap.mean()),
        "cu_utilization_model_pct": float(100.0 * model_cu.sum() / capacity.sum()),
        "cu_utilization_teacher_pct": float(100.0 * teacher_cu.sum() / capacity.sum()),
        # 4. Diferidos.
        "deferred_model_total": int(sum(int((r.predicted_index == 0).sum()) for r in results)),
        "deferred_teacher_total": int(sum(int((r.target_index == 0).sum()) for r in results)),
        # 5. Métricas de clasificación (secundarias).
        "macro_f1": macro_f1,
        "raw_assignment_accuracy": accuracy,
        **class_level_agreement(results),
        "confusion_matrix": confusion(results, n_labels),
        # Contexto.
        "n_episodes": len(results),
        "n_vehicle_rows": int(len(y_true)),
    }
