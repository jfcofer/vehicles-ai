"""Tensores por par: esquema, enmascarado del relleno y ausencia de fugas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.features import (
    CONTEXT_BLOCK,
    MASK_NEG_INF,
    TRUCK_BLOCK,
    VEHICLE_BLOCK,
    BlockScaler,
    balanced_sample_weights,
    build_all_episodes,
    build_episode_tensors,
    build_model_arrays,
    feature_names,
)

CLASSES = ["AUTOMOVIL", "CAMIONETA", "JEEP", "MOTOCICLETA"]


def _rows(episode_id: str, specs: list[tuple[str, str, float, str]], caps: list[float]):
    """specs = [(uid, clase, cu, truck), ...]"""
    return pd.DataFrame(
        [
            {
                "episode_id": episode_id,
                "uid": uid,
                "clase": clase,
                "cu": cu,
                "canton": 10701,
                "truck": truck,
                "loaded": truck != "SIN_CAMION",
                "iso_year": 2024,
                "n_trucks": len(caps),
                "truck_capacities": caps,
                "n_loaded": sum(1 for s in specs if s[3] != "SIN_CAMION"),
                "cu_utilized": sum(s[2] for s in specs if s[3] != "SIN_CAMION"),
            }
            for uid, clase, cu, truck in specs
        ]
    )


def test_dimensiones_del_esquema():
    names = feature_names(CLASSES)
    assert len(names[VEHICLE_BLOCK]) == 6
    assert len(names[TRUCK_BLOCK]) == 3
    assert len(names[CONTEXT_BLOCK]) == 10


def test_las_features_del_vehiculo_no_contienen_identificadores():
    names = feature_names(CLASSES)
    plano = " ".join(sum(names.values(), []))
    for prohibido in ("uid", "codigo", "truck_id", "canton", "posicion"):
        assert prohibido not in plano


def test_el_objetivo_usa_el_orden_canonico():
    # CAMION_2 original (6.9) es el mayor -> índice canónico 1.
    rows = _rows("E1", [("a", "AUTOMOVIL", 1.0, "CAMION_2")], caps=[3.2, 6.9])
    ep = build_episode_tensors(rows, CLASSES)

    assert ep.capacities.tolist() == [6.9, 3.2]
    assert ep.target.tolist() == [1]


def test_sin_camion_es_el_indice_cero():
    rows = _rows("E1", [("a", "AUTOMOVIL", 1.0, "SIN_CAMION")], caps=[6.0])
    assert build_episode_tensors(rows, CLASSES).target.tolist() == [0]


def test_los_vehiculos_se_ordenan_por_uid_de_forma_determinista():
    specs = [("c", "JEEP", 1.1, "CAMION_1"), ("a", "AUTOMOVIL", 1.0, "CAMION_1")]
    ep = build_episode_tensors(_rows("E1", specs, [6.0]), CLASSES)
    otro = build_episode_tensors(_rows("E1", specs[::-1], [6.0]), CLASSES)
    assert np.array_equal(ep.vehicle, otro.vehicle)


def test_el_contexto_agrega_el_manifiesto_completo():
    specs = [
        ("a", "AUTOMOVIL", 1.0, "CAMION_1"),
        ("b", "MOTOCICLETA", 0.2, "CAMION_1"),
        ("c", "MOTOCICLETA", 0.2, "SIN_CAMION"),
    ]
    ep = build_episode_tensors(_rows("E1", specs, [6.0, 4.0]), CLASSES)
    n_vehiculos, n_camiones, cu_total, cap_total, deficit, ratio = ep.context[:6]

    assert n_vehiculos == 3
    assert n_camiones == 2
    assert cu_total == pytest.approx(1.4)
    assert cap_total == pytest.approx(10.0)
    assert deficit == pytest.approx(-8.6)
    assert ratio == pytest.approx(0.14)
    assert ep.context[6:].tolist() == [1.0, 0.0, 0.0, 2.0]  # conteos por clase


def test_n_misma_clase_cuenta_los_hermanos_de_clase():
    specs = [
        ("a", "MOTOCICLETA", 0.2, "CAMION_1"),
        ("b", "MOTOCICLETA", 0.2, "CAMION_1"),
        ("c", "AUTOMOVIL", 1.0, "CAMION_1"),
    ]
    ep = build_episode_tensors(_rows("E1", specs, [6.0]), CLASSES)
    # Ordenados por uid: a=MOTOCICLETA, b=MOTOCICLETA, c=AUTOMOVIL.
    assert ep.vehicle[:, -1].tolist() == [2.0, 2.0, 1.0]


def test_el_relleno_queda_enmascarado_con_menos_infinito():
    episodes = [
        build_episode_tensors(_rows("E1", [("a", "AUTOMOVIL", 1.0, "CAMION_1")], [6.0]), CLASSES),
        build_episode_tensors(
            _rows("E2", [("b", "AUTOMOVIL", 1.0, "CAMION_1")], [6.0, 5.0, 4.0]), CLASSES
        ),
    ]
    scaler = BlockScaler.fit(episodes, CLASSES)
    arrays = build_model_arrays(episodes, scaler)

    assert arrays.max_trucks == 3
    assert arrays.mask_bias[0].tolist() == [0.0, MASK_NEG_INF, MASK_NEG_INF]
    assert arrays.mask_bias[1].tolist() == [0.0, 0.0, 0.0]


def test_las_dimensiones_de_los_tensores_cuadran():
    episodes = build_all_episodes(
        _rows(
            "E1",
            [("a", "AUTOMOVIL", 1.0, "CAMION_1"), ("b", "JEEP", 1.1, "SIN_CAMION")],
            [6.0, 4.0],
        ),
        CLASSES,
    )
    scaler = BlockScaler.fit(episodes, CLASSES)
    arrays = build_model_arrays(episodes, scaler)

    assert arrays.pair.shape == (2, 2, 19)
    assert arrays.defer.shape == (2, 16)
    assert arrays.mask_bias.shape == (2, 2)
    assert arrays.target.shape == (2,)


def test_el_escalador_deja_intactos_los_one_hot():
    episodes = build_all_episodes(
        _rows(
            "E1",
            [("a", "AUTOMOVIL", 1.0, "CAMION_1"), ("b", "MOTOCICLETA", 0.2, "CAMION_1")],
            [6.0],
        ),
        CLASSES,
    )
    scaler = BlockScaler.fit(episodes, CLASSES)
    escalado = scaler.transform(VEHICLE_BLOCK, episodes[0].vehicle)

    # Columnas 1..4 son one-hot y deben seguir siendo 0 o 1.
    assert set(np.unique(escalado[:, 1:5])) <= {0.0, 1.0}


def test_el_escalador_sobrevive_a_una_columna_constante():
    """Una columna constante en entrenamiento tiene desviación 0; dividir por
    ella produciría NaN silenciosos que sólo aparecerían como pérdida NaN."""
    episodes = build_all_episodes(
        _rows("E1", [("a", "AUTOMOVIL", 1.0, "CAMION_1")], [6.0]), CLASSES
    )
    scaler = BlockScaler.fit(episodes, CLASSES)
    arrays = build_model_arrays(episodes, scaler)
    assert np.isfinite(arrays.pair).all()
    assert np.isfinite(arrays.defer).all()


def test_el_escalador_va_y_vuelve_por_json():
    episodes = build_all_episodes(
        _rows(
            "E1",
            [("a", "AUTOMOVIL", 1.0, "CAMION_1"), ("b", "JEEP", 1.1, "CAMION_1")],
            [6.0, 3.0],
        ),
        CLASSES,
    )
    scaler = BlockScaler.fit(episodes, CLASSES)
    recargado = BlockScaler.from_dict(scaler.to_dict())

    for block in (VEHICLE_BLOCK, TRUCK_BLOCK, CONTEXT_BLOCK):
        assert np.allclose(scaler.mean[block], recargado.mean[block])
        assert np.allclose(scaler.std[block], recargado.std[block])
        assert np.array_equal(scaler.mask[block], recargado.mask[block])


def test_los_pesos_compensan_el_desbalance_medido():
    target = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 1 diferido de 10
    weights = balanced_sample_weights(target)
    assert weights[0] == pytest.approx(5.0)
    assert weights[1] == pytest.approx(10 / 18)
    # El peso total de cada grupo queda igualado.
    assert weights[target == 0].sum() == pytest.approx(weights[target == 1].sum())


def test_sin_diferidos_los_pesos_son_neutros():
    weights = balanced_sample_weights(np.array([1, 1, 2]))
    assert weights.tolist() == [1.0, 1.0, 1.0]


def test_falla_si_la_flota_excede_el_relleno():
    episodes = build_all_episodes(
        _rows("E1", [("a", "AUTOMOVIL", 1.0, "CAMION_1")], [6.0, 5.0]), CLASSES
    )
    scaler = BlockScaler.fit(episodes, CLASSES)
    with pytest.raises(ValueError, match="camiones y el relleno"):
        build_model_arrays(episodes, scaler, max_trucks=1)
