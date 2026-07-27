"""Decoder de capacidad. La invariante que importa: nunca excede la capacidad."""

from __future__ import annotations

import numpy as np
import pytest

from src.modeling.capacity_decoder import (
    DEFERRED,
    POLICIES,
    decode_episode,
    greedy_first_fit_decreasing,
    split_logits,
)


def _logits(preferences: list[list[float]], defer: list[float]) -> np.ndarray:
    """Arma (V, 1+T) con el diferimiento en la columna 0."""
    return np.column_stack([np.array(defer, dtype=float), np.array(preferences, dtype=float)])


@pytest.mark.parametrize("policy", POLICIES)
def test_nunca_excede_la_capacidad_aunque_el_modelo_insista(policy):
    """Tres vehículos de 3.0 CU, un camión de 6.0, y el modelo quiere meterlos
    todos ahí. Sólo pueden entrar dos."""
    logits = _logits([[9.0], [9.0], [9.0]], defer=[-9.0, -9.0, -9.0])
    decoded = decode_episode(logits, cu=[3.0, 3.0, 3.0], capacities=[6.0], policy=policy)

    assert decoded.is_feasible
    assert decoded.max_overflow == 0.0
    assert decoded.n_loaded == 2
    assert decoded.n_deferred == 1


@pytest.mark.parametrize("policy", POLICIES)
def test_flota_vacia_difiere_todo(policy):
    logits = np.array([[1.0], [1.0]])
    decoded = decode_episode(logits, cu=[1.0, 1.0], capacities=[], policy=policy)
    assert decoded.n_loaded == 0
    assert decoded.assignment.tolist() == [DEFERRED, DEFERRED]


@pytest.mark.parametrize("policy", POLICIES)
def test_manifiesto_vacio(policy):
    decoded = decode_episode(np.zeros((0, 2)), cu=[], capacities=[6.0], policy=policy)
    assert decoded.n_loaded == 0
    assert decoded.is_feasible


def test_un_solo_camion_con_capacidad_de_sobra():
    logits = _logits([[5.0], [5.0]], defer=[0.0, 0.0])
    decoded = decode_episode(logits, cu=[1.0, 1.4], capacities=[6.0])
    assert decoded.n_loaded == 2
    assert decoded.cu_loaded == pytest.approx(2.4)


def test_carga_exacta_hasta_el_ultimo_cu():
    """2/3 CU x 9 = 6.0 exacto. El decoder no puede fallar por un epsilon."""
    cu = [2 / 3] * 9
    logits = _logits([[1.0]] * 9, defer=[0.0] * 9)
    decoded = decode_episode(logits, cu=cu, capacities=[6.0])

    assert decoded.n_loaded == 9
    assert decoded.is_feasible
    assert decoded.cu_loaded == pytest.approx(6.0)


def test_capacidad_insuficiente_para_cualquier_vehiculo():
    logits = _logits([[5.0], [5.0]], defer=[0.0, 0.0])
    decoded = decode_episode(logits, cu=[4.0, 4.0], capacities=[3.0])
    assert decoded.n_loaded == 0
    assert decoded.n_deferred == 2


def test_cuatro_camiones_respeta_la_preferencia_del_modelo():
    # El modelo prefiere el camión 2 (índice 2) para el único vehículo.
    logits = _logits([[0.1, 0.2, 9.0, 0.3]], defer=[-5.0])
    decoded = decode_episode(logits, cu=[1.0], capacities=[6.0, 6.0, 6.0, 6.0])
    assert decoded.assignment.tolist() == [2]


def test_cae_al_siguiente_camion_cuando_el_preferido_esta_lleno():
    logits = _logits([[9.0, 1.0], [9.0, 1.0]], defer=[-5.0, -5.0])
    decoded = decode_episode(logits, cu=[4.0, 4.0], capacities=[6.0, 6.0])
    assert sorted(decoded.assignment.tolist()) == [0, 1]


def test_capacidades_iguales_no_rompen_la_factibilidad():
    logits = _logits([[1.0, 1.0]] * 6, defer=[0.0] * 6)
    decoded = decode_episode(logits, cu=[2.0] * 6, capacities=[6.0, 6.0])
    assert decoded.is_feasible
    assert decoded.n_loaded == 6


def test_la_politica_de_conteo_carga_al_menos_tanto_como_la_del_modelo():
    """Con capacidad escasa, atender primero a los vehículos pequeños maximiza
    el conteo, que es el objetivo primario del maestro."""
    cu = [5.0, 1.0, 1.0, 1.0]
    # El modelo prefiere fuertemente cargar el grande primero.
    logits = _logits([[9.0], [1.0], [1.0], [1.0]], defer=[-1.0, -1.0, -1.0, -1.0])

    por_modelo = decode_episode(logits, cu, capacities=[6.0], policy="model")
    por_conteo = decode_episode(logits, cu, capacities=[6.0], policy="count")

    assert por_modelo.n_loaded == 2  # 5.0 + 1.0
    assert por_conteo.n_loaded == 3  # 1.0 + 1.0 + 1.0
    assert por_conteo.n_loaded >= por_modelo.n_loaded


def test_respect_defer_honra_el_sin_camion_predicho():
    # El vehículo 1 tiene margen negativo: el modelo prefiere diferirlo.
    logits = _logits([[1.0], [-5.0]], defer=[0.0, 5.0])
    respetando = decode_episode(logits, cu=[1.0, 1.0], capacities=[6.0], policy="respect_defer")
    ignorando = decode_episode(logits, cu=[1.0, 1.0], capacities=[6.0], policy="count")

    assert respetando.n_loaded == 1
    assert ignorando.n_loaded == 2  # cabía perfectamente


def test_la_utilizacion_se_calcula_sobre_la_capacidad_total():
    logits = _logits([[5.0, 1.0]], defer=[0.0])
    decoded = decode_episode(logits, cu=[3.0], capacities=[6.0, 4.0])
    assert decoded.utilization == pytest.approx(3.0 / 10.0)


def test_split_logits_separa_diferir_de_los_camiones():
    logits = np.array([[0.5, 1.0, 2.0, 3.0]])
    pares, diferir = split_logits(logits, n_trucks=3)
    assert diferir.tolist() == [0.5]
    assert pares.tolist() == [[1.0, 2.0, 3.0]]


def test_split_logits_falla_si_faltan_columnas():
    with pytest.raises(ValueError, match="al menos"):
        split_logits(np.zeros((2, 3)), n_trucks=4)


def test_el_greedy_de_referencia_tambien_es_factible():
    decoded = greedy_first_fit_decreasing(cu=[4.0, 3.0, 2.0, 1.0], capacities=[6.0, 6.0])
    assert decoded.is_feasible
    assert decoded.n_loaded == 4


def test_el_greedy_puede_ser_peor_que_el_optimo():
    """El caso del reporte: llenar con el más grande primero desperdicia espacio."""
    decoded = greedy_first_fit_decreasing(cu=[4.0, 4.0, 3.0, 3.0], capacities=[6.0, 6.0])
    assert decoded.n_loaded == 2  # 4+4 en camiones distintos, los 3.0 no caben
    assert decoded.is_feasible


def test_politica_desconocida_falla():
    with pytest.raises(ValueError, match="Política desconocida"):
        decode_episode(np.zeros((1, 2)), cu=[1.0], capacities=[6.0], policy="magia")
