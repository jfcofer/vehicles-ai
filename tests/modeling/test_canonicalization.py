"""Canonicalización de la flota: capacidad descendente y remapeo de etiquetas."""

from __future__ import annotations

import pytest

from src.modeling.canonicalization import (
    DEFER_LABEL,
    canonical_target_index,
    canonicalize_fleet,
    parse_truck_label,
    truck_label,
)


def test_capacidades_quedan_en_orden_descendente():
    fleet = canonicalize_fleet([3.2, 6.9, 7.4, 3.3])
    assert fleet.capacities == (7.4, 6.9, 3.3, 3.2)


def test_etiquetas_se_remapean_a_la_nueva_posicion():
    # Original: CAMION_1=3.2, CAMION_2=6.9, CAMION_3=7.4, CAMION_4=3.3
    fleet = canonicalize_fleet([3.2, 6.9, 7.4, 3.3])
    assert fleet.label_map["CAMION_3"] == "CAMION_1"  # 7.4, el mayor
    assert fleet.label_map["CAMION_2"] == "CAMION_2"  # 6.9
    assert fleet.label_map["CAMION_4"] == "CAMION_3"  # 3.3
    assert fleet.label_map["CAMION_1"] == "CAMION_4"  # 3.2, el menor
    assert fleet.label_map[DEFER_LABEL] == DEFER_LABEL


def test_el_diferimiento_es_siempre_el_indice_cero():
    for caps in ([5.0], [5.0, 5.0], [1.0, 9.0, 3.0, 7.0]):
        fleet = canonicalize_fleet(caps)
        assert canonical_target_index(DEFER_LABEL, fleet) == 0


def test_el_camion_mayor_es_siempre_el_indice_uno():
    fleet = canonicalize_fleet([3.2, 6.9, 7.4, 3.3])
    assert canonical_target_index("CAMION_3", fleet) == 1


def test_es_idempotente():
    once = canonicalize_fleet([3.2, 6.9, 7.4, 3.3])
    twice = canonicalize_fleet(once.capacities)
    assert twice.capacities == once.capacities
    assert twice.order == (0, 1, 2, 3)


def test_es_invariante_a_la_permutacion_de_entrada():
    """El resultado operativo no puede depender del orden en que el generador
    aleatorio escupió las capacidades -- que es exactamente el defecto que este
    módulo corrige."""
    a = canonicalize_fleet([3.2, 6.9, 7.4, 3.3])
    b = canonicalize_fleet([7.4, 3.3, 3.2, 6.9])
    assert a.capacities == b.capacities


def test_capacidades_iguales_desempatan_por_indice_original():
    fleet = canonicalize_fleet([5.0, 5.0, 5.0])
    assert fleet.order == (0, 1, 2)
    assert fleet.capacities == (5.0, 5.0, 5.0)


def test_flota_de_un_solo_camion():
    fleet = canonicalize_fleet([7.4])
    assert fleet.n_trucks == 1
    assert canonical_target_index("CAMION_1", fleet) == 1
    assert canonical_target_index(DEFER_LABEL, fleet) == 0


def test_flota_grande_no_tiene_limite_codificado():
    caps = [float(i) for i in range(1, 21)]
    fleet = canonicalize_fleet(caps)
    assert fleet.n_trucks == 20
    assert fleet.capacities[0] == 20.0
    assert canonical_target_index("CAMION_20", fleet) == 1


def test_etiqueta_fuera_de_rango_falla():
    fleet = canonicalize_fleet([5.0, 6.0])
    with pytest.raises(KeyError):
        canonical_target_index("CAMION_9", fleet)


def test_ida_y_vuelta_de_etiquetas():
    assert parse_truck_label(truck_label(0)) == 0
    assert parse_truck_label(truck_label(41)) == 41
    assert parse_truck_label(DEFER_LABEL) is None


def test_el_mapa_inverso_deshace_el_remapeo():
    fleet = canonicalize_fleet([3.2, 6.9, 7.4, 3.3])
    inverse = fleet.inverse_label_map
    for old, new in fleet.label_map.items():
        assert inverse[new] == old
