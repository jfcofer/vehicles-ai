"""Cotas exactas sobre la exactitud cruda alcanzable.

Cada caso se verifica contra aritmética escrita a mano, no contra la propia
implementación: si la fórmula estuviera mal derivada, estos números no cuadrarían.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.modeling.metrics import label_ceilings


def _ceilings(rows: list[tuple[int, int, int]]) -> dict:
    """`rows` = [(episodio, clase, destino), ...]; destino 0 = SIN_CAMION."""
    arr = np.array(rows, dtype=np.int64)
    return label_ceilings(target_index=arr[:, 2], class_index=arr[:, 1], episode_index=arr[:, 0])


def test_una_clase_entera_al_mismo_camion_no_tiene_ambiguedad():
    """Si nadie de la clase se separa, no hay nada que adivinar: el techo es 1."""
    out = _ceilings([(0, 0, 1), (0, 0, 1), (0, 0, 1)])
    assert out["ceiling_argmax_micro"] == pytest.approx(1.0)
    assert out["ceiling_decoder_micro"] == pytest.approx(1.0)


def test_clase_partida_por_la_mitad_es_una_moneda_al_aire():
    """2 motos idénticas, una a cada camión: ninguna entrada las distingue.

    Cota A: el mejor destino único acierta 1 de 2          -> 0,5
    Cota B: (1² + 1²) / 2 = 1 acierto esperado de 2        -> 0,5
    """
    out = _ceilings([(0, 0, 1), (0, 0, 2)])
    assert out["ceiling_argmax_micro"] == pytest.approx(0.5)
    assert out["ceiling_decoder_micro"] == pytest.approx(0.5)


def test_reparto_desigual_separa_las_dos_cotas():
    """3 vehículos de una clase repartidos 2/1 entre CAMION_1 y diferir.

    Cota A: max(2, 1) = 2 aciertos de 3                    -> 2/3
    Cota B: (2² + 1²) / 3 = 5/3 aciertos esperados de 3    -> 5/9

    Es el caso que muestra por qué son dos cotas y no una: apostar siempre a la
    moda gana más coincidencias que repartir, aunque reparta bien los conteos.
    """
    out = _ceilings([(0, 0, 1), (0, 0, 1), (0, 0, 0)])
    assert out["ceiling_argmax_micro"] == pytest.approx(2 / 3)
    assert out["ceiling_decoder_micro"] == pytest.approx(5 / 9)
    assert out["ceiling_decoder_micro"] < out["ceiling_argmax_micro"]


def test_clases_distintas_no_se_mezclan():
    """El one-hot de clase sí distingue: cada clase aporta su propio máximo.

    2 automóviles a CAMION_1 (sin ambigüedad) + 2 motos repartidas 1/1.
    Cota A = (2 + 1) / 4 = 0,75.
    """
    out = _ceilings([(0, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 2)])
    assert out["ceiling_argmax_micro"] == pytest.approx(0.75)
    assert out["ceiling_decoder_micro"] == pytest.approx((2 + 1) / 4)


def test_las_cotas_no_cambian_al_renombrar_los_camiones():
    """Invariancia a la canonicalización.

    Es la demostración formal de que canonicalizar no puede subir el techo de
    exactitud: permutar el eje de destinos no altera ni `max_t` ni `Σ_t n²`.
    """
    original = _ceilings([(0, 0, 1), (0, 0, 1), (0, 1, 2), (0, 1, 3)])
    renombrado = _ceilings([(0, 0, 3), (0, 0, 3), (0, 1, 1), (0, 1, 2)])
    assert original["ceiling_argmax_micro"] == pytest.approx(renombrado["ceiling_argmax_micro"])
    assert original["ceiling_decoder_micro"] == pytest.approx(renombrado["ceiling_decoder_micro"])


def test_micro_pondera_por_vehiculo_y_macro_por_episodio():
    """Un episodio perfecto de 1 fila y uno ambiguo de 4 no pesan igual.

    Episodio 0: 1 vehículo, sin ambigüedad          -> 1,0
    Episodio 1: 4 vehículos de una clase, 2/2       -> 0,5
    micro = (1 + 2) / 5 = 0,6 ; macro = (1,0 + 0,5) / 2 = 0,75
    """
    out = _ceilings(
        [(0, 0, 1), (1, 0, 1), (1, 0, 1), (1, 0, 2), (1, 0, 2)],
    )
    assert out["ceiling_argmax_micro"] == pytest.approx(0.6)
    assert out["ceiling_argmax_macro"] == pytest.approx(0.75)
    assert out["n_episodes"] == 2
    assert out["n_vehicle_rows"] == 5


def test_la_cota_del_decodificador_nunca_supera_la_del_argmax():
    """Propiedad general, sobre repartos aleatorios."""
    rng = np.random.default_rng(20260727)
    for _ in range(50):
        n = int(rng.integers(5, 40))
        rows = [
            (int(rng.integers(0, 4)), int(rng.integers(0, 4)), int(rng.integers(0, 5)))
            for _ in range(n)
        ]
        out = _ceilings(rows)
        assert out["ceiling_decoder_micro"] <= out["ceiling_argmax_micro"] + 1e-12
        assert out["ceiling_argmax_micro"] <= 1.0 + 1e-12


def test_rechaza_vectores_de_distinta_longitud():
    with pytest.raises(ValueError, match="misma longitud"):
        label_ceilings(np.array([0, 1]), np.array([0]), np.array([0, 0]))
