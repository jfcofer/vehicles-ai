"""El MLP compartido: serialización, enmascarado y número variable de camiones."""

from __future__ import annotations

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from src.modeling.mlp_classifier import (  # noqa: E402
    DEFER_INPUT,
    MASK_INPUT,
    PAIR_INPUT,
    MLPConfig,
    build_pairwise_mlp,
    compile_model,
    model_summary_text,
)

PAIR_DIM, DEFER_DIM = 19, 16
CONFIG = MLPConfig(pair_units=(8, 4), defer_units=(8, 4), dropout=0.0, epochs=2)


def _batch(n: int, t: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        PAIR_INPUT: rng.normal(size=(n, t, PAIR_DIM)).astype(np.float32),
        DEFER_INPUT: rng.normal(size=(n, DEFER_DIM)).astype(np.float32),
        MASK_INPUT: np.zeros((n, t), dtype=np.float32),
    }


def _model():
    return compile_model(build_pairwise_mlp(PAIR_DIM, DEFER_DIM, CONFIG), CONFIG)


def test_la_salida_tiene_un_logit_por_camion_mas_el_diferimiento():
    out = _model().predict(_batch(5, 4), verbose=0)
    assert out.shape == (5, 5)  # 1 + 4


def test_el_numero_de_parametros_no_depende_de_la_cantidad_de_camiones():
    """La propiedad que distingue esta arquitectura de un Dense(5) de slots fijos."""
    assert build_pairwise_mlp(PAIR_DIM, DEFER_DIM, CONFIG).count_params() == (
        build_pairwise_mlp(PAIR_DIM, DEFER_DIM, CONFIG).count_params()
    )
    model = _model()
    antes = model.count_params()
    model.predict(_batch(3, 2), verbose=0)
    model.predict(_batch(3, 12), verbose=0)
    assert model.count_params() == antes


def test_los_mismos_pesos_atienden_2_y_50_camiones():
    """Prueba directa del requisito 'sin límite codificado' de la planificación."""
    model = _model()
    for t in (1, 2, 4, 10, 50):
        out = model.predict(_batch(3, t), verbose=0)
        assert out.shape == (3, t + 1)


def test_el_relleno_enmascarado_no_recibe_probabilidad():
    model = _model()
    batch = _batch(4, 6)
    batch[MASK_INPUT][:, 3:] = -1e9  # sólo 3 camiones reales

    logits = model.predict(batch, verbose=0)
    probs = keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))

    # Columnas 4,5,6 corresponden a los camiones rellenados (offset por diferir).
    assert probs[:, 4:].max() < 1e-12
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_el_scorer_es_equivariante_a_la_permutacion_de_camiones():
    """Al reordenar los camiones, las puntuaciones se reordenan igual: el modelo
    juzga por capacidad, no por posición."""
    model = _model()
    batch = _batch(4, 3, seed=7)

    base = model.predict(batch, verbose=0)
    permutado = dict(batch)
    permutado[PAIR_INPUT] = batch[PAIR_INPUT][:, [2, 0, 1], :]
    salida = model.predict(permutado, verbose=0)

    assert np.allclose(salida[:, 0], base[:, 0], atol=1e-5)  # diferir no cambia
    assert np.allclose(salida[:, 1:], base[:, [3, 1, 2]], atol=1e-5)


def test_entrena_dos_epocas_sin_nan():
    model = _model()
    batch = _batch(64, 4)
    y = np.random.default_rng(1).integers(0, 5, size=64)

    history = model.fit(batch, y, epochs=2, batch_size=16, verbose=0)
    assert np.isfinite(history.history["loss"]).all()


def test_guardar_y_recargar_produce_logits_identicos(tmp_path):
    """El `keras.Model` subclasificado sin `get_config()` falla exactamente aquí:
    `ModelCheckpoint` guarda y `load_model` no puede reconstruirlo."""
    model = _model()
    batch = _batch(8, 3, seed=3)
    esperado = model.predict(batch, verbose=0)

    path = tmp_path / "model.keras"
    model.save(path)
    recargado = keras.models.load_model(path)

    assert np.allclose(recargado.predict(batch, verbose=0), esperado, atol=1e-6)


def test_el_modelo_recargado_sigue_aceptando_otro_numero_de_camiones(tmp_path):
    model = _model()
    path = tmp_path / "model.keras"
    model.save(path)

    recargado = keras.models.load_model(path)
    assert recargado.predict(_batch(2, 9), verbose=0).shape == (2, 10)


def test_el_resumen_del_modelo_es_texto_para_el_reporte():
    texto = model_summary_text(_model())
    assert "pairwise_assignment_mlp" in texto
    assert "pair_logit" in texto
    assert "defer_logit" in texto


def test_la_configuracion_rechaza_claves_desconocidas():
    with pytest.raises(ValueError, match="desconocidas"):
        MLPConfig.from_dict({"pair_units": [64], "aprendizaje": 0.1})


def test_la_configuracion_va_y_vuelve_por_diccionario():
    config = MLPConfig(pair_units=(128, 64), dropout=0.3)
    assert MLPConfig.from_dict(config.as_dict()) == config
