"""src/modeling/mlp_classifier.py

Perceptrón multicapa compartido por par (vehículo, camión), en Keras 3.

Idea central
------------
El modelo no tiene una salida llamada `CAMION_1` y otra llamada `CAMION_2`. Tiene
**un solo** MLP que responde a una pregunta por par::

    ¿qué tan compatible es este vehículo con este camión,
     dado el manifiesto completo?

Como `Dense` opera sobre el último eje, el mismo MLP se aplica a `(v_i, t_1)`,
`(v_i, t_2)`, …, `(v_i, t_n)` con los mismos pesos. El número de parámetros no
depende de la cantidad de camiones, y el eje de camiones se declara `None`: el
modelo guardado acepta literalmente cualquier `T` en inferencia. Ésa es la
diferencia estructural con un `Dense(5)` de slots fijos, que quedaría atado a
cuatro camiones para siempre.

Una segunda cabeza, alimentada sólo con vehículo y contexto, produce el logit de
`SIN_CAMION`: diferir no es "asignar a un camión de capacidad cero", es una
decisión distinta que no depende de ningún camión en particular.

Salida: `(B, 1 + T)`, con **`SIN_CAMION` en el índice 0** y los camiones canónicos
en `1..T`. Poner el diferimiento primero hace que el índice objetivo no dependa
del relleno del lote.

API funcional, no subclase
--------------------------
Un `keras.Model` subclasificado sin `get_config()` no se puede reconstruir desde
un `.keras`: `ModelCheckpoint` guarda y `load_model` falla al recargar. Con la API
funcional el grafo se serializa solo, sin `custom_objects`, y además
`model.summary()` sale presentable para el reporte. El enmascarado del relleno se
hace con un `Add` sobre un tensor de sesgo (`0` o `-1e9`) que entra como entrada,
en lugar de una `Lambda`, que volvería a romper la serialización.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
from keras import layers

PAIR_INPUT = "pair_features"
DEFER_INPUT = "defer_features"
MASK_INPUT = "mask_bias"


@dataclass(frozen=True)
class MLPConfig:
    """Hiper-parámetros de arquitectura y optimización.

    Los valores por defecto son el punto de partida documentado en la sección
    VI-D; la configuración efectiva se carga desde `config/mlp.yaml`.
    """

    pair_units: tuple[int, ...] = (64, 32)
    defer_units: tuple[int, ...] = (32, 16)
    dropout: float = 0.20
    l2: float = 1e-4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    clipnorm: float = 1.0
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    seed: int = 20260725

    @classmethod
    def from_dict(cls, payload: dict) -> MLPConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Claves desconocidas en la configuración: {sorted(unknown)}")
        data = dict(payload)
        for key in ("pair_units", "defer_units"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def as_dict(self) -> dict:
        return {
            f: (list(v) if isinstance(v, tuple) else v)
            for f, v in ((f, getattr(self, f)) for f in self.__dataclass_fields__)
        }


def build_pairwise_mlp(pair_dim: int, defer_dim: int, config: MLPConfig) -> keras.Model:
    """Construye el modelo. `T` queda dinámico en todas las entradas."""
    regularizer = keras.regularizers.L2(config.l2)

    pair_in = keras.Input(shape=(None, pair_dim), name=PAIR_INPUT)
    defer_in = keras.Input(shape=(defer_dim,), name=DEFER_INPUT)
    mask_in = keras.Input(shape=(None,), name=MASK_INPUT)

    # --- Cabeza de pares: los mismos pesos para todos los camiones. -----------
    x = pair_in
    for i, units in enumerate(config.pair_units):
        x = layers.Dense(
            units, activation="relu", kernel_regularizer=regularizer, name=f"pair_dense_{i}"
        )(x)
        x = layers.Dropout(config.dropout, name=f"pair_dropout_{i}")(x)
    x = layers.Dense(1, name="pair_logit")(x)  # (B, T, 1)
    pair_logits = layers.Reshape((-1,), name="pair_logits")(x)  # (B, T)

    # Los slots de relleno reciben -1e9 y desaparecen del softmax.
    pair_logits = layers.Add(name="masked_pair_logits")([pair_logits, mask_in])

    # --- Cabeza de diferimiento: no depende de ningún camión. -----------------
    d = defer_in
    for i, units in enumerate(config.defer_units):
        d = layers.Dense(
            units, activation="relu", kernel_regularizer=regularizer, name=f"defer_dense_{i}"
        )(d)
        d = layers.Dropout(config.dropout, name=f"defer_dropout_{i}")(d)
    defer_logit = layers.Dense(1, name="defer_logit")(d)  # (B, 1)

    # Índice 0 = SIN_CAMION, índices 1..T = camiones en orden canónico.
    logits = layers.Concatenate(axis=1, name="assignment_logits")([defer_logit, pair_logits])

    return keras.Model(
        inputs={PAIR_INPUT: pair_in, DEFER_INPUT: defer_in, MASK_INPUT: mask_in},
        outputs=logits,
        name="pairwise_assignment_mlp",
    )


def compile_model(model: keras.Model, config: MLPConfig) -> keras.Model:
    """AdamW + entropía cruzada categórica dispersa sobre logits.

    `from_logits=True` porque la última capa no lleva softmax: Keras aplica
    internamente la versión numéricamente estable de softmax + entropía cruzada.
    """
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            clipnorm=config.clipnorm,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="raw_assignment_accuracy")],
    )
    return model


def build_callbacks(config: MLPConfig, checkpoint_path: str, history_path: str) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0
        ),
        keras.callbacks.CSVLogger(history_path),
        keras.callbacks.TerminateOnNaN(),
    ]


def model_summary_text(model: keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append, line_length=100)
    return "\n".join(lines) + "\n"
