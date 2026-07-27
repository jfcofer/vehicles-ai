#!/usr/bin/env python3
"""Entrena el clasificador MLP compartido por par (vehículo, camión).

    data/episodes/episodes.parquet + episode_vehicles.parquet
            -> join, descarte de episodios no-óptimos, partición temporal
            -> canonicalización de la flota por capacidad
            -> tensores por par + contexto del manifiesto
            -> Keras 3 (backend TensorFlow)
            -> artifacts/mlp/

Uso (desde la raíz del repositorio):
    uv run python scripts/train_mlp.py
    uv run python scripts/train_mlp.py --episodes-dir data/episodes/smoke200 --split hash
    uv run python scripts/train_mlp.py --tag lr3e-4 --override optimization.learning_rate=0.0003
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.modeling.dataset import (  # noqa: E402
    assert_no_episode_leakage,
    drop_non_optimal,
    load_episode_tables,
    split_by_episode_hash,
    split_by_time,
    summarize_splits,
)
from src.modeling.features import (  # noqa: E402
    BlockScaler,
    as_model_inputs,
    balanced_sample_weights,
    build_all_episodes,
    build_model_arrays,
)
from src.modeling.mlp_classifier import (  # noqa: E402
    MLPConfig,
    build_callbacks,
    build_pairwise_mlp,
    compile_model,
    model_summary_text,
)
from src.pipeline.transformation.derived_fields import VehicleClassConfig  # noqa: E402

DEFAULT_EPISODES_DIR = REPO_ROOT / "data" / "episodes"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "mlp"
CLASS_CONFIG_PATH = REPO_ROOT / "config" / "vehicle_classes.yaml"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "mlp.yaml"


def load_config(path: Path, overrides: list[str]) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for override in overrides:
        key, _, value = override.partition("=")
        section, _, field = key.partition(".")
        if not field or section not in raw:
            raise ValueError(f"--override mal formado: {override!r} (use seccion.campo=valor)")
        raw[section][field] = yaml.safe_load(value)
    return raw


def flatten_model_config(raw: dict) -> MLPConfig:
    merged = {**raw.get("model", {}), **raw.get("optimization", {})}
    return MLPConfig.from_dict(merged)


def plot_learning_curves(history: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(history["loss"], label="entrenamiento (ponderada)")
    axes[0].plot(history["val_loss"], label="validación (sin ponderar)")
    axes[0].set_title("Pérdida (entropía cruzada)")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # La separación entre ambas curvas NO es sobreajuste: el entrenamiento aplica
    # `sample_weight` para compensar el desbalance de SIN_CAMION y la validación no,
    # así que las dos series no están en la misma escala. Evaluadas ambas sin pesos,
    # la pérdida de validación resulta incluso menor que la de entrenamiento.
    axes[0].text(
        0.5,
        -0.30,
        "Las dos series no son comparables directamente: el entrenamiento aplica pesos de\n"
        "clase y la validación no. Sin pesos, la pérdida de validación es menor que la de\n"
        "entrenamiento (ver training_report.json → unweighted_loss).",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        style="italic",
    )

    key = "raw_assignment_accuracy"
    axes[1].plot(history[key], label="entrenamiento")
    axes[1].plot(history[f"val_{key}"], label="validación")
    axes[1].set_title("Exactitud cruda de asignación")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Curvas de aprendizaje — MLP por par (vehículo, camión)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--split",
        choices=("time", "hash"),
        default="time",
        help="'time' es la partición honesta; 'hash' sólo para la muestra de humo, "
        "que cae entera en una sola semana.",
    )
    parser.add_argument("--tag", default=None, help="Sufijo para el subdirectorio de salida")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    raw_config = load_config(args.config, args.override)
    config = flatten_model_config(raw_config)

    import keras

    keras.utils.set_random_seed(config.seed)

    out_dir = args.out_dir if args.tag is None else args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Datos -------------------------------------------------------------
    joined = load_episode_tables(
        args.episodes_dir / "episodes.parquet",
        args.episodes_dir / "episode_vehicles.parquet",
    )
    joined, n_non_optimal = drop_non_optimal(joined)
    if args.max_episodes:
        keep = sorted(joined["episode_id"].unique())[: args.max_episodes]
        joined = joined[joined["episode_id"].isin(keep)].reset_index(drop=True)

    data_cfg = raw_config.get("data", {})
    if args.split == "time":
        splits = split_by_time(
            joined,
            tuple(data_cfg.get("train_years", (2018, 2019, 2020, 2021, 2022, 2023, 2024))),
            tuple(data_cfg.get("val_years", (2025,))),
            tuple(data_cfg.get("test_years", (2026,))),
        )
    else:
        splits = split_by_episode_hash(joined)

    assert_no_episode_leakage(splits)
    summaries = summarize_splits(splits)
    for s in summaries:
        if s.n_rows == 0:
            raise SystemExit(
                f"La partición '{s.name}' quedó vacía. Con la muestra de humo use "
                "--split hash: los 200 episodios caen todos en 2018-W02."
            )

    classes = sorted(VehicleClassConfig.from_yaml(str(CLASS_CONFIG_PATH)).in_scope_classes)
    episodes = {name: build_all_episodes(df, classes) for name, df in splits.items()}

    max_trucks = max(e.n_trucks for eps in episodes.values() for e in eps)
    scaler = BlockScaler.fit(episodes["train"], classes)
    arrays = {n: build_model_arrays(e, scaler, max_trucks) for n, e in episodes.items()}

    train, val = arrays["train"], arrays["val"]
    sample_weight = balanced_sample_weights(train.target)

    print(f"Clases en alcance: {classes}")
    print(f"Camiones máximos en el dataset: {max_trucks}")
    for s in summaries:
        print(
            f"  {s.name:<5} episodios={s.n_episodes:>7,}  filas={s.n_rows:>8,}  "
            f"diferidos={s.n_deferred_rows:>7,} ({s.deferred_pct:.2f}%)  años={list(s.years)}"
        )

    # --- Modelo ------------------------------------------------------------
    model = compile_model(
        build_pairwise_mlp(train.pair.shape[-1], train.defer.shape[-1], config), config
    )
    (out_dir / "model_summary.txt").write_text(model_summary_text(model), encoding="utf-8")

    t0 = time.perf_counter()
    history = model.fit(
        as_model_inputs(train),
        train.target,
        sample_weight=sample_weight,
        validation_data=(as_model_inputs(val), val.target),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=build_callbacks(
            config,
            checkpoint_path=str(out_dir / "best_model.keras"),
            history_path=str(out_dir / "training_history.csv"),
        ),
        verbose=2,
    )
    elapsed = time.perf_counter() - t0

    model.save(out_dir / "model.keras")
    plot_learning_curves(history.history, out_dir / "learning_curves.png")

    # El historial compara pérdida de entrenamiento PONDERADA contra validación SIN
    # ponderar, así que su separación no mide sobreajuste. Se reevalúan las tres
    # particiones sin pesos para tener cifras comparables entre sí.
    unweighted = {}
    for name, a in arrays.items():
        loss, accuracy = model.evaluate(as_model_inputs(a), a.target, verbose=0, batch_size=1024)
        unweighted[name] = {"loss": float(loss), "raw_assignment_accuracy": float(accuracy)}

    (out_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "max_trucks_padding": max_trucks,
                "pair_dim": int(train.pair.shape[-1]),
                "defer_dim": int(train.defer.shape[-1]),
                "blocks": scaler.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (out_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "0": "SIN_CAMION",
                **{str(i + 1): f"CAMION_{i + 1}" for i in range(max_trucks)},
                "_nota": (
                    "Índices canónicos: CAMION_1 es el camión de MAYOR capacidad del "
                    "episodio, no el primero que produjo el generador. Ver "
                    "src/modeling/canonicalization.py."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    best_epoch = int(np.argmin(history.history["val_loss"]))
    label_counts = {
        name: {
            str(k): int(v) for k, v in zip(*np.unique(a.target, return_counts=True), strict=True)
        }
        for name, a in arrays.items()
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                "episodes_dir": str(args.episodes_dir),
                "split_strategy": args.split,
                "episodes_dropped_non_optimal": n_non_optimal,
                "splits": [s.as_dict() for s in summaries],
                "canonical_label_counts": label_counts,
                "deferred_weight": float(sample_weight[train.target == 0][:1].tolist()[0])
                if (train.target == 0).any()
                else 1.0,
                "config": config.as_dict(),
                "unweighted_loss": unweighted,
                "epochs_run": len(history.history["loss"]),
                "best_epoch_zero_based": best_epoch,
                "best_val_loss": float(history.history["val_loss"][best_epoch]),
                "best_val_raw_accuracy": float(
                    history.history["val_raw_assignment_accuracy"][best_epoch]
                ),
                "training_seconds": round(elapsed, 1),
                "n_parameters": int(model.count_params()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nEntrenamiento: {len(history.history['loss'])} épocas en {elapsed:.1f}s")
    print(f"Mejor época: {best_epoch} (val_loss={history.history['val_loss'][best_epoch]:.4f})")
    print(f"Parámetros: {model.count_params():,}")
    print(f"Artefactos en {out_dir}")


if __name__ == "__main__":
    main()
