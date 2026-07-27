#!/usr/bin/env python3
"""Evalúa el MLP a nivel de episodio contra el maestro exacto.

Selecciona la política del decoder por **validación** (brecha de conteo, no
exactitud) y reporta la partición de prueba con esa política, junto a la línea
base greedy.

Uso (desde la raíz del repositorio):
    uv run python scripts/evaluate_mlp.py
    uv run python scripts/evaluate_mlp.py --model-dir artifacts/mlp/smoke200 \
        --episodes-dir data/episodes/smoke200 --split hash
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

from src.modeling.capacity_decoder import POLICIES  # noqa: E402
from src.modeling.dataset import (  # noqa: E402
    drop_non_optimal,
    load_episode_tables,
    split_by_episode_hash,
    split_by_time,
)
from src.modeling.features import (  # noqa: E402
    BlockScaler,
    as_model_inputs,
    build_all_episodes,
    build_model_arrays,
)
from src.modeling.metrics import aggregate, evaluate_greedy, evaluate_model  # noqa: E402

DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "mlp"
DEFAULT_EPISODES_DIR = REPO_ROOT / "data" / "episodes"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "mlp.yaml"


def plot_confusion(matrix: list[list[int]], labels: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.asarray(matrix, dtype=float)
    # Normalizada por fila: sin esto, MOTOCICLETA/CAMION_1 aplasta todo lo demás.
    with np.errstate(invalid="ignore"):
        norm = np.where(m.sum(axis=1, keepdims=True) > 0, m / m.sum(axis=1, keepdims=True), 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicho por el modelo")
    ax.set_ylabel("Maestro exacto")
    ax.set_title("Matriz de confusión (normalizada por fila)\nEtiquetas canónicas por capacidad")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}\n{int(m[i, j]):,}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="Proporción de la fila")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def measure_latency(model, episodes, arrays, policy: str, sample: int = 200) -> dict:
    """Latencia de un manifiesto completo: puntuación + decodificación."""
    from src.modeling.capacity_decoder import decode_episode
    from src.modeling.metrics import episode_logits

    rng = np.random.default_rng(0)
    picks = rng.choice(len(episodes), size=min(sample, len(episodes)), replace=False)
    timings = []
    for ep_i in picks:
        ep = episodes[ep_i]
        rows = np.flatnonzero(arrays.episode_index == ep_i)
        batch = {k: v[rows] for k, v in as_model_inputs(arrays).items()}
        t0 = time.perf_counter()
        logits = model.predict(batch, verbose=0)
        decode_episode(
            episode_logits(logits, np.arange(len(rows)), ep.n_trucks),
            cu=ep.cu,
            capacities=ep.capacities,
            policy=policy,
        )
        timings.append((time.perf_counter() - t0) * 1000.0)

    t = np.asarray(timings)
    return {
        "n_manifests_timed": int(len(t)),
        "mean_ms": float(t.mean()),
        "median_ms": float(np.median(t)),
        "p99_ms": float(np.quantile(t, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", choices=("time", "hash", "single"), default="time")
    parser.add_argument(
        "--policy",
        default=None,
        help="Fuerza la política del decoder. Obligatorio con --split single, donde no "
        "hay validación con la que elegirla.",
    )
    parser.add_argument("--out-name", default="metrics.json")
    args = parser.parse_args()
    if args.split == "single" and args.policy is None:
        parser.error("--split single requiere --policy (no hay validación para elegirla).")

    import keras

    model = keras.models.load_model(args.model_dir / "model.keras")
    schema = json.loads((args.model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    classes = schema["classes"]
    scaler = BlockScaler.from_dict(schema["blocks"])
    max_trucks = int(schema["max_trucks_padding"])

    joined = load_episode_tables(
        args.episodes_dir / "episodes.parquet",
        args.episodes_dir / "episode_vehicles.parquet",
    )
    joined, _ = drop_non_optimal(joined)

    data_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")).get("data", {})
    if args.split == "single":
        # Modo extrapolación: todo el directorio es un único conjunto de prueba, y
        # el relleno se toma del propio conjunto -- que es justamente el punto:
        # el modelo guardado acepta más camiones de los que vio entrenando.
        splits = {"test": joined}
        episodes = {"test": build_all_episodes(joined, classes)}
        max_trucks = max(e.n_trucks for e in episodes["test"])
        arrays = {"test": build_model_arrays(episodes["test"], scaler, max_trucks)}
        logits = {"test": model.predict(as_model_inputs(arrays["test"]), verbose=0)}
        n_classes, n_labels = len(classes), max_trucks + 1
        results = evaluate_model(
            episodes["test"], arrays["test"], logits["test"], args.policy, n_classes
        )
        m = aggregate(results, n_labels)
        g = aggregate(evaluate_greedy(episodes["test"], arrays["test"], n_classes), n_labels)
        latency = measure_latency(model, episodes["test"], arrays["test"], args.policy)

        payload = {
            "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "mode": "single",
            "episodes_dir": str(args.episodes_dir),
            "decoder_policy_selected": args.policy,
            "max_trucks_in_set": max_trucks,
            "max_trucks_seen_in_training": int(schema["max_trucks_padding"]),
            "model": {"test": m},
            "baseline_greedy": {"test": g},
            "inference_latency_per_manifest": latency,
        }
        (args.model_dir / args.out_name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"Camiones en el conjunto: hasta {max_trucks} "
            f"(el entrenamiento vio hasta {schema['max_trucks_padding']})"
        )
        print(f"Episodios: {m['n_episodes']:,}  filas: {m['n_vehicle_rows']:,}")
        print(f"  1. Violación de capacidad      {m['capacity_violation_rate']:.4f}")
        print(
            f"  2. Brecha de conteo (media)    {m['loaded_gap_mean']:+.4f}   "
            f"greedy {g['loaded_gap_mean']:+.4f}"
        )
        print(
            f"     Episodios que igualan       {m['episodes_matching_teacher_count_pct']:.2f}%  "
            f"greedy {g['episodes_matching_teacher_count_pct']:.2f}%"
        )
        print(
            f"  3. Utilización                 {m['cu_utilization_model_pct']:.2f}%  "
            f"maestro {m['cu_utilization_teacher_pct']:.2f}%"
        )
        print(f"  5. Concordancia por clase      {m['class_level_agreement_mean']:.4f}")
        print(f"  6. Latencia media              {latency['mean_ms']:.2f} ms")
        print(f"\nMétricas en {args.model_dir / args.out_name}")
        return

    if args.split == "time":
        splits = split_by_time(
            joined,
            tuple(data_cfg["train_years"]),
            tuple(data_cfg["val_years"]),
            tuple(data_cfg["test_years"]),
        )
    else:
        splits = split_by_episode_hash(joined)

    episodes = {n: build_all_episodes(df, classes) for n, df in splits.items()}
    arrays = {n: build_model_arrays(e, scaler, max_trucks) for n, e in episodes.items()}
    logits = {n: model.predict(as_model_inputs(a), verbose=0) for n, a in arrays.items()}

    n_labels = max_trucks + 1
    labels = ["SIN_CAMION"] + [f"CAMION_{i + 1}" for i in range(max_trucks)]

    # --- 1. La política del decoder se elige en VALIDACIÓN. ------------------
    n_classes = len(classes)
    policy_scan = {}
    for policy in POLICIES:
        results = evaluate_model(episodes["val"], arrays["val"], logits["val"], policy, n_classes)
        policy_scan[policy] = aggregate(results, n_labels)

    best_policy = min(policy_scan, key=lambda p: policy_scan[p]["loaded_gap_mean"])
    print("Selección de política del decoder (por brecha de conteo en validación):")
    for policy, m in policy_scan.items():
        marca = " <-- elegida" if policy == best_policy else ""
        print(
            f"  {policy:<14} brecha_conteo={m['loaded_gap_mean']:+.4f}  "
            f"violaciones={m['capacity_violation_rate']:.4f}  "
            f"F1_macro={m['macro_f1']:.4f}{marca}"
        )

    # --- 2. Todas las particiones con la política elegida. -------------------
    model_metrics, greedy_metrics = {}, {}
    for name in ("train", "val", "test"):
        model_metrics[name] = aggregate(
            evaluate_model(episodes[name], arrays[name], logits[name], best_policy, n_classes),
            n_labels,
        )
        greedy_metrics[name] = aggregate(
            evaluate_greedy(episodes[name], arrays[name], n_classes), n_labels
        )

    latency = measure_latency(model, episodes["test"], arrays["test"], best_policy)

    plot_confusion(
        model_metrics["test"]["confusion_matrix"], labels, args.model_dir / "confusion_matrix.png"
    )

    payload = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "model_dir": str(args.model_dir),
        "episodes_dir": str(args.episodes_dir),
        "split_strategy": args.split,
        "decoder_policy_selected": best_policy,
        "decoder_policy_scan_on_val": {p: _slim(m) for p, m in policy_scan.items()},
        "labels": labels,
        "model": model_metrics,
        "baseline_greedy": greedy_metrics,
        "inference_latency_per_manifest": latency,
    }
    (args.model_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- 3. Salida legible. --------------------------------------------------
    for name in ("val", "test"):
        m, g = model_metrics[name], greedy_metrics[name]
        print(
            f"\n=== {name.upper()} ({m['n_episodes']:,} episodios, {m['n_vehicle_rows']:,} filas)"
        )
        print(
            f"  1. Violación de capacidad     modelo {m['capacity_violation_rate']:.4f}   "
            f"greedy {g['capacity_violation_rate']:.4f}"
        )
        print(
            f"  2. Brecha de conteo (media)   modelo {m['loaded_gap_mean']:+.4f}   "
            f"greedy {g['loaded_gap_mean']:+.4f}"
        )
        print(
            f"     Episodios que igualan      "
            f"modelo {m['episodes_matching_teacher_count_pct']:.2f}%  "
            f"greedy {g['episodes_matching_teacher_count_pct']:.2f}%"
        )
        print(
            f"  3. Brecha de CU (media)       modelo {m['cu_gap_mean']:+.4f}   "
            f"greedy {g['cu_gap_mean']:+.4f}"
        )
        print(
            f"     Utilización                modelo {m['cu_utilization_model_pct']:.2f}%  "
            f"greedy {g['cu_utilization_model_pct']:.2f}%  "
            f"maestro {m['cu_utilization_teacher_pct']:.2f}%"
        )
        print(
            f"  4. Diferidos                  modelo {m['deferred_model_total']:,}   "
            f"maestro {m['deferred_teacher_total']:,}"
        )
        print(
            f"  5. F1 macro                   modelo {m['macro_f1']:.4f}   "
            f"greedy {g['macro_f1']:.4f}"
        )
        print(
            f"     Concordancia por clase     modelo {m['class_level_agreement_mean']:.4f}   "
            f"greedy {g['class_level_agreement_mean']:.4f}"
        )
        print(
            f"     Planes idénticos al maestro "
            f"modelo {m['episodes_identical_to_teacher_pct']:.2f}%  "
            f"greedy {g['episodes_identical_to_teacher_pct']:.2f}%"
        )
        print(
            f"  7. Exactitud cruda            modelo {m['raw_assignment_accuracy']:.4f}   "
            f"greedy {g['raw_assignment_accuracy']:.4f}"
        )

    print(
        f"\n  6. Latencia por manifiesto: media {latency['mean_ms']:.2f} ms, "
        f"p99 {latency['p99_ms']:.2f} ms"
    )
    print(f"\nMétricas en {args.model_dir / 'metrics.json'}")


def _slim(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "confusion_matrix"}


if __name__ == "__main__":
    main()
