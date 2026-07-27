#!/usr/bin/env python3
"""Búsqueda controlada de hiper-parámetros, seleccionada por métrica de dominio.

El criterio de selección es la **brecha de conteo en validación**, no la
exactitud: el objetivo primario del maestro es cuántos vehículos carga, y la
exactitud cruda está contaminada por el desempate arbitrario del etiquetador
(ver `scripts/teacher_self_agreement.py`).

Cada configuración entrena y evalúa por separado bajo `artifacts/mlp/sweep/<tag>`;
el resumen comparativo queda en `artifacts/mlp/sweep/summary.json`.

Uso (desde la raíz del repositorio):
    uv run python scripts/sweep_mlp.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Rejilla deliberadamente pequeña: con la brecha de conteo ya en +0.02 el margen
# de mejora está en la brecha de CU, que es donde el greedy todavía gana.
GRID: list[tuple[str, list[str]]] = [
    ("base_64_32", []),
    ("ancha_128_64_32", ["model.pair_units=[128, 64, 32]", "model.defer_units=[64, 32]"]),
    ("dropout_010", ["model.dropout=0.10"]),
    ("dropout_030", ["model.dropout=0.30"]),
    ("lr_3e-4", ["optimization.learning_rate=0.0003"]),
    ("lr_3e-3", ["optimization.learning_rate=0.003"]),
    ("batch_512", ["optimization.batch_size=512"]),
    (
        "ancha_dropout_010",
        [
            "model.pair_units=[128, 64, 32]",
            "model.dropout=0.10",
            "optimization.learning_rate=0.0003",
        ],
    ),
]


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Falló {' '.join(cmd)}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "artifacts" / "mlp" / "sweep")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, overrides in GRID:
        out_dir = args.out_root / tag
        flags = [f for o in overrides for f in ("--override", o)]
        print(f"--- {tag}: {overrides or 'configuración base'}", flush=True)

        t0 = time.perf_counter()
        run(
            [
                sys.executable,
                "scripts/train_mlp.py",
                "--out-dir",
                str(args.out_root),
                "--tag",
                tag,
                *flags,
            ]
        )
        run([sys.executable, "scripts/evaluate_mlp.py", "--model-dir", str(out_dir)])
        elapsed = time.perf_counter() - t0

        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        report = json.loads((out_dir / "training_report.json").read_text(encoding="utf-8"))
        val, test = metrics["model"]["val"], metrics["model"]["test"]
        rows.append(
            {
                "tag": tag,
                "overrides": overrides,
                "n_parameters": report["n_parameters"],
                "epochs_run": report["epochs_run"],
                "decoder_policy": metrics["decoder_policy_selected"],
                "val_loaded_gap_mean": val["loaded_gap_mean"],
                "val_cu_gap_mean": val["cu_gap_mean"],
                "val_capacity_violation_rate": val["capacity_violation_rate"],
                "val_class_level_agreement": val["class_level_agreement_mean"],
                "test_loaded_gap_mean": test["loaded_gap_mean"],
                "test_cu_gap_mean": test["cu_gap_mean"],
                "test_episodes_matching_pct": test["episodes_matching_teacher_count_pct"],
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"    val brecha_conteo={val['loaded_gap_mean']:+.4f}  "
            f"val brecha_cu={val['cu_gap_mean']:+.4f}  ({elapsed:.0f}s)",
            flush=True,
        )

    # Desempate: primero conteo (objetivo primario), luego CU (secundario).
    rows.sort(key=lambda r: (r["val_loaded_gap_mean"], r["val_cu_gap_mean"]))
    (args.out_root / "summary.json").write_text(
        json.dumps(
            {"selection_metric": "val_loaded_gap_mean, luego val_cu_gap_mean", "runs": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Ranking por brecha de conteo en validación ===")
    for r in rows:
        print(
            f"  {r['tag']:<22} conteo={r['val_loaded_gap_mean']:+.4f}  "
            f"cu={r['val_cu_gap_mean']:+.4f}  params={r['n_parameters']:>6,}  "
            f"epocas={r['epochs_run']:>3}"
        )
    print(f"\nGanadora: {rows[0]['tag']}  ->  {args.out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
