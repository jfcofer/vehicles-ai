# Tarea #4 — Entregables de Juan Francisco Fernández Ramos

Índice de lo producido para la Tarea #4, con su destino en el reporte del grupo
(el Google Doc en formato IEEE).

| Archivo | Qué es | Dónde va |
|---|---|---|
| [`01_tabla_v_caso_uso_analisis.md`](01_tabla_v_caso_uso_analisis.md) | Los dos escenarios del caso de uso de análisis | **Tabla V**, pág. 13 — reemplaza los marcadores `1.`, `2a.`, `a` |
| [`02_seccion_VI_D_mlp.md`](02_seccion_VI_D_mlp.md) | Arquitectura, plataforma, esquema de datos e hiper-parámetros justificados, con dos diagramas UML | **Sección VI-D**, pág. 25 — hoy vacía |
| [`03_resultados_mlp.md`](03_resultados_mlp.md) | Resultados medidos, ablaciones y limitaciones | **Sección VII** — aporta la parte del MLP |
| [`04_anexo_ia_juan.md`](04_anexo_ia_juan.md) | Preguntas, respuestas, qué se aceptó y qué se rechazó | **Anexo A, B y C** |
| [`05_hallazgos_para_el_equipo.md`](05_hallazgos_para_el_equipo.md) | Lo que afecta a los cinco modelos y los vacíos sin dueño | **Interno — no va al reporte** |
| [`06_canonicalizacion_y_etiquetado.md`](06_canonicalizacion_y_etiquetado.md) | Las cuatro fuentes de arbitrariedad de la etiqueta, el techo exacto de exactitud y el arreglo medido | **Interno**, salvo §5 y §8 → **VII** y **VIII** |
| [`07_mensaje_al_equipo.md`](07_mensaje_al_equipo.md) | Versión corta del `06_` para enviar por chat, con la decisión que hay que tomar | **Interno — para Víctor y Nicolás** |

Transcripción de la sesión de IA:
[`chat/2026-07-25-juan-mlp-design-training-evaluation.md`](../../chat/2026-07-25-juan-mlp-design-training-evaluation.md).

---

## Resumen de lo entregado

**Los cuatro entregables asignados en la planificación (págs. 4, 6, 7 y 8) están
completos**, con resultados medidos sobre el conjunto de datos completo — 34.839
episodios y 534.680 filas — y no sobre la muestra de 200 episodios.

Titulares:

- El plan entregado **nunca excede la capacidad** de un camión, en ninguno de los 34.839
  episodios.
- Sobre el objetivo primario —cuántos vehículos se transportan— el modelo **iguala la
  solución óptima en el 97,78 %** de los episodios de prueba, frente al 87,98 % de la
  heurística greedy.
- La arquitectura de puntuación por par cumple el requisito de flota sin límite codificado:
  se verificó ejecutando **los mismos pesos sobre manifiestos de diez camiones**, sin
  reentrenar.
- **La baja exactitud (0,53) es una propiedad del generador de datos, no del modelo — y es
  recuperable.** El techo exacto sobre estas etiquetas es **0,9243** y el modelo alcanza el
  58,8 % de él. La brecha se explica: el orden aleatorio de la flota cambia el *plan* del
  etiquetador y no es una entrada observable. Fijando ese orden, el mismo modelo llega a
  **0,8458** de exactitud y **0,8131** de F1 macro sin mover las métricas operativas.
  Sólo unos 8 puntos son ruido irreducible.

Limitaciones declaradas, no ocultadas: el modelo **no aporta a la elección de camión**
(demostrado por ablación), pierde frente al greedy en aprovechamiento de CU, y la
generalización a flotas grandes está demostrada como factibilidad, no como calidad.

> **Corrección del 27 de julio.** Una versión anterior de este resumen concluía que "cerca
> del 60 % de la etiqueta es ruido irreducible", apoyándose en la auto-concordancia del
> etiquetador (0,3983). Esa cifra nunca fue un techo. Calculado el techo real, la mayor
> parte de esa brecha resultó **eliminable**, no irreducible. El análisis completo está en
> [`06_canonicalizacion_y_etiquetado.md`](06_canonicalizacion_y_etiquetado.md).

---

## Reproducir

```bash
git lfs pull                                        # 522 MB de datos reales
uv venv --python 3.12 && uv sync                    # Keras 3.15 + TensorFlow 2.21
uv run python scripts/build_vehicle_features.py     # ~1 min
uv run python scripts/build_scenarios.py            # ~7 min
uv run python scripts/train_mlp.py                  # ~3 min en CPU
uv run python scripts/evaluate_mlp.py
uv run python scripts/teacher_self_agreement.py --years 2026
uv run python scripts/label_ceiling.py              # techo exacto de exactitud
uv run pytest tests/modeling                        # 88 pruebas
```

Experimento del orden de la flota (§6 del documento 06), sin tocar `data/episodes/`:

```bash
uv run python artifacts/mlp/fleet_order_experiment/build_sorted_episodes.py \
    --order asc --out /tmp/episodes_asc              # ~8 min
uv run python scripts/train_mlp.py    --episodes-dir /tmp/episodes_asc --out-dir /tmp/mlp_asc
uv run python scripts/evaluate_mlp.py --model-dir /tmp/mlp_asc --episodes-dir /tmp/episodes_asc
uv run python scripts/teacher_self_agreement.py --years 2026 --fleet-order asc
```

> **Nota de entorno.** El intérprete del sistema es Python 3.14 y TensorFlow 2.21 sólo
> publica ruedas hasta CPython 3.13. Sin el `--python 3.12` el entorno no puede instalar
> Keras.

Extrapolación a flotas mayores:

```bash
uv run python scripts/build_extrapolation_set.py --n-trucks 8 10 --cap-mode constant-total
uv run python scripts/evaluate_mlp.py --split single --policy model \
    --episodes-dir data/episodes/extrap_8_10_constanttotal \
    --out-name metrics_extrap_8_10_constanttotal.json
```

## Código añadido

Todo es **aditivo**: no se modificó `src/loading/labeler.py`, `src/loading/scenarios.py`
ni `scripts/build_scenarios.py`, para no bloquear el trabajo de Víctor y Nicolás.

```
config/mlp.yaml
pyproject.toml + uv.lock
src/modeling/{canonicalization,dataset,features,mlp_classifier,capacity_decoder,metrics}.py
scripts/{train_mlp,evaluate_mlp,sweep_mlp,build_extrapolation_set,teacher_self_agreement}.py
scripts/label_ceiling.py
tests/modeling/            88 pruebas
artifacts/mlp/             modelo, métricas, curvas, matriz de confusión
artifacts/mlp/fleet_order_experiment/   evidencia del §6 del documento 06
```
