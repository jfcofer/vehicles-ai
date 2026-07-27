# Hallazgos para el equipo — Tarea #4

> **Documento interno. No va al reporte.**
> Autor: Juan Francisco Fernández Ramos · 25 de julio de 2026
> Para: Víctor Borbor, Nicolás Fiallo

Al implementar el MLP encontré cosas que **afectan a los cinco modelos**, no sólo al
mío. Las dejo aquí con la evidencia para que ustedes decidan, no para imponer nada.
Nada de lo que hice modifica `src/loading/labeler.py`, `src/loading/scenarios.py` ni
`scripts/build_scenarios.py`: todo vive en `src/modeling/`, así que su trabajo no se
rompe con esto.

---

## 1. Sin el join, ningún modelo ve las capacidades de la flota

`episode_vehicles.parquet` tiene el vehículo y su etiqueta, pero **no** cuántos camiones
hay ni de qué tamaño. Eso está sólo en `episodes.parquet` (`n_trucks`,
`truck_capacities`).

Un modelo entrenado sólo con la tabla de vehículos está tratando de predecir a qué
camión va un vehículo sin saber si hay uno o cuatro camiones. Si alguno de ustedes está
leyendo únicamente `episode_vehicles.parquet`, ése es probablemente el primer problema a
revisar.

El join está en `src/modeling/dataset.py:load_episode_tables()` y se puede usar tal cual.

---

## 2. La etiqueta `CAMION_k` es en gran parte arbitraria — lo medí

Esto es lo más importante del documento.

`generate_fleet()` produce las capacidades en orden aleatorio, y la programación dinámica
del maestro recorre los camiones **por índice**, llenando el de índice 0 tan lleno como
puede antes de pasar al siguiente (`labeler.py:188-213`). Ese índice 0 es un camión de
capacidad aleatoria. Además, dentro de una clase el maestro reparte los cupos con un
`random.shuffle` sembrado (`labeler.py:224-229`): dos vehículos con features idénticas
reciben etiquetas distintas por sorteo.

Para saber cuánto pesa esto, le presenté al maestro **la misma flota permutada** — mismos
camiones, mismas capacidades, situación operativamente idéntica — y comparé su nueva
respuesta con la original (`scripts/teacher_self_agreement.py`, sobre los 1.158 episodios
de 2026 con dos o más camiones):

| | |
|---|---|
| Exactitud cruda que el maestro reproduce de sí mismo | **0,3983** |
| Concordancia por clase | **0,4493** |
| Episodios reproducidos idénticos | **35,58 %** |
| \|Δ vehículos cargados\| | **0,0000** |
| \|Δ CU aprovechada\| | **0,0000** |

Léanlo así: **el propio oráculo que generó las etiquetas sólo reproduce ~40 % de ellas**,
mientras que el resultado operativo es 100 % determinista. El ~60 % restante es ruido de
desempate. Ningún modelo — ni el mío, ni LightGBM, ni Random Forest — puede predecirlo,
porque no está en las entradas.

**Consecuencia práctica: si están midiendo el éxito por accuracy, están midiendo el
techo del ruido, no la calidad del plan.** Mi MLP saca 0,5297 de exactitud cruda, por
encima de la autoconsistencia del maestro, y aun así entrega planes que igualan el conteo
óptimo en el 97,78 % de los episodios.

### Qué propongo

1. **Reportar métricas de dominio como principales** y accuracy como diagnóstico
   secundario: violación de capacidad, brecha de vehículos cargados, brecha de CU,
   diferidos, latencia. Es lo que ya pide `05_evaluation.md`.
2. **Canonicalizar la flota antes de entrenar.** `src/modeling/canonicalization.py`
   ordena los camiones por capacidad descendente y remapea las etiquetas. No cambia
   ningún plan: sólo cambia el nombre del camión. Está probado
   (`tests/modeling/test_canonicalization.py`) y se puede importar sin arrastrar el resto
   de mi paquete. Si los tres lo aplicamos, la comparación entre los cinco modelos es
   justa; si no, cada uno mide contra una convención distinta.

---

## 3. Features que conviene NO usar

| Feature | Por qué |
|---|---|
| `uid`, `codigo_vehiculo` | Identificadores. Sólo permiten memorizar. |
| Posición del vehículo dentro de su clase | Sale del `shuffle` sembrado del maestro. Es ruido puro. |
| `canton` | El maestro **lo ignora**: `labeler.py:139-145` agrupa por clase y el cantón no entra en la restricción de capacidad. Como entrada sólo puede aprender ruido o identidad de episodio. Conviene conservarlo en la salida y el reporte, pero medirlo en ablación antes de meterlo al modelo. |
| El número del camión | Es exactamente la etiqueta arbitraria del punto 2. |

Sí sirven: `cu`, one-hot de clase, conteo de la misma clase en el manifiesto, y los
agregados del manifiesto y de la flota.

---

## 4. La partición no puede ser por filas

`train_test_split` sobre filas de vehículos pone vehículos del mismo episodio a ambos
lados. Comparten flota, manifiesto y contexto agregado: las métricas salen infladas y no
sobreviven a datos nuevos.

La partición honesta es temporal y por episodio completo. **Ojo: 2017 no existe en el
dataset** — ese CSV del SRI no trae `FECHA PROCESO` y `load_all_years` lo descarta (está
escrito en `08_feature_coverage.md`, "Skipped years"). La cobertura real es 2018-2026, así
que la partición es:

```
Entrenamiento: 2018-2024   Validación: 2025   Prueba: 2026
```

Implementado en `src/modeling/dataset.py:split_by_time()`, con
`assert_no_episode_leakage()` que falla ruidosamente si algo se cruza.

---

## 5. Cifras reales del dataset completo

Generé el dataset completo (`scripts/build_scenarios.py` sin `--limit`, 7 min):

| | |
|---|---|
| Grupos cantón-semana totales | 55.076 |
| Excluidos por el piso N<5 | 20.237 |
| Episodios construidos | **34.839** |
| Filas en `episode_vehicles.parquet` | **534.680** |
| Episodios triviales (nadie diferido) | 29.860 (85,7 %) |
| Episodios no-óptimos | **0** |
| Vehículos diferidos | 22.653 (**4,24 %** de las filas) |

Dos cosas para el reporte:

- La sección VI-A dice *"35 mil × 13 = 455 mil ejemplos"*. El número medido es
  **534.680**. Conviene corregirlo, ahora que existe.
- La sección VI-A también dice que `SIN CAMION` está *"alrededor de 33 veces"*
  subrepresentada y fija `scale_pos_weight = 33`. Medido, el desbalance es de
  **22,6 a 1** (4,24 %), no 33 a 1. Sigue justificando ponderar la clase, pero el número
  conviene recalcularlo sobre la partición de entrenamiento de cada modelo.

---

## 6. Inconsistencias entre el reporte y el repositorio

Las dejo listadas; no son mías para arreglar solo.

| Dónde | Dice | Realidad |
|---|---|---|
| `reports/.../04_method.md` | El modelo estudiante se implementa en **PyTorch** | La planificación y la sección VI del reporte dicen **Keras 3**. Yo usé Keras 3.15 con backend TensorFlow 2.21. |
| Reporte, Sec. VI | Polars 1.42, PyArrow 24.0, `uv` con bloqueo de versiones | El repo usaba `requirements.txt` sin versiones y pandas. Añadí `pyproject.toml` + `uv.lock` (aditivo, `requirements.txt` sigue ahí). PyArrow resuelto es 25.0. El pipeline sigue en pandas: si queremos afirmar Polars, hay que migrarlo o corregir el texto. |
| Reporte, Sec. VI | Python 3.12 | Correcto, pero el intérprete del sistema es 3.14 y **TensorFlow no publica ruedas para 3.14**. Hay que crear el entorno con `uv venv --python 3.12` o Keras no instala. |
| `05_evaluation.md` | *"(To be filled after `scripts/eval_loading.py` runs)"* | Ese script no existe. Yo hice `scripts/evaluate_mlp.py` para mi modelo; los resultados de los cinco siguen sin consolidar. |

---

## 7. Vacíos del reporte que no tienen dueño visible

De la planificación, y contrastado con el PDF actual:

| Sección | Estado | Asignado a |
|---|---|---|
| V-A Diseño preliminar de interfaz gráfica (pág. 21) | **Vacía** | Víctor |
| VI-B Regresión logística multinomial (pág. 25) | **Vacía** | Nicolás |
| VI-C Random Forest (pág. 25) | **Vacía** | Nicolás |
| VIII Posibilidades futuras (pág. 27) | **Vacía** | Nicolás |
| VII Resultados | Sólo define métricas, sin números | Sin dueño |
| Consolidación de los 5 modelos en una tabla comparable | No existe | **Sin dueño** |
| Modelo formal de diseño de la arquitectura | La Fig. 9 es un diagrama de flujo, no un modelo de diseño. La orden 2 pide lenguaje de modelado formal | **Sin dueño** — yo aporto el de componentes del MLP en `02_seccion_VI_D_mlp.md`, pero falta el del sistema completo |

### Erratas de edición detectadas

- Referencia `{9}` debería ser `[9]`.
- Las referencias [8] y [9] son vídeos de YouTube. Para formato IEEE de ciencias de la
  computación conviene sustituirlas por la documentación oficial de XGBoost/LightGBM o
  los papers originales.
- **Numeración de figuras rota:** el texto cita "Fig. 3", "Fig. 4", "Fig. 5", "Fig. 6" y
  "Fig. 7", pero los pies de figura correspondientes dicen Fig. 5, 6, 7, 8 y 9.
- El texto dice "Tabla VII" y el encabezado de esa misma tabla dice "TABLA VIII".
- El texto anuncia "Tabla IX, Tabla X y Tabla XI" para los casos de uso, pero las tablas
  están numeradas IV, V y VI.
- La sección de anexo B termina con una "Y" suelta (pág. 26).
- **El índice está desactualizado.** Anuncia la interfaz gráfica en la 18 y las secciones
  VI-B/C/D en la 20, pero en el PDF están en la 21 y la 25 respectivamente. Las páginas
  citadas en este documento son las **reales del PDF**, no las del índice.

---

## 8. Qué hay disponible para reutilizar

Todo bajo `src/modeling/`, con 80 tests en `tests/modeling/`:

| Módulo | Qué resuelve |
|---|---|
| `dataset.py` | Join, partición temporal por episodio, descarte de episodios no-óptimos |
| `canonicalization.py` | Orden canónico de la flota y remapeo de etiquetas |
| `features.py` | Esquema de features y estandarización ajustada sólo con entrenamiento |
| `capacity_decoder.py` | Decodificación factible — **sirve para cualquiera de los cinco modelos**, sólo necesita puntuaciones por vehículo y camión |
| `metrics.py` | Métricas de dominio y concordancia por clase |

`capacity_decoder.py` y `metrics.py` son los más reutilizables: si ustedes producen
probabilidades por vehículo, pueden pasarlas por el mismo decoder y medir con las mismas
métricas, y entonces la comparación entre los cinco modelos sí es una comparación.

Cómo reproducir todo:

```bash
git lfs pull
uv venv --python 3.12 && uv sync
uv run python scripts/build_vehicle_features.py
uv run python scripts/build_scenarios.py
uv run python scripts/train_mlp.py
uv run python scripts/evaluate_mlp.py
uv run python scripts/teacher_self_agreement.py --years 2026
```
