# Sesión — Diseño, entrenamiento y evaluación del MLP clasificador

**Fecha:** 25 de julio de 2026
**Autor:** Juan Francisco Fernández Ramos
**Herramientas:** Claude Opus 5 (Claude Code, sobre el repositorio) y Claude Chat
**Alcance:** entregables de la Tarea #4 asignados a Juan Francisco — escenarios del caso
de uso de análisis, arquitectura e hiper-parámetros del MLP, entrenamiento con Keras 3.

> **Nota sobre el formato.** Este es un registro editado de la sesión, no un volcado
> literal: se conservan las preguntas tal como se formularon y la sustancia de las
> respuestas, y se omiten las iteraciones de depuración de código y los mensajes de
> herramienta. Las decisiones y las cifras son las reales; los artefactos citados existen
> en `artifacts/mlp/`.

---

## Pregunta 1 — ¿Qué debo hacer primero, y en qué orden?

**Contexto entregado:** los tres PDF de la tarea (órdenes, reporte, planificación) y el
repositorio completo.

**Respuesta obtenida (resumen).** Se identificaron cuatro entregables propios a partir de
las páginas 4, 6, 7 y 8 de la planificación: dos escenarios para la etapa de análisis
(Tabla V, hoy con marcadores `1.`, `2a.`, `a`), la definición y justificación de
hiper-parámetros del MLP (sección VI-D, vacía), el entrenamiento con Keras 3 (inexistente
en el repositorio) y el aporte al anexo de IA.

Se recomendó además atacar los problemas de datos **antes** de escribir código de
entrenamiento, con tres hipótesis concretas: que las filas de vehículos no contienen las
capacidades de la flota, que el etiquetador reparte los cupos dentro de una clase de forma
arbitraria, y que las etiquetas de camión son sensibles a permutación.

**Qué se hizo.** Se verificaron las tres hipótesis en el código fuente. Las tres
resultaron ciertas:

- `episode_vehicles.parquet` no tiene `n_trucks` ni `truck_capacities`; están sólo en
  `episodes.parquet`.
- `labeler.py:221-229` baraja los vehículos de cada clase con una semilla antes de
  repartir los cupos.
- `labeler.py:188-213` recorre los camiones por índice y llena el primero tanto como
  puede; ese índice 0 es un camión de capacidad aleatoria.

Se encontraron además cuatro bloqueadores que no estaban en la recomendación: los CSV de
`data/clean` son punteros Git LFS de 133 bytes y `git lfs` no estaba instalado; el
intérprete del sistema es Python 3.14 y TensorFlow 2.21 sólo publica ruedas hasta 3.13;
no había ningún paquete instalado ni entorno virtual; y `data/features` y `data/episodes`
no existían.

---

## Pregunta 2 — ¿Slots fijos o puntuación por par vehículo–camión?

> El diagrama de secuencia de la planificación dice explícitamente "SIN LÍMITE de
> camiones" y el reporte (Fig. 6) dice CAMIÓN 1..n. Pero el generador usa
> `N_TRUCKS_RANGE = (1, 4)`. ¿Qué arquitectura de MLP implemento?

**Respuesta obtenida (resumen).** Se recomendó la puntuación por par. Una capa `Dense(5)`
deja la cantidad de camiones escrita en la arquitectura: con cinco camiones haría falta
`Dense(6)` y volver a entrenar. Un perceptrón compartido aplicado a cada par
(vehículo, camión), más una cabeza separada para `SIN CAMIÓN`, mantiene el número de
parámetros independiente de *n*.

Se precisó también que «sin límite» no puede significar infinitos camiones: significa que
la cantidad no debe estar codificada en la capa de salida.

**Decisión.** Se adoptó la puntuación por par. La afirmación se convirtió en prueba:
`test_los_mismos_pesos_atienden_2_y_50_camiones` verifica que el modelo produce salidas
de longitud correcta para *n* = 1, 2, 4, 10 y 50 sin reconstruir nada, y
`test_el_scorer_es_equivariante_a_la_permutacion_de_camiones` verifica que reordenar los
camiones reordena las puntuaciones de la misma forma.

---

## Pregunta 3 — Revisión crítica de una guía de implementación externa

Se entregó un documento de recomendaciones de terceros con código Keras completo, y se
pidió evaluarlo en lugar de seguirlo.

**Qué se aceptó:** compartición de parámetros, cabeza separada para `SIN CAMIÓN`,
enmascarado del relleno con −1e9, decodificador con restricción de capacidad,
canonicalización de camiones por capacidad descendente, partición por episodio, exclusión
de la posición del vehículo dentro de su clase y del identificador del camión.

**Qué se rechazó o corrigió, y por qué:**

| Recomendación | Motivo del rechazo | Evidencia |
|---|---|---|
| Usar OR-Tools / CP-SAT como oráculo exacto | El repositorio prohíbe solvers externos por decisión de alcance, y el maestro exacto ya existe | `src/loading/labeler.py:6-9`, `reports/.../02_scope.md` |
| `keras.Model` subclasificado sin `get_config()` | No se puede recargar desde `.keras`. Reproducido con Keras 3.15: sin `custom_objects` da `TypeError: Could not locate class`; con ellos da `ValueError: Layer 'dense_3' expected 2 variables, but received 0`, porque reconstruye con los valores por defecto del constructor | Fijado en `test_guardar_y_recargar_produce_logits_identicos` |
| `SIN CAMIÓN` en el último índice del lote | El índice objetivo cambia según el manifiesto más grande del lote | Se movió al índice 0 |
| Partición de entrenamiento 2017–2024 | 2017 no existe: ese CSV no trae `FECHA PROCESO` y el pipeline lo descarta | `reports/.../08_feature_coverage.md`, "Skipped years" |
| Umbral que difiere antes de intentar ubicar | El objetivo es lexicográfico, primero conteo; diferir un vehículo que cabe sólo puede empeorar | Medido: brecha de conteo +0,0303 → +0,9883 |

---

## Pregunta 4 — ¿Por qué la exactitud se queda en 0,53?

Tras entrenar sobre el dataset completo, la exactitud cruda quedó en 0,5297 y la
concordancia por clase en 0,5507, apenas por encima de la línea base greedy (0,4928 y
0,5252). La pregunta fue si el modelo estaba mal o la etiqueta no era predecible.

**Diseño del experimento (aporte propio).** Presentar al etiquetador exacto **la misma
flota permutada** —mismos camiones, mismas capacidades, situación operativamente
idéntica— y comparar su nueva respuesta con la original, ambas canonicalizadas.
Implementado en `scripts/teacher_self_agreement.py`.

**Resultado sobre los 1.158 episodios de 2026 con dos o más camiones:**

```
TECHO EMPÍRICO -- el maestro contra sí mismo, misma flota en otro orden:
  Exactitud cruda reproducida        0.3983
  Concordancia por clase             0.4493
  Episodios reproducidos idénticos   35.58%

CONTROL -- lo que sí es determinista (el objetivo real):
  |Δ vehículos cargados| medio       0.0000
  |Δ CU aprovechada| medio           0.0000
```

**Interpretación.** El oráculo que generó las etiquetas reproduce menos del 40 % de ellas,
mientras el resultado operativo es idéntico. Cerca del 60 % de la etiqueta es ruido de
desempate. La exactitud del modelo (0,5297) queda por encima de esa autoconsistencia.

Precisión necesaria sobre el alcance: la autoconsistencia del maestro **no** es un techo
estricto, porque predecir la moda de una variable aleatoria coincide con una muestra más a
menudo de lo que dos muestras independientes coinciden entre sí. Lo que sí cuantifica es
cuánta de la etiqueta es arbitraria, y justifica reportar la exactitud como diagnóstico
secundario.

---

## Pregunta 5 — ¿La canonicalización sirve de algo, medible?

Se comparó el reparto de etiquetas antes y después de reordenar la flota por capacidad,
sobre la misma partición de entrenamiento (444.051 filas):

| Etiqueta | Orden original | Orden canónico |
|---|---:|---:|
| `SIN CAMIÓN` | 4,26 % | 4,26 % |
| `CAMIÓN 1` | 78,41 % | 52,29 % |
| `CAMIÓN 2` | 14,38 % | 25,33 % |
| `CAMIÓN 3` | 2,70 % | 13,16 % |
| `CAMIÓN 4` | 0,26 % | 4,97 % |

`CAMIÓN 4` pasa de estar prácticamente ausente a tener representación medible. La
canonicalización no cambia ningún plan —sólo el nombre del camión—, pero convierte la
etiqueta en función de una característica que el modelo observa.

---

## Resultados finales

Dataset completo: 34.839 episodios, 534.680 filas, 0 episodios no-óptimos.
Modelo: 4.482 parámetros, 56 épocas, 150 s.

| Métrica (partición de prueba, 2026) | MLP | Greedy | Maestro |
|---|---:|---:|---:|
| Tasa de violación de capacidad | **0,0000** | 0,0000 | — |
| Brecha de vehículos cargados | **+0,0229** | +0,5990 | 0 |
| Episodios que igualan el conteo óptimo | **97,78 %** | 87,98 % | 100 % |
| Brecha de CU aprovechada | +0,0732 | **+0,0007** | 0 |
| Concordancia por clase | **0,5507** | 0,5252 | — |
| Exactitud cruda | **0,5297** | 0,4928 | — |
| Latencia por manifiesto | 43,35 ms | — | 10,8 ms |

El modelo gana con claridad en el objetivo primario y **pierde** frente al greedy en el
secundario: el greedy carga menos vehículos pero más voluminosos, así que aprovecha más CU.
Queda declarado como limitación en la sección VII.

Extrapolación con los mismos pesos, sin reentrenar (conjuntos generados por
`scripts/build_extrapolation_set.py`):

| Conjunto | Camiones | Violaciones | Brecha de conteo |
|---|---|---:|---:|
| Prueba 2026 | 1–4 (visto) | 0,0000 | +0,0229 |
| Extrapolación | 5–6 | 0,0000 | +0,0000 |
| Extrapolación | 8–10 | 0,0000 | +0,0000 |
| Extrapolación, capacidad total constante | 8–10 | 0,0000 | +0,0137 |

Los dos primeros conjuntos de extrapolación resultaron poco exigentes (ningún vehículo
diferido), así que la evidencia sostiene factibilidad y ausencia de violaciones con
flotas mayores que las vistas, no calidad demostrada a esa escala.
