# Resultados preliminares — Modelo MLP

> **Aporte de Juan Francisco Fernández Ramos a la sección VII del reporte.**
> Cubre únicamente el MLP; los resultados de los otros cuatro modelos y su
> consolidación en una tabla comparable siguen sin dueño asignado (ver
> `05_hallazgos_para_el_equipo.md`).
>
> Toda cifra de este documento proviene de `artifacts/mlp/metrics.json`,
> `artifacts/mlp/training_report.json` o `artifacts/mlp/teacher_self_agreement.json`.
> No hay valores estimados ni proyectados.

---

## 1. Conjunto de datos utilizado

Generado con `scripts/build_scenarios.py` sobre los 2.491.511 vehículos en alcance del
conjunto del SRI:

| | |
|---|---:|
| Grupos cantón-semana totales | 55.076 |
| Excluidos por el piso N<5 | 20.237 |
| Episodios construidos y etiquetados | **34.839** |
| Filas vehículo-episodio | **534.680** |
| Episodios donde el etiquetador no probó optimalidad | **0** |
| Episodios triviales (ningún vehículo diferido) | 29.860 (85,7 %) |
| Vehículos diferidos | 22.653 (4,24 %) |

> **Corrección respecto a la sección VI-A.** El reporte estima *"35 mil × 13 = 455 mil
> ejemplos"*. El valor medido es **534.680**. Del mismo modo, la sección VI-A supone que
> `SIN CAMIÓN` está unas 33 veces subrepresentada; el desbalance medido es de **22,6 a 1**
> (4,24 % de las filas).

Partición temporal por episodio completo — entrenamiento 2018-2024 (29.278 episodios,
444.051 filas), validación 2025 (4.030 / 66.399), prueba 2026 (1.531 / 24.230).

---

## 2. Entrenamiento

| | |
|---|---:|
| Parámetros del modelo | 4.482 |
| Épocas ejecutadas (máximo 100) | 56 |
| Mejor época, por pérdida de validación | 46 |
| Tiempo de entrenamiento (CPU) | 150,5 s |
| Pérdida de validación en la mejor época | 0,9237 |
| Reducciones automáticas de la tasa de aprendizaje | 5 (de 1e-3 a 3,9e-6) |

La parada temprana se activó en la época 56 y restauró los pesos de la 46. Curvas en
`artifacts/mlp/learning_curves.png`, historial completo en `training_history.csv`.

### 2.1 Sobre la separación aparente entre las curvas

En la figura de curvas de aprendizaje, la pérdida de entrenamiento (≈0,54) queda muy por
debajo de la de validación (≈0,92). **Esa separación no es sobreajuste**: el entrenamiento
aplica pesos de clase para compensar el desbalance de `SIN CAMIÓN` y la validación no, de
modo que las dos series no están en la misma escala. Evaluando las tres particiones con el
mismo criterio:

| Partición | Pérdida sin pesos | Pérdida con pesos | Exactitud |
|---|---:|---:|---:|
| Entrenamiento | 0,9303 | 0,5320 | 0,5177 |
| Validación | **0,9237** | 0,5298 | 0,5252 |
| Prueba | **0,9085** | 0,5173 | 0,5225 |

Medidas de forma comparable, la pérdida de validación y la de prueba resultan **menores**
que la de entrenamiento, y la exactitud es ligeramente mayor. No hay sobreajuste: con 4.482
parámetros frente a 444.051 filas de entrenamiento, el modelo está limitado por la señal
disponible, no por su capacidad. Las cifras quedan registradas en
`artifacts/mlp/training_report.json` bajo `unweighted_loss`.

---

## 3. Resultados sobre la partición de prueba (2026)

Métricas en orden de relevancia para el dominio. La comparación es contra el **etiquetador
exacto**, que resuelve el mismo problema de forma óptima y sirve de referencia.

| Métrica | MLP + decodificador | Greedy (primer ajuste) | Etiquetador exacto |
|---|---:|---:|---:|
| **1. Tasa de violación de capacidad** | **0,0000** | 0,0000 | 0 |
| **2. Brecha de vehículos cargados** (media) | **+0,0229** | +0,5990 | 0 |
| Brecha máxima en un episodio | 2 | 13 | 0 |
| Episodios que igualan el conteo óptimo | **97,78 %** | 87,98 % | 100 % |
| Brecha de optimalidad relativa | **0,14 %** | 4,17 % | 0 % |
| **3. Brecha de CU aprovechada** (media) | +0,0732 | **+0,0007** | 0 |
| Utilización de la capacidad | 32,73 % | 33,22 % | 33,22 % |
| **4. Vehículos diferidos** | 1.035 | 1.917 | 1.000 |
| **5. F1 macro** | **0,2996** | 0,2421 | — |
| Concordancia por clase | **0,5507** | 0,5252 | — |
| Planes idénticos al óptimo | **42,19 %** | 40,89 % | 100 % |
| **6. Latencia por manifiesto** (media / p99) | 43,3 / 66,6 ms | — | 10,8 ms |
| **7. Exactitud cruda de asignación** | 0,5297 | 0,4928 | — |

**Lo que sí se consiguió.** El plan entregado nunca excede la capacidad de un camión, en
ninguna de las tres particiones y en ningún episodio. Sobre el objetivo primario del
problema —cuántos vehículos se transportan— el modelo iguala la solución óptima en el
97,78 % de los episodios y deja una brecha media de 0,023 vehículos por manifiesto, frente
a 0,599 de la heurística greedy: una reducción de **26 veces**.

**Lo que no se consiguió.** El greedy aprovecha *más* capacidad (brecha de CU +0,0007
frente a +0,0732). No es contradictorio: el greedy carga primero los vehículos voluminosos,
así que transporta menos unidades pero llena más espacio. Bajo el objetivo lexicográfico
del problema —primero cantidad, después aprovechamiento— el modelo gana; bajo el
aprovechamiento aislado, pierde.

---

## 4. Selección de la política del decodificador

Elegida sobre validación (2025) por brecha de conteo, nunca sobre prueba:

| Política | Brecha de conteo | Brecha de CU | Violaciones |
|---|---:|---:|---:|
| **`model`** — orden por margen del modelo | **+0,0303** | +0,0779 | 0,0000 |
| `count` — orden por CU ascendente | +0,0370 | +0,1086 | 0,0000 |
| `respect_defer` — honra el `SIN CAMIÓN` predicho | +0,9883 | +0,8525 | 0,0000 |

El resultado más informativo es el tercero: **honrar el `SIN CAMIÓN` predicho degrada la
brecha de conteo en un factor de 32**. Confirma experimentalmente la decisión de diseño de
que el decodificador no difiera voluntariamente un vehículo que cabe, dado que el objetivo
primario es maximizar la cantidad transportada.

---

## 5. Análisis crítico: dónde aporta el modelo y dónde no

### 5.1 Ablación — ¿aporta el MLP, o basta el decodificador?

Se sustituyeron las puntuaciones del modelo por logits nulos, dejando actuar sólo al
decodificador, sobre los mismos 1.531 episodios de prueba:

| Configuración | Brecha de conteo | Iguala el óptimo | Brecha de CU | Concordancia por clase |
|---|---:|---:|---:|---:|
| MLP + decodificador | **+0,0229** | **97,78 %** | +0,0732 | 0,5507 |
| Logits nulos + decodificador | +0,2371 | 90,66 % | **+0,0267** | 0,5469 |

**Conclusión.** El aporte del MLP es real y está concentrado en el objetivo primario:
reduce la brecha de conteo **diez veces** y sube el porcentaje de episodios óptimos del
90,66 % al 97,78 %. En cambio, **no aporta nada medible a la elección de camión**: la
concordancia por clase es prácticamente la misma con y sin modelo (0,5507 frente a 0,5469).

### 5.2 Matriz de confusión — el modelo colapsa hacia el camión mayor

Cobertura por etiqueta canónica sobre el plan decodificado:

| Etiqueta | Aciertos / total | Cobertura |
|---|---:|---:|
| `SIN CAMIÓN` | 642 / 1.000 | 64,2 % |
| `CAMIÓN 1` (mayor capacidad) | 11.523 / 12.706 | 90,7 % |
| `CAMIÓN 2` | 619 / 6.204 | 10,0 % |
| `CAMIÓN 3` | 51 / 2.999 | 1,7 % |
| `CAMIÓN 4` (menor capacidad) | 0 / 1.321 | **0,0 %** |

La política aprendida es, en la práctica, *«al camión de mayor capacidad mientras quepa;
diferir cuando no quepa en ninguno»*. Eso basta para el objetivo de conteo —que no depende
de en qué camión concreto viaja cada vehículo— pero significa que el modelo **no reprodujo
el reparto del etiquetador entre camiones**, y explica tanto la brecha de CU como el F1
macro bajo.

Este colapso resultó **no ser una limitación del modelo**: desaparece al fijar el orden de
la flota en el etiquetador, sin tocar la arquitectura (§5.5).

### 5.3 El modelo es insensible a los hiper-parámetros dentro del ruido

Ocho configuraciones entrenadas y evaluadas (`artifacts/mlp/sweep/summary.json`) producen
brechas de conteo entre +0,0275 y +0,0315 sobre validación. Contrastando la configuración
adoptada contra la mejor del ranking sobre los mismos 4.030 episodios:

| | |
|---|---:|
| Diferencia emparejada | +0,0027 |
| Error estándar | 0,0018 |
| Estadístico *t* | 1,51 (no significativo al 95 %) |
| Episodios en que ambos modelos difieren | **47 de 4.030** |

Duplicar los parámetros (14.018 y 16.130 frente a 4.482) no mejora nada: ambas variantes
anchas quedan en la mitad inferior. El factor limitante no es la capacidad del modelo ni la
elección de hiper-parámetros, sino la señal disponible en la etiqueta — que es lo que mide
la sección siguiente.

### 5.4 ¿Cuánta de la etiqueta era predecible?

Antes de atribuir al modelo la baja exactitud, se midió cuánta información contiene
realmente la etiqueta. Se presentó al etiquetador exacto **la misma flota permutada** —los
mismos camiones, las mismas capacidades, una situación operativamente idéntica— y se
comparó su nueva respuesta con la original, ambas canonicalizadas
(`scripts/teacher_self_agreement.py`, 1.158 episodios de 2026 con dos o más camiones):

| | |
|---|---:|
| Exactitud que el etiquetador reproduce de sí mismo | **0,3983** |
| Concordancia por clase consigo mismo | **0,4493** |
| Episodios que reproduce de forma idéntica | 35,58 % |
| \|Δ vehículos cargados\| | **0,0000** |
| \|Δ CU aprovechada\| | **0,0000** |

El oráculo que generó las etiquetas reproduce menos del 40 % de ellas, mientras el
resultado operativo es exactamente el mismo. Esa discrepancia tiene dos orígenes: el
barajado sembrado con que el etiquetador reparte los cupos dentro de una clase, y el hecho
de que su programación dinámica llena primero el camión de índice 0, que es de capacidad
aleatoria.

Esa cifra, sin embargo, **no es un techo**: predecir la moda de una variable aleatoria
coincide con una muestra más a menudo de lo que dos muestras independientes coinciden entre
sí. El techo hay que calcularlo aparte, y es lo que hace la sección siguiente.

### 5.5 El techo exacto, y qué parte de la brecha es recuperable

Dos vehículos de la misma clase en el mismo episodio tienen features idénticas: misma `cu`,
mismo one-hot, mismo `n_misma_clase`, mismo contexto. Ninguna entrada los distingue, así que
la exactitud alcanzable está acotada en forma cerrada. Con `n(c,t)` = vehículos de la clase
`c` que el etiquetador mandó al destino `t`, y `m_c = Σ_t n(c,t)`
(`src/modeling/metrics.py:label_ceilings`):

```
Cota A  (modelo determinista por vehículo)  =  Σ_c max_t n(c,t)      / N
Cota B  (pipeline con decodificador)        =  Σ_c Σ_t n(c,t)² / m_c / N
```

| Partición de prueba (2026) | |
|---|---:|
| Cota A — techo de un clasificador determinista | **0,9243** |
| Cota B — techo del pipeline con decodificador | **0,9005** |
| Exactitud medida del modelo | 0,5297 |
| Fracción del techo alcanzada | **58,8 %** |

Dos comprobaciones de que la cifra es correcta: es **invariante a la canonicalización**
(0,9084 global con y sin canonicalizar, idéntico — permutar el eje de destinos no altera
`max_t` ni `Σ_t n²`), y **cierra contra una medición independiente**: con la flota fijada,
la auto-concordancia del etiquetador debe igualar la cota B, y da 0,9309 analítica frente a
0,9306 medida sobre los mismos 1.158 episodios.

La conclusión corrige la lectura natural de la sección 5.4: el ruido irreducible vale unos
**8 puntos**, no 60. Los otros ~37 puntos de brecha vienen de que el orden en que llega la
flota cambia el *plan* del etiquetador, y ese orden no es una entrada del modelo. Es una
propiedad del generador de datos, no del clasificador ni de las etiquetas en sí: fijando el
orden de la flota antes de etiquetar —una línea en `src/loading/scenarios.py`— el mismo
modelo, con los mismos hiper-parámetros y la misma semilla, pasa a **0,8458** de exactitud,
**0,8131** de F1 macro y **0,872** de cobertura en `CAMIÓN 4`, sin que las métricas
operativas se muevan. La medición completa está en
[`06_canonicalizacion_y_etiquetado.md`](06_canonicalizacion_y_etiquetado.md); el cambio no
se aplicó porque invalida los modelos ya entrenados por el resto del equipo.

---

## 6. Generalización a flotas mayores que las vistas

Los datos de entrenamiento contienen entre uno y cuatro camiones. Como el perceptrón se
comparte entre todos los pares, **los mismos pesos** pueden puntuar manifiestos con más
camiones sin reconstruir ni reentrenar la red. Para comprobarlo se tomaron los mismos
manifiestos de prueba, se les cambió únicamente la flota y se reetiquetaron con el
etiquetador exacto (`scripts/build_extrapolation_set.py`):

| Conjunto | Camiones | Violaciones | Brecha de conteo | Iguala el óptimo |
|---|---|---:|---:|---:|
| Prueba 2026 | 1–4 (visto en entrenamiento) | 0,0000 | +0,0229 | 97,78 % |
| Extrapolación | 5–6 | 0,0000 | +0,0000 | 100,00 % |
| Extrapolación | 8–10 | 0,0000 | +0,0000 | 100,00 % |
| Extrapolación, capacidad total constante | 8–10 | 0,0000 | +0,0137 | 98,95 % |

**Alcance de esta evidencia.** Queda demostrado que el modelo guardado **acepta y resuelve**
manifiestos con hasta diez camiones usando los mismos pesos, sin violar capacidad. No queda
demostrada calidad a esa escala: al añadir camiones manteniendo el rango de capacidades del
entrenamiento, la capacidad total crece y los episodios se vuelven triviales —ningún
vehículo diferido—, de modo que el 100 % de coincidencia también lo alcanza la heurística
greedy. El conjunto de capacidad total constante es el único exigente de los tres, y ahí la
ventaja sobre el greedy desaparece (+0,0137 frente a +0,0157).

---

## 7. Conclusiones

1. **La restricción de capacidad se cumple siempre.** Cero violaciones en 34.839 episodios,
   por construcción del decodificador y no por acierto del clasificador.
2. **Sobre el objetivo primario el modelo es claramente superior a la heurística:** brecha
   de optimalidad del 0,14 % frente al 4,17 % del greedy.
3. **El aporte del MLP está en decidir qué se carga y en qué orden, no en qué camión.**
   La ablación con logits nulos lo cuantifica: sin el modelo, la brecha de conteo se
   multiplica por diez; la concordancia por clase, en cambio, no cambia.
4. **La arquitectura por pares cumple el requisito de flota sin límite codificado**, y se
   verificó ejecutando el modelo entrenado sobre manifiestos de diez camiones.
5. **La baja exactitud es una propiedad del generador de datos, no del modelo — y es
   recuperable.** El techo exacto sobre estas etiquetas es 0,9243 y el modelo alcanza el
   58,8 % de él. La brecha se explica: el orden aleatorio de la flota cambia el plan del
   etiquetador y no es una entrada observable. Fijando ese orden, el mismo modelo llega a
   0,8458 (97,7 % de su techo). Sólo unos 8 puntos son ruido irreducible.
6. **El techo no está en los hiper-parámetros.** Ocho configuraciones caen dentro de
   ±0,004 y la diferencia entre la mejor y la adoptada no es significativa (t = 1,51);
   triplicar los parámetros empeora el resultado.
7. **La latencia (43 ms por manifiesto) es holgada** frente al requisito operativo, aunque
   está dominada por la sobrecarga de una llamada de inferencia por manifiesto y no por el
   cómputo del modelo, que tiene 4.482 parámetros.

## 8. Trabajo pendiente identificado

- **Cerrar la brecha de CU.** El greedy aprovecha más capacidad. Una función de pérdida que
  penalice el desperdicio, o un decodificador que reordene por CU cuando la capacidad
  escasea, son las dos vías inmediatas.
- **Recuperar la discriminación entre camiones — ya está diagnosticado y medido.** La
  cobertura nula de `CAMIÓN 4` no es un límite del modelo: ordenando la flota antes de
  etiquetar sube a 0,872 y el F1 macro a 0,8131 (§5.5). Falta la decisión del equipo de
  regenerar el dataset, porque obliga a reentrenar los cinco modelos.
- **Construir conjuntos de extrapolación exigentes**, con déficit de capacidad forzado, para
  que la prueba de flotas grandes discrimine.
- **Ablación de `canton`**, hoy excluido por no participar en la restricción de capacidad.
