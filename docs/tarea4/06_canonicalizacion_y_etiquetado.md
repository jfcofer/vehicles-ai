# Canonicalización y calidad del etiquetado

> **Documento interno. No va al reporte** (salvo las secciones 5 y 8, que sí deberían
> entrar en VII y VIII).
> Autor: Juan Francisco Fernández Ramos · 27 de julio de 2026
> Para: Víctor Borbor, Nicolás Fiallo
>
> Complementa y **corrige en un punto importante** a
> [`05_hallazgos_para_el_equipo.md`](05_hallazgos_para_el_equipo.md).

---

## 0. Resumen en diez líneas

Les iba a pasar `src/modeling/canonicalization.py` diciendo que la baja exactitud era ruido
irreducible de la etiqueta. **Lo medí a fondo y esa conclusión estaba mal.**

- El ruido verdaderamente irreducible es de **~8 puntos**, no de ~60. El techo exacto de
  exactitud cruda es **0,9243**, no 0,3983.
- Los otros ~40 puntos vienen de que `generate_fleet()` devuelve las capacidades **en orden
  aleatorio**, y eso cambia el *plan* que produce el maestro, no sólo el *nombre* del camión.
- La canonicalización aguas abajo arregla el nombre. **No arregla el plan.**
- Lo que sí lo arregla es **una línea** en `scenarios.py`: ordenar la flota antes de
  etiquetar. Lo medí de punta a punta.

| Métrica (prueba 2026) | Hoy | Con la flota ordenada |
|---|---:|---:|
| Exactitud cruda | 0,5297 | **0,8458** |
| F1 macro | 0,2996 | **0,8131** |
| Concordancia por clase | 0,5507 | **0,9293** |
| *Recall* de `CAMION_4` | 0,000 | **0,872** |
| Brecha de vehículos cargados | +0,0229 | +0,0242 |
| Violaciones de capacidad | 0,0000 | 0,0000 |

Cuesta **8 minutos** de regeneración e **invalida los cinco modelos ya entrenados**.
Por eso **no apliqué el parche**: la decisión es del grupo. Todo lo de abajo es la
evidencia para tomarla.

---

## 1. Qué es exactamente la etiqueta y cómo se fabrica

Para cada vehículo, `episode_vehicles.parquet` guarda una columna `truck` con valores
`SIN_CAMION` o `CAMION_k`. Se produce en tres pasos:

```
scenarios.py:106   sampled  = stratified_subsample(group, MAX_N, rng)
scenarios.py:107   fleet    = generate_fleet(rng)          <-- capacidades SIN ORDENAR
scenarios.py:111   result   = assign_vehicles(vehicles, fleet, seed=labeler_seed)
```

Y dentro de `assign_vehicles` (`labeler.py`):

```
labeler.py:143     classes  = sorted(by_class.keys())        <-- orden alfabético
labeler.py:188-213 solve()   <-- PD exacta; recorre camiones POR ÍNDICE
labeler.py:221-229 queues    <-- random.shuffle sembrado dentro de cada clase
```

---

## 2. Hay cuatro arbitrariedades, no una

Esto es lo que yo mismo no tenía separado, y por eso la conclusión anterior salió mal.

| | Qué es arbitrario | Dónde nace | ¿Lo ve el modelo? | ¿Lo arregla `canonicalization.py`? |
|---|---|---|---|---|
| **S1** | El **nombre**: `CAMION_1` es el que salió primero del generador | `scenarios.py:82-84` | No | **Sí, del todo** |
| **S2** | El **plan**: la PD llena el camión de índice 0 primero, y ese índice es de capacidad aleatoria | `labeler.py:188-213` | No | **No** |
| **S3** | **Cuál** vehículo concreto de una clase recibe el cupo | `labeler.py:221-229` | No — dos filas de la misma clase son idénticas | **No** |
| **S4** | Se llena prefiriendo las clases alfabéticamente primeras | `labeler.py:143`, `199` | **Sí** (one-hot de clase) | No hace falta: es determinista y aprendible |

### El ejemplo que hace entender S2 en treinta segundos

3 motocicletas de 0,5 CU y 2 automóviles de 2,0 CU. Flota `{6,0 · 4,0}`.

| Orden en que llega la flota | Qué hace la PD | Plan resultante |
|---|---|---|
| `[6,0 · 4,0]` | El índice 0 es el grande: se lo lleva todo (2 autos + 3 motos = 5,5 ≤ 6,0) | Todo en el camión de 6,0 |
| `[4,0 · 6,0]` | El índice 0 es el chico: se lleva los 2 autos (4,0 exactos) | Autos en el de 4,0, motos en el de 6,0 |

Los dos planes cargan **5 vehículos** y aprovechan **5,5 CU**: son *igual de óptimos*, y el
maestro no tiene ninguna razón para preferir uno. Pero **no existe ningún renombramiento
que lleve un plan al otro**: en uno los autos van con las motos y en el otro no. Es una
diferencia de *plan*, no de *nombre*, y por eso canonicalizar la salida no la reconcilia.

Eso explica exactamente lo que ya estaba medido y que no supimos leer:
`|Δ vehículos cargados| = 0,0000`, `|Δ CU| = 0,0000`, y aun así concordancia 0,3983.
El **resultado** es determinista; la **etiqueta** no.

---

## 3. Qué hace la canonicalización, y qué NO hace

`canonicalize_fleet()` ordena los camiones por capacidad descendente y remapea las
etiquetas. `CAMION_1` pasa a ser siempre el de mayor capacidad.

**Lo que sí consigue** — reparto de etiquetas en la partición de entrenamiento (444.051 filas):

| Etiqueta | Orden original | Canonicalizado |
|---|---:|---:|
| `SIN_CAMION` | 4,26 % | 4,26 % |
| `CAMION_1` | 78,41 % | 52,29 % |
| `CAMION_2` | 14,38 % | 25,33 % |
| `CAMION_3` | 2,70 % | 13,16 % |
| `CAMION_4` | **0,26 %** | **4,97 %** |

`CAMION_4` pasa de estar prácticamente ausente a tener representación medible. Con 0,26 %
de soporte ningún clasificador va a aprender esa clase; con 4,97 % es al menos posible.
Y como todos usamos la misma convención, los cinco modelos quedan comparables entre sí.

> ### Lo que NO consigue
>
> **La canonicalización sola no sube la exactitud.** Elimina S1 y nada más. Si la aplican
> esperando que el número salte, no va a saltar, y van a descartar la herramienta correcta
> por la razón equivocada. Sirve para (a) desbloquear `CAMION_3` y `CAMION_4`, y (b) que
> los cinco modelos midan contra la misma convención. El salto grande está en la sección 6.

---

## 4. Cómo lo integra cada uno de los cinco modelos

`canonicalization.py` es Python puro de biblioteca estándar: no arrastra numpy, keras ni el
resto de `src/modeling/`. Son tres líneas:

```python
from src.modeling.canonicalization import canonicalize_fleet

fleet = canonicalize_fleet(row["truck_capacities"])
y     = fleet.label_map[row["truck"]]   # etiqueta canónica: "CAMION_2" -> "CAMION_1", etc.
caps  = fleet.capacities                # capacidades en el MISMO orden que la etiqueta
```

### Dos errores que hay que evitar

**1. Canonicalizar la etiqueta y no las features de capacidad deja el dataset peor que
antes.** Si su modelo tiene columnas `cap_1..cap_4` en el orden original del parquet y la
etiqueta canonicalizada, la correspondencia entre ambas se rompe: le están diciendo al
modelo "este vehículo va al `CAMION_1`" mientras `cap_1` describe un camión distinto.
Van juntas o no va ninguna. Por eso `canonicalize_fleet` devuelve `capacities` en el mismo
objeto — usen el objeto entero y el error es imposible.

**2. Para la salida al operador hay que revertir.** El operador conoce sus camiones por su
nombre real, no por su puesto en el ranking de capacidad:

```python
etiqueta_operador = fleet.inverse_label_map[etiqueta_predicha]
```

---

## 5. El techo de exactitud, calculado exacto

En `05_hallazgos` escribí que el maestro sólo reproduce el 39,83 % de sus propias etiquetas
y que "el ~60 % restante es ruido". **Esa lectura era incorrecta**, y yo mismo había dejado
anotada la salvedad: la autoconsistencia no es un techo, porque predecir la moda de una
variable aleatoria coincide con una muestra más a menudo de lo que dos muestras
independientes coinciden entre sí.

Ahora está calculado en forma cerrada, sin entrenar nada
(`src/modeling/metrics.py:label_ceilings`, `scripts/label_ceiling.py`).

Dos vehículos de la misma clase en el mismo episodio tienen **features idénticas**: misma
`cu`, mismo one-hot, mismo `n_misma_clase`, mismo contexto. Ninguna entrada los distingue.
Con `n(c,t)` = vehículos de la clase `c` que van al destino `t`, y `m_c = Σ_t n(c,t)`:

```
Cota A  (modelo determinista por vehículo)  =  Σ_c max_t n(c,t)      / N
Cota B  (pipeline con decodificador)        =  Σ_c Σ_t n(c,t)² / m_c / N
```

| Sobre el dataset actual | Global | Prueba 2026 |
|---|---:|---:|
| Cota A | 0,9084 | **0,9243** |
| Cota B | 0,8793 | **0,9005** |
| Exactitud medida del MLP | — | 0,5297 |

**El techo real es 0,92, no 0,40.** El MLP está capturando apenas el 59 % de lo alcanzable.
El ruido verdaderamente irreducible — el de S3, el `shuffle` intra-clase — vale unos
**8 puntos**, no 60.

Dos propiedades que dan confianza en la cifra:

- **Es invariante a la canonicalización.** Renombrar camiones permuta el eje `t`, y ni
  `max_t` ni `Σ_t n²` cambian con una permutación. Medido: 0,9084 con y sin canonicalizar,
  idéntico. Es la demostración formal de que canonicalizar **no puede** subir el techo.
- **Cierra contra una medición independiente.** Con la flota fijada, la auto-concordancia
  del maestro debe coincidir con la cota B. Sobre los mismos 1.158 episodios:
  **0,9309 analítica vs 0,9306 medida.**

---

## 6. El arreglo de verdad: ordenar la flota antes de etiquetar

### Por qué funciona

`assign_vehicles()` recibe `truck_capacities`. Si esa lista llega **siempre ordenada**,
cualquier permutación de la misma flota produce la misma entrada a la PD y por tanto la
misma salida. La invariancia deja de ser algo que se mide y pasa a ser cierta **por
construcción**. Comprobado con `scripts/teacher_self_agreement.py --fleet-order`:

| Orden de la flota | Concordancia por clase | Episodios idénticos | Exactitud reproducida |
|---|---:|---:|---:|
| `as-is` (hoy) | 0,4493 | 35,58 % | 0,3983 |
| `desc` | **1,0000** | **100,00 %** | 0,9306 |
| `asc` | **1,0000** | **100,00 %** | 0,8704 |

S1 y S2 desaparecen del todo. Lo que queda por debajo de 1 en la última columna es
exactamente S3, el ruido irreducible.

### Por qué es más barato de lo que parece

Ordenar **no consume aleatoriedad**, así que el flujo del RNG del episodio no cambia: se
submuestrean los mismos vehículos, se sortean las mismas capacidades y el etiquetador
recibe la misma semilla. Regeneré el dataset completo y comparé columna por columna
(34.839 episodios, 534.680 filas, 8 min):

| Columna | ¿Cambia? |
|---|---|
| `episode_id`, `iso_year`, `iso_week`, `canton`, `n_original`, `n_sampled`, `n_trucks`, `seed` | **Idénticas** |
| `n_loaded`, `n_deferred`, `cu_utilized`, `optimal` | **Idénticas en los 34.839 episodios** |
| `truck_capacities` | Mismo multiconjunto, distinto orden |
| `truck` | Difiere en 43.687 de 534.680 filas (8,2 %) |
| `loaded` | Difiere en **16** filas (0,003 %) |
| `search_time_ms`, `nodes_explored` | Difieren — reloj de pared y orden de búsqueda |

Que `n_loaded` y `cu_utilized` salgan idénticos en los 34.839 episodios es la comprobación
que importa: **la PD es exacta, así que permutar camiones nunca cambia el óptimo, sólo cuál
de los óptimos empatados se devuelve.** Las 16 filas de `loaded` son empates al nivel de la
mezcla de clases, no un cambio de calidad.

### Lo que se gana, medido de punta a punta

Mismo modelo, mismos hiper-parámetros, misma semilla, misma partición. Sólo cambia el orden
de la flota al etiquetar. Partición de prueba (2026):

| | Hoy (`as-is`) | `desc` | `asc` |
|---|---:|---:|---:|
| **Exactitud cruda** | 0,5297 | **0,9051** | 0,8458 |
| **F1 macro** | 0,2996 | 0,5387 | **0,8131** |
| **Concordancia por clase** | 0,5507 | **0,9531** | 0,9293 |
| Techo (cota B) del propio dataset | 0,9005 | 0,9229 | 0,8659 |
| **% del techo alcanzado** | **58,8 %** | 98,1 % | 97,7 % |
| Brecha de vehículos cargados | +0,0229 | +0,0255 | +0,0242 |
| Episodios que igualan el óptimo | 97,78 % | 97,52 % | 97,58 % |
| Violaciones de capacidad | 0,0000 | 0,0000 | 0,0000 |

*Recall* por camión — aquí se ve el colapso actual y su desaparición:

| | `SIN_CAMION` | `CAMION_1` | `CAMION_2` | `CAMION_3` | `CAMION_4` |
|---|---:|---:|---:|---:|---:|
| Hoy | 0,642 | 0,907 | 0,100 | 0,017 | **0,000** |
| `desc` | 0,650 | 0,952 | 0,594 | 0,450 | 0,118 |
| `asc` | 0,647 | 0,901 | 0,812 | 0,829 | **0,872** |

Las métricas operativas **no se mueven** (la brecha de conteo pasa de +0,0229 a +0,0242,
dentro del ruido). Lo que cambia es que la etiqueta se vuelve aprendible.

### Cuál de los dos órdenes: recomiendo `asc`

`desc` da mejor exactitud cruda, pero por una razón que conviene mirar de cerca: concentra
el 83,93 % de las etiquetas en `CAMION_1`, así que **acertar mucho es fácil**. La prueba es
la línea base greedy, que no es un modelo entrenado:

| | Exactitud del MLP | Exactitud del greedy | Ganancia real sobre lo trivial |
|---|---:|---:|---:|
| Hoy | 0,5297 | 0,4928 | +0,0369 |
| `desc` | 0,9051 | 0,8559 | **+0,0492** |
| `asc` | 0,8458 | 0,2683 | **+0,5775** |

Con `desc`, el maestro llena siempre el camión grande primero — que es literalmente lo que
hace el greedy —, así que los cinco modelos del grupo aterrizarían todos cerca del 0,86 de
una heurística de tres líneas y **la comparación del reporte no distinguiría nada**. Con
`asc` el objetivo exige de verdad razonar sobre la flota, el F1 macro sube a 0,8131 y los
cuatro camiones son aprendibles.

**Contrapartida honesta de `asc`:** usa más camiones por episodio (1,504 frente a 1,266 de
`desc`; hoy 1,375). El objetivo declarado en la Fig. 1 no penaliza usar camiones de más, así
que los dos son igual de óptimos *según el objetivo*, pero un operador real preferiría
`desc`. Si al grupo le importa esa dimensión, hay que decirlo en el reporte — o añadir el
número de camiones como tercer criterio del objetivo, que sería un cambio mayor.

### El parche, sin aplicar

```diff
 def generate_fleet(rng: random.Random) -> list[float]:
     n_trucks = rng.randint(*N_TRUCKS_RANGE)
-    return [round(rng.uniform(*CAP_RANGE), 1) for _ in range(n_trucks)]
+    caps = [round(rng.uniform(*CAP_RANGE), 1) for _ in range(n_trucks)]
+    # Orden canónico ANTES de etiquetar: hace la salida del maestro invariante a
+    # permutaciones de la flota por construcción, en vez de por casualidad. No
+    # consume aleatoriedad, así que el submuestreo y la semilla del etiquetador
+    # no cambian -- ver docs/tarea4/06_canonicalizacion_y_etiquetado.md.
+    return sorted(caps)
```

Después: `uv run python scripts/build_scenarios.py` (~8 min) y reentrenar los cinco modelos.

La canonicalización aguas abajo **se queda igual y sigue haciendo falta**: son dos cosas
distintas. El orden de arriba fija el *plan*; `canonicalize_fleet` fija el *nombre*. Con el
parche aplicado, la canonicalización pasa a ser un renombrado idempotente que ya no puede
salir mal. Los números de la tabla de arriba se midieron con esa combinación exacta:
`asc` aguas arriba + `canonicalize_fleet` (descendente) aguas abajo.

Reproducible sin tocar el repositorio:

```bash
uv run python artifacts/mlp/fleet_order_experiment/build_sorted_episodes.py \
    --order asc --out /tmp/episodes_asc
uv run python scripts/train_mlp.py    --episodes-dir /tmp/episodes_asc --out-dir /tmp/mlp_asc
uv run python scripts/evaluate_mlp.py --model-dir /tmp/mlp_asc --episodes-dir /tmp/episodes_asc
```

---

## 7. El menú completo, con costos

| | Opción | Elimina | Costo | ¿Invalida lo entrenado? | Recomendación |
|---|---|---|---|---|---|
| **A** | Canonicalización aguas abajo (ya existe, con pruebas) | S1 | ~10 min por modelo | No | **Hacerla** |
| **B** | Ordenar la flota en `scenarios.py` (1 línea) | S1 + **S2** | 8 min + reentrenar los 5 | **Sí** | **Decisión del grupo.** Medida, no aplicada |
| **C** | Desempate intra-clase determinista por `uid` | nada útil | — | Sí | **Rechazar** (ver abajo) |
| **D** | Cambiar el objetivo a conteos por clase | S3 | Rediseño de los 5 modelos | Sí | Trabajo futuro |
| **E** | No tocar el etiquetador; corregir la métrica | — | 0 | No | **Hacerla**, junto con A |

### Por qué rechazo C, aunque parezca la solución obvia a S3

Tentación: quitar el `random.shuffle` y repartir los cupos por `uid` ordenado, para que el
resultado sea determinista. **Es peor que el problema.** El propio `labeler.py:106-116` lo
advierte: con orden fijo por `uid`, *los mismos vehículos quedan siempre excluidos* cada vez
que se repite una mezcla de clases parecida. El modelo aprendería ese patrón, que no
significa nada, y la exactitud de entrenamiento subiría sin que mejore ni un plan.

El `shuffle` sembrado **no es un defecto, es la decisión correcta**: hace que el objetivo
sea *insesgado*. Para "¿se carga esta moto concreta?", el predictor óptimo de Bayes es la
marginal `cargadas/total` de su clase — exactamente aquello a lo que converge la entropía
cruzada. El único costo es un techo en la exactitud cruda, y ese techo ya está cuantificado
en la sección 5: unos 8 puntos.

---

## 8. Qué significa esto para el reporte

1. **La sección VII no puede reportar sólo exactitud.** Con las etiquetas de hoy, 0,53 no
   dice si el modelo es bueno: dice que la etiqueta depende de algo invisible. Las métricas
   principales son las de dominio — violación de capacidad, brecha de vehículos cargados,
   brecha de CU — y la exactitud va como diagnóstico secundario, **acompañada de su techo**.
2. **Hay que citar el techo.** Decir "0,5297 de exactitud" invita a la pregunta obvia.
   Decir "0,5297 sobre un techo exacto de 0,9243, con la brecha explicada y cuantificada"
   es un resultado.
3. **Si se aplica la opción B, hay que rehacer todas las cifras** de la sección VII y
   volver a exportar las matrices de confusión.
4. **Si no se aplica**, queda declarado en VIII (posibilidades futuras) con la medición
   hecha: se sabe cuál es la causa, cuánto cuesta arreglarla y cuánto se gana.

---

## 9. Lo que necesito de ustedes

Me dijeron que tienen "un problema con el etiquetador" pero no cuál. Esta tabla mapea el
síntoma a la causa, para no dar vueltas:

| Síntoma que estén viendo | Causa probable | Dónde está tratado |
|---|---|---|
| La exactitud no pasa de ~0,5 por más que ajusten | S1 + S2 | Secciones 2 y 6 |
| `CAMION_3`/`CAMION_4` sin ejemplos suficientes, o *recall* 0 | S1 | Sección 3 |
| Dos corridas dan resultados distintos | S3, o semilla no fijada | Sección 7-C |
| La matriz de confusión colapsa a una sola columna | S1 + S2 | Sección 6 |
| El modelo no sabe cuántos camiones hay | Falta el join con `episodes.parquet` | `05_hallazgos` §1 |
| El etiquetador se demora | Nada de esto — es `time_budget_s` de `labeler.py` | — |

Y la decisión que hay que tomar hoy: **¿aplicamos la opción B?** Está medida, cuesta 8
minutos de regeneración, y obliga a reentrenar los cinco modelos. Yo no la aplico solo.
