# Anexo — Uso de herramientas basadas en IA (Juan Francisco Fernández Ramos)

> **Aporte de Juan Francisco Fernández Ramos** al Anexo del reporte (Planificación,
> pág. 7). Se integra en los epígrafes A, B y C del Anexo existente, que hoy sólo
> documentan el trabajo sobre *gradient boosting*.
>
> Transcripciones: `chat/2026-07-25-juan-mlp-design-training-evaluation.md`,
> `chat/2026-07-19-01-scenarios.md`, `chat/2026-07-22-input-output-use-case.md`.

---

## A. Cómo se integraron las respuestas generadas

### 1. Escenarios del caso de uso de análisis

La sesión `chat/2026-07-19-01-scenarios.md` se abrió preguntando qué caracteriza a un
escenario centrado en el problema y no en la solución. La herramienta propuso siete
escenarios de dominio (semana pico, tensión de empaquetado por clases mixtas, desborde de
un único vehículo, dispersión geográfica, agrupación en un solo cantón, casos límite de
precisión en CU, dos vehículos que juntos exceden un camión) y, sobre todo, un criterio
explícito: un escenario problem-focused describe la situación del dominio y las
restricciones que deben cumplirse, sin nombrar búsqueda, modelo ni algoritmo.

Ese criterio se adoptó y determinó la redacción de la Tabla V. Los siete escenarios, en
cambio, **no** se trasladaron uno a uno. Al analizarlos se concluyó que el caso de uso de
análisis se bifurca en un único punto de decisión del dominio —si el manifiesto cabe o no
cabe en la flota— y que los demás son variaciones dentro de una de esas dos ramas, no
escenarios distintos del caso de uso. La Tabla V quedó por tanto con una secuencia por
defecto y una alternativa, siguiendo la estructura que ya usaban las Tablas IV y VI.

La sesión `chat/2026-07-22-input-output-use-case.md` aportó la distinción entre lo que el
operador ve y lo que el sistema hace internamente. De ahí se corrigió la descripción de la
Tabla V: el original decía que *el Operador de Carga analiza* el manifiesto, cuando en la
Fig. 2 este caso de uso es un `<<include>>` del caso de entrada y quien analiza es el
sistema.

### 2. Arquitectura del clasificador

La consulta inicial planteó dos alternativas para un MLP con salida `CAMIÓN 1..n`: una
capa `Dense(5)` de posiciones fijas, o un perceptrón compartido que puntúa cada par
vehículo–camión. La respuesta recomendó la segunda y aportó tres elementos que se
adoptaron íntegros:

- **Compartición de parámetros.** Aplicar el mismo perceptrón a todos los pares hace que
  el número de parámetros no dependa de la cantidad de camiones. Ésta es la razón técnica
  por la que la arquitectura satisface el requisito de la planificación de considerar una
  flota sin límite codificado, y quedó documentada en la sección VI-D.
- **Enmascarado del relleno.** Asignar un logit de −1e9 a las posiciones de relleno para
  que desaparezcan del softmax, y una cabeza separada para `SIN CAMIÓN`.
- **Decodificador con restricción de capacidad.** El argumento —tres vehículos de 3 CU
  con predicciones individualmente razonables suman 9 en un camión de 6— se comprobó
  experimentalmente y se convirtió en una prueba del repositorio
  (`test_nunca_excede_la_capacidad_aunque_el_modelo_insista`).

### 3. Diagnóstico de la baja exactitud

La herramienta identificó tres causas potenciales antes de escribir código: que las filas
de vehículos no contienen las capacidades de la flota, que el etiquetador reparte los
cupos dentro de una clase con un barajado sembrado, y que las etiquetas de camión son
sensibles a permutación. Las tres se verificaron en el código fuente y las tres resultaron
ciertas. De ahí salieron `src/modeling/dataset.py:load_episode_tables()` (el join) y
`src/modeling/canonicalization.py` (el orden canónico por capacidad).

---

## B. Criterios propios añadidos más allá de las respuestas generadas

Cinco recomendaciones se rechazaron o modificaron tras contrastarlas con el código y con
la documentación del propio proyecto.

**1. Se rechazó el uso de OR-Tools / CP-SAT como oráculo exacto.** La herramienta lo
propuso para instancias pequeñas, ignorando que el repositorio tiene una decisión de
alcance escrita en `src/loading/labeler.py` y en `02_scope.md`: *"No external solver
(OR-Tools, PuLP, MIP libraries, etc.) is used. This is a from-scratch exact search…
written and owned by the team"*. Además, el maestro exacto ya existe y expone un indicador
`optimal`. Incorporar el solver habría violado un acuerdo del grupo sin aportar nada.

**2. Se rechazó implementar el modelo como subclase de `keras.Model`.** El código sugerido
definía la clase con su método `call` pero sin `get_config()`. En lugar de suponer que eso
daba problemas, se reprodujo el caso con Keras 3.15. Falla de dos maneras distintas:

```
load_model("subclass.keras")
  -> TypeError: Could not locate class 'VariableTruckAssignmentMLP'.

load_model("subclass.keras", custom_objects={...})
  -> ValueError: A total of 3 objects could not be loaded.
     Layer 'dense_3' expected 2 variables, but received 0 variables during loading.
```

El segundo fallo es el instructivo: aun entregando la clase, sin `get_config()` Keras
reconstruye el modelo con los **valores por defecto** del constructor en lugar de los que
se usaron al entrenar, de modo que la arquitectura recargada no coincide con los pesos
guardados. El error aparece al final del entrenamiento, cuando el tiempo de cómputo ya se
gastó.

Se reimplementó con la API funcional, que serializa sin objetos personalizados y además
produce un `model.summary()` presentable para el reporte. La prueba
`test_guardar_y_recargar_produce_logits_identicos` fija ese comportamiento.

**3. Se movió `SIN CAMIÓN` del último índice al primero.** La propuesta situaba el
diferimiento en la posición `max_trucks` del lote, que cambia según el manifiesto más
grande de cada lote. Es una fuente silenciosa de errores de etiquetado. Colocándolo en el
índice 0, el objetivo es estable con dos o con ocho camiones.

**4. Se corrigió la partición temporal.** La recomendación proponía entrenar con 2017–2024.
El archivo del SRI de 2017 no contiene la columna `FECHA PROCESO` y `load_all_years` lo
descarta; está registrado en `08_feature_coverage.md` bajo *"Skipped years"*. La partición
efectiva es 2018–2024 / 2025 / 2026.

**5. Se eliminó el umbral de diferimiento del decodificador, y se midió el costo de
mantenerlo.** El decodificador sugerido saltaba los vehículos cuyo margen fuera negativo,
sin siquiera intentar ubicarlos. Como el objetivo del maestro es lexicográfico —primero
cuántos vehículos carga, después cuánta capacidad aprovecha—, diferir voluntariamente un
vehículo que cabe sólo puede empeorar la métrica principal. En vez de argumentarlo, se
implementó como política alternativa y se midió sobre validación: honrar el `SIN CAMIÓN`
predicho degrada la brecha de conteo de **+0,0303 a +0,9883**, un factor de 32.

### Aportes que no salieron de ninguna herramienta

**El experimento del techo empírico.** Ante una exactitud cruda de 0,53, la pregunta
natural es si el modelo está mal o la etiqueta no es predecible. Ninguna herramienta lo
planteó. Se diseñó `scripts/teacher_self_agreement.py`: se le presenta al etiquetador
exacto **la misma flota permutada** —mismos camiones, mismas capacidades, situación
operativamente idéntica— y se compara su nueva respuesta con la original. Resultado sobre
2026: el maestro reproduce el 39,83 % de sus propias etiquetas, mientras el resultado
operativo es idéntico (|Δ cargados| = 0,0000). Es decir, cerca del 60 % de la etiqueta es
ruido de desempate. Ese dato reencuadra por completo la discusión que el grupo tenía sobre
la baja exactitud.

Conviene precisar el alcance de esa cifra: la autoconsistencia del maestro **no** es un
techo estricto para el modelo, porque predecir la moda de una variable aleatoria coincide
con una muestra más a menudo de lo que dos muestras independientes coinciden entre sí.
Lo que sí cuantifica es cuánta de la etiqueta es arbitraria.

**La métrica de concordancia por clase.** Derivada del hallazgo anterior: si cuál vehículo
concreto de una clase recibe el cupo es una moneda al aire, la métrica debe comparar lo
único que el maestro sí determinó —cuántos vehículos de cada clase van a cada camión—.
Se implementó como `1 −` distancia de variación total en `src/modeling/metrics.py`.

**El criterio de selección de hiper-parámetros.** Se seleccionó por brecha de conteo en
validación, no por exactitud ni por pérdida, precisamente porque la exactitud está
contaminada por el ruido medido arriba.

---

## C. Beneficios y limitaciones encontrados

### Beneficios

- **Velocidad para explorar el espacio de diseño.** La comparación entre posiciones fijas
  y puntuación por par, con sus implicaciones sobre el número de parámetros y la
  generalización, habría tomado bastante más tiempo de lectura de documentación.
- **Anticipación de defectos de datos.** Las tres causas de baja exactitud se señalaron
  antes de escribir la primera línea de código de entrenamiento, y las tres se confirmaron
  al inspeccionar el código fuente. Sin esa advertencia, lo más probable habría sido
  entrenar un clasificador de cinco clases sobre el Parquet de vehículos y quedarse con
  una exactitud baja sin entender el motivo.
- **Andamiaje de pruebas.** Las herramientas fueron útiles para enumerar casos límite del
  decodificador (cero camiones, capacidades iguales, carga exacta) que conviene fijar en
  pruebas.

### Limitaciones

- **Recomiendan sin conocer los acuerdos del proyecto.** La propuesta de OR-Tools era
  técnicamente razonable y contradecía una decisión de alcance escrita en el propio
  repositorio. La herramienta no lee esos acuerdos salvo que se le entreguen, y aun
  entonces tiende a optimizar el problema aislado y no el proyecto.
- **Producen código que parece correcto y no lo es.** El `keras.Model` sin `get_config()`
  compila, entrena y sólo falla al recargar. Es el tipo de defecto que un lector rápido
  aprueba y que aparece al final. La conclusión práctica es que el código generado necesita
  una prueba que ejercite el ciclo completo, no una lectura.
- **Arrastran supuestos no verificados sobre los datos.** La partición «2017–2024» se
  propuso sin comprobar que 2017 existiera en el conjunto; el dato contrario estaba en un
  reporte autogenerado del repositorio.
- **Optimizan la métrica equivocada por defecto.** Varias sugerencias giraban en torno a
  maximizar exactitud, cuando en este problema la exactitud está limitada por ruido de
  etiquetado y la métrica que importa es la brecha operativa contra el óptimo. La
  herramienta no puede saber eso sin que se le mida y se le diga.
- **No cuestionan la calidad de la etiqueta.** Ninguna sesión sugirió comprobar si el
  objetivo era predecible antes de intentar predecirlo. Fue necesario formular esa
  hipótesis y diseñar el experimento manualmente.

### Conclusión

La utilidad fue alta para explorar alternativas de diseño y anticipar defectos, y baja
para decidir. Cada recomendación adoptada se verificó contra el código fuente del
repositorio, y cinco de ellas se rechazaron o modificaron por esa vía. El aporte que más
cambió el rumbo del trabajo —medir cuánta de la etiqueta es reproducible— no provino de
ninguna herramienta, sino de desconfiar de una métrica que no cuadraba.
