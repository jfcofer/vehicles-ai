# Tabla V — Caso de Uso de Análisis

> **Entregable de Juan Francisco Fernández Ramos** (Planificación, pág. 4:
> *"2 Escenarios para la Etapa de Análisis"*).
> Reemplaza los marcadores `1.`, `2a.` y `a` de la pág. 13 de `Tarea #4.pdf`.
>
> **Para pegar en el Google Doc:** copiar la tabla de la sección 1. El resto de este
> archivo es justificación y trazabilidad, y **no va al reporte**.

---

## 1. Tabla lista para pegar

TABLA V

CASO DE USO DE ANÁLISIS

| | |
|---|---|
| **Caso de Uso** | Analizar manifiesto de vehículos a transportar |
| **Actor** | Operador de Carga |
| **Descripción** | El sistema analiza el manifiesto de vehículos a transportar entregado por el Operador de Carga, considerando el identificador único de cada vehículo, el cantón de destino de cada vehículo, la clase de cada vehículo y el espacio que ocupa cada vehículo, para determinar la ubicación de cada vehículo en los camiones de carga disponibles. |
| **Pre-Condición** | El sistema ha guardado el manifiesto de vehículos a transportar. El Operador de Carga tiene información de la cantidad de camiones de carga disponibles y del espacio disponible en cada uno de los camiones de carga. |
| **Pos-Condición** | El sistema ha analizado el manifiesto de vehículos a transportar. Cada vehículo del manifiesto queda asociado a exactamente un camión de carga o queda marcado como diferido, y para cada camión de carga la suma de las unidades de almacenamiento de los vehículos asociados no supera su espacio disponible. |
| **Secuencia por defecto** | 1. El sistema toma el manifiesto de vehículos a transportar normalizado, la cantidad de camiones de carga disponibles y el espacio disponible de cada camión de carga.<br><br>2. El sistema determina que la suma de las unidades de almacenamiento de los vehículos del manifiesto no supera la capacidad agregada de los camiones de carga disponibles.<br><br>3. El sistema determina a qué camión de carga corresponde cada vehículo del manifiesto, de modo que en ningún camión de carga la suma de las unidades de almacenamiento de los vehículos asignados supere su espacio disponible.<br><br>4. El sistema entrega el análisis del manifiesto con todos los vehículos ubicados en un camión de carga y sin vehículos diferidos. |
| **Secuencia alternativa** | 2a. El sistema determina que la suma de las unidades de almacenamiento de los vehículos del manifiesto supera la capacidad agregada de los camiones de carga disponibles.<br><br>3a. El sistema determina la mayor cantidad de vehículos del manifiesto que pueden ubicarse sin que ningún camión de carga supere su espacio disponible y, entre las alternativas que ubican esa misma cantidad de vehículos, la que aprovecha la mayor cantidad de unidades de almacenamiento.<br><br>4a. El sistema entrega el análisis del manifiesto indicando qué vehículos quedan ubicados en un camión de carga y qué vehículos quedan diferidos por no contar con espacio disponible.<br><br>a La suma de las unidades de almacenamiento de los vehículos del manifiesto supera la capacidad agregada de los camiones de carga disponibles. |

### 1.b Las dos secuencias en texto plano

Por si se copia el markdown en crudo y los `<br>` salen literales. El contenido es el
mismo que el de la tabla de arriba.

**Secuencia por defecto**

```
1. El sistema toma el manifiesto de vehículos a transportar normalizado, la cantidad de
   camiones de carga disponibles y el espacio disponible de cada camión de carga.

2. El sistema determina que la suma de las unidades de almacenamiento de los vehículos
   del manifiesto no supera la capacidad agregada de los camiones de carga disponibles.

3. El sistema determina a qué camión de carga corresponde cada vehículo del manifiesto,
   de modo que en ningún camión de carga la suma de las unidades de almacenamiento de
   los vehículos asignados supere su espacio disponible.

4. El sistema entrega el análisis del manifiesto con todos los vehículos ubicados en un
   camión de carga y sin vehículos diferidos.
```

**Secuencia alternativa**

```
2a. El sistema determina que la suma de las unidades de almacenamiento de los vehículos
    del manifiesto supera la capacidad agregada de los camiones de carga disponibles.

3a. El sistema determina la mayor cantidad de vehículos del manifiesto que pueden
    ubicarse sin que ningún camión de carga supere su espacio disponible y, entre las
    alternativas que ubican esa misma cantidad de vehículos, la que aprovecha la mayor
    cantidad de unidades de almacenamiento.

4a. El sistema entrega el análisis del manifiesto indicando qué vehículos quedan
    ubicados en un camión de carga y qué vehículos quedan diferidos por no contar con
    espacio disponible.

a   La suma de las unidades de almacenamiento de los vehículos del manifiesto supera la
    capacidad agregada de los camiones de carga disponibles.
```

---

## 2. Los dos escenarios

| # | Escenario | Situación del dominio | Lo que debe cumplirse |
|---|---|---|---|
| 1 | **Capacidad suficiente** (secuencia por defecto) | La suma de unidades de almacenamiento del manifiesto cabe en la capacidad agregada de la flota. | Todo vehículo queda ubicado; ningún camión supera su espacio disponible; no hay vehículos diferidos. |
| 2 | **Capacidad excedida** (secuencia alternativa, condición `a`) | La suma de unidades de almacenamiento del manifiesto supera la capacidad agregada de la flota. | Se ubica la **mayor cantidad posible** de vehículos; ante empate en cantidad, se aprovecha la **mayor cantidad de unidades de almacenamiento**; el resto queda explícitamente diferido; ningún camión supera su espacio disponible. |

---

## 3. Justificación

**Por qué exactamente estos dos.** El caso de uso de análisis se bifurca en un único punto
de decisión del dominio: si el manifiesto cabe o no cabe en la flota. Todo lo demás
(cuántos cantones, qué clases, cuántos camiones) es variación dentro de una de esas dos
ramas, no un escenario distinto del caso de uso. Las Tablas IV y VI ya cubren las
variaciones de forma del manifiesto y del plan de salida; duplicarlas aquí sería redundante.

**Por qué el paso 3a está redactado así.** El orden *primero cantidad de vehículos, después
unidades de almacenamiento aprovechadas* no es una elección arbitraria: es el objetivo
declarado en la Fig. 1 de la planificación (`LoadingRequirement`):

```
Goal: maximize |{v: v in assignedTo(Truck1) ∪ assignedTo(Truck2)}|
then: minimize sum(t.capacity - used(t) for t in Truck)
then: minimize |{v: v in deferred}|
```

Minimizar el espacio sobrante equivale a maximizar las unidades de almacenamiento
aprovechadas, y minimizar los diferidos ya está implicado por maximizar los cargados.
La Tabla V queda así trazable, palabra por palabra, al requerimiento formal.

**Por qué no se menciona ningún modelo.** El profesor pide escenarios por caso de uso, y
la sección IV del reporte es *análisis y definición del problema*, no diseño de la solución.
Ninguno de los dos escenarios nombra clasificador, red, búsqueda ni entrenamiento: sólo
describen configuraciones del dominio y las restricciones que la ubicación debe satisfacer.
Este criterio ya estaba fijado en `chat/2026-07-19-01-scenarios.md`.

---

## 4. Correcciones propuestas al esqueleto actual

Dos cambios respecto a lo que hoy está en la pág. 13. **Requieren tu visto bueno**, porque
tocan celdas que ya estaban escritas:

| Celda | Antes | Ahora | Motivo |
|---|---|---|---|
| **Descripción** | *"El Operador de Carga **analiza** en el manifiesto…"* | *"El **sistema analiza** el manifiesto… entregado por el Operador de Carga…"* | Quien analiza es el sistema. En la Fig. 2 este caso de uso es un `<<include>>` de "Entregar manifiesto", no una acción manual del operador. Las Tablas IV y VI son consistentes en esto: el operador *entrega* y *obtiene*; el sistema *acepta*, *normaliza*, *guarda* e *indica*. |
| **Pre-Condición** | Sólo la información de la flota | Se añade *"El sistema ha guardado el manifiesto…"* | Es literalmente la Pos-Condición de la Tabla IV. Encadena los tres casos de uso: entrada → análisis → salida. |

Si prefieres no tocarlas, los escenarios de la sección 1 funcionan igual; sólo quedaría la
incoherencia de que el operador "analiza" un manifiesto que el sistema ya guardó.
