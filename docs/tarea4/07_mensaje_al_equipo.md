# Mensaje al equipo — 27 de julio

> Versión corta para enviar por chat. El detalle está en
> [`06_canonicalizacion_y_etiquetado.md`](06_canonicalizacion_y_etiquetado.md).

---

Encontré la causa del problema del etiquetador y la medí. Hay **una decisión que tomar hoy**.

**Corrección primero:** les había dicho que la baja exactitud era ruido irreducible de la
etiqueta (~60 %). **Eso estaba mal.** Calculé el techo real de exactitud y es **0,9243**,
no 0,3983. El modelo estaba capturando el 59 % de lo alcanzable, no el 100 %. Si alguien ya
escribió lo del "60 % de ruido" en el reporte, hay que quitarlo.

**La causa:** `generate_fleet()` devuelve las capacidades **en orden aleatorio**. Eso no
sólo cambia el *nombre* del camión — cambia el *plan* que produce el etiquetador. Con la
flota `[6,0 · 4,0]` la PD llena el camión grande y con `[4,0 · 6,0]` llena el chico: los dos
planes son igual de óptimos (mismos vehículos cargados, misma CU) pero **ningún
renombramiento lleva uno al otro**. Por eso canonicalizar la salida no lo arregla.

**El arreglo:** ordenar la flota **antes** de etiquetar. Una línea en `scenarios.py`.
Lo probé de punta a punta — regeneré el dataset completo y reentrené con los mismos
hiper-parámetros y la misma semilla:

| Prueba 2026 | Hoy | Con la flota ordenada |
|---|---:|---:|
| Exactitud cruda | 0,5297 | **0,8458** |
| F1 macro | 0,2996 | **0,8131** |
| Concordancia por clase | 0,5507 | **0,9293** |
| *Recall* de `CAMION_4` | **0,000** | **0,872** |
| Brecha de vehículos cargados | +0,0229 | +0,0242 |
| Violaciones de capacidad | 0,0000 | 0,0000 |

Las métricas operativas no se mueven. Lo que cambia es que la etiqueta se vuelve aprendible
y la matriz de confusión deja de colapsar.

**Es más barato de lo que parece:** ordenar no consume aleatoriedad, así que `n_loaded`,
`n_deferred`, `cu_utilized` y `optimal` salen **idénticos en los 34.839 episodios**. Sólo
cambia la columna `truck`. Son **8 minutos** de regeneración.

---

## La decisión

**¿Regeneramos el dataset con la flota ordenada?** Cuesta 8 min + reentrenar los cinco
modelos, y hay que rehacer las cifras de la sección VII.

- **Si sí:** hay que arrancar ya. Aplico el parche, corro `build_scenarios.py`, avisan
  cuando esté y cada uno reentrena.
- **Si no:** queda declarado en la sección VIII como trabajo futuro, con la medición hecha.
  Ya está escrito así, no hay que redactar nada.

**No lo apliqué solo** porque invalida lo que ya entrenaron.

---

## Lo que pueden usar hoy, decidan lo que decidan

**1. Canonicalizar la flota** — `src/modeling/canonicalization.py`, tres líneas, Python puro
sin dependencias:

```python
from src.modeling.canonicalization import canonicalize_fleet

fleet = canonicalize_fleet(row["truck_capacities"])
y     = fleet.label_map[row["truck"]]   # etiqueta canónica
caps  = fleet.capacities                # capacidades en el MISMO orden
```

Esto **no sube la exactitud** — que quede claro para que no lo descarten al ver que el
número no se mueve. Lo que hace es subir `CAMION_4` de 0,26 % a 4,97 % de soporte (con
0,26 % ningún modelo la aprende) y que los cinco midamos contra la misma convención.

> ⚠ **Si canonicalizan la etiqueta, canonicalicen también las features de capacidad.**
> Si tienen `cap_1..cap_4` en el orden del parquet y la etiqueta remapeada, la
> correspondencia se rompe y queda peor que antes. Usen el objeto `fleet` entero.
> Y para mostrarle el resultado al operador, reviertan con `fleet.inverse_label_map`.

**2. El techo de exactitud** — `uv run python scripts/label_ceiling.py`. No depende de la
arquitectura, sirve para los cinco modelos. Reportar "0,53 de exactitud" invita a la
pregunta obvia; reportar "0,53 sobre un techo exacto de 0,92, con la brecha explicada" es
un resultado.

**3. El decodificador y las métricas** — `capacity_decoder.py` y `metrics.py` funcionan con
cualquier modelo que produzca puntuaciones por vehículo y camión. Si los tres usamos los
mismos, la tabla comparativa de los cinco modelos es una comparación de verdad.

---

## Díganme cuál es su síntoma

| Lo que ven | Causa | Dónde está tratado |
|---|---|---|
| La exactitud no pasa de ~0,5 por más que ajusten | Orden de la flota | `06_` §2 y §6 |
| `CAMION_3`/`CAMION_4` sin ejemplos, o *recall* 0 | Nombre del camión | `06_` §3 |
| La matriz de confusión colapsa a una columna | Orden de la flota | `06_` §6 |
| Dos corridas dan resultados distintos | Semilla, o reparto intra-clase | `06_` §7-C |
| El modelo no sabe cuántos camiones hay | Falta el join con `episodes.parquet` | `05_` §1 |
| El etiquetador se demora | Nada de esto — es `time_budget_s` | — |
