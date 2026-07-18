# Scenarios Coverage

> **Auto-generated.** Reproduce with:
> ```bash
> python3 scripts/build_scenarios.py
> ```

**Generated:** 2026-07-18 15:57 UTC  
**Elapsed:** 3.6s  
**Floor (min N kept):** 5  
**Max N per episode (subsample cap):** 20

---

## Episode universe

| | |
|---|---|
| Grupos semana-cantón totales | 360 |
| Excluidos por piso (N<5) | 160 |
| Episodios construidos y etiquetados | 200 |

## Resultado del labeler

| | |
|---|---|
| Filas en episode_vehicles.parquet | 2,894 |
| Episodios triviales (nadie deferido) | 153 (76.5%)
| Episodios no-óptimos (time_budget agotado) | 0 |
| search_time_ms promedio | 13.8
| search_time_ms p99 | 138.2
