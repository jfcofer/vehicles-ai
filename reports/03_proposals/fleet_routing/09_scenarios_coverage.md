# Scenarios Coverage

> **Auto-generated.** Reproduce with:
> ```bash
> python3 scripts/build_scenarios.py
> ```

**Generated:** 2026-07-25 22:33 UTC  
**Elapsed:** 420.4s  
**Floor (min N kept):** 5  
**Max N per episode (subsample cap):** 20

---

## Episode universe

| | |
|---|---|
| Grupos semana-cantón totales | 55,076 |
| Excluidos por piso (N<5) | 20,237 |
| Episodios construidos y etiquetados | 34,839 |

## Resultado del labeler

| | |
|---|---|
| Filas en episode_vehicles.parquet | 534,680 |
| Episodios triviales (nadie deferido) | 29,860 (85.7%)
| Episodios no-óptimos (time_budget agotado) | 0 |
| search_time_ms promedio | 10.8
| search_time_ms p99 | 167.6
