# Vehicle Feature Coverage

> **Auto-generated.** Reproduce with:
> ```bash
> python3 scripts/build_vehicle_features.py
> ```

**Generated:** 2026-07-25 22:25 UTC  
**Output:** `data/features/vehicles_in_scope.parquet`  
**Skipped years (no process-date column):** 2017

---

## Vehicle-code deduplication

Same `CÓDIGO DE VEHÍCULO` can appear in multiple rows with only `FECHA PROCESO` differing (reprocessing) -- distinct from the exact-row duplicates already removed per `docs/deduplication_workflow.md`. One row is kept per vehicle (earliest `fecha`).

| | |
|---|---|
| Rows before | 3,006,478 |
| Unique vehicles after | 2,587,129 |
| Rows removed | 419,349 (13.9%) |
| Vehicles that spanned 2+ different ISO weeks pre-dedup | 166,798 |

---

## Scope filter

| | Rows | % |
|---|------|---|
| Total (all SRI classes, post vehicle-dedup) | 2,587,129 | 100.0% |
| Kept (in-scope classes) | 2,491,511 | 96.3% |
| Dropped (out-of-scope classes) | 95,618 | 3.7% |

### Dropped, by class

| Clase | Rows dropped |
|-------|--------------|
| CAMION | 59,996 |
| OMNIBUS | 9,418 |
| TRAILER | 9,275 |
| ESPECIAL | 8,726 |
| VOLQUETA | 5,142 |
| TANQUERO | 3,061 |

No unrecognized classes — config/vehicle_classes.yaml covers 100% of the raw CLASE catalog.
