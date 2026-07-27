"""Contrato de datos: el join trae la flota y las particiones no filtran episodios."""

from __future__ import annotations

import pandas as pd
import pytest

from src.modeling.dataset import (
    assert_no_episode_leakage,
    drop_non_optimal,
    load_episode_tables,
    split_by_episode_hash,
    split_by_time,
    summarize_splits,
)


def _write_tables(tmp_path, episodes: list[dict], vehicles: list[dict]):
    ep_path = tmp_path / "episodes.parquet"
    ve_path = tmp_path / "episode_vehicles.parquet"
    pd.DataFrame(episodes).to_parquet(ep_path, index=False)
    pd.DataFrame(vehicles).to_parquet(ve_path, index=False)
    return ep_path, ve_path


def _episode(episode_id: str, iso_year: int, caps: list[float], optimal: bool = True) -> dict:
    return {
        "episode_id": episode_id,
        "iso_year": iso_year,
        "iso_week": 2,
        "canton": 10701,
        "n_sampled": 2,
        "n_trucks": len(caps),
        "truck_capacities": caps,
        "n_loaded": 2,
        "n_deferred": 0,
        "cu_utilized": 2.0,
        "optimal": optimal,
    }


def _vehicle(episode_id: str, uid: str, truck: str = "CAMION_1") -> dict:
    return {
        "episode_id": episode_id,
        "uid": uid,
        "clase": "AUTOMOVIL",
        "cu": 1.0,
        "canton": 10701,
        "truck": truck,
        "loaded": truck != "SIN_CAMION",
    }


def test_el_join_trae_las_capacidades_de_la_flota_a_cada_fila(tmp_path):
    """Sin esto, ninguna fila sabe cuántos camiones hay ni de qué tamaño."""
    ep, ve = _write_tables(
        tmp_path,
        [_episode("E1", 2024, [6.0, 3.0])],
        [_vehicle("E1", "a"), _vehicle("E1", "b")],
    )
    joined = load_episode_tables(ep, ve)

    assert len(joined) == 2
    assert "truck_capacities" in joined.columns
    assert "n_trucks" in joined.columns
    assert list(joined.loc[0, "truck_capacities"]) == [6.0, 3.0]


def test_falla_si_un_vehiculo_no_tiene_episodio(tmp_path):
    ep, ve = _write_tables(
        tmp_path,
        [_episode("E1", 2024, [6.0])],
        [_vehicle("E1", "a"), _vehicle("HUERFANO", "b")],
    )
    with pytest.raises(ValueError, match="no encontraron su episodio"):
        load_episode_tables(ep, ve)


def test_falla_si_hay_episode_id_duplicado(tmp_path):
    ep, ve = _write_tables(
        tmp_path,
        [_episode("E1", 2024, [6.0]), _episode("E1", 2025, [3.0])],
        [_vehicle("E1", "a")],
    )
    with pytest.raises(ValueError, match="duplicados"):
        load_episode_tables(ep, ve)


def test_falla_si_faltan_columnas(tmp_path):
    ep_path = tmp_path / "episodes.parquet"
    ve_path = tmp_path / "episode_vehicles.parquet"
    pd.DataFrame([{"episode_id": "E1"}]).to_parquet(ep_path, index=False)
    pd.DataFrame([_vehicle("E1", "a")]).to_parquet(ve_path, index=False)
    with pytest.raises(ValueError, match="columnas requeridas"):
        load_episode_tables(ep_path, ve_path)


def test_se_descartan_los_episodios_no_optimos(tmp_path):
    ep, ve = _write_tables(
        tmp_path,
        [_episode("E1", 2024, [6.0]), _episode("E2", 2024, [6.0], optimal=False)],
        [_vehicle("E1", "a"), _vehicle("E2", "b")],
    )
    joined = load_episode_tables(ep, ve)
    kept, dropped = drop_non_optimal(joined)

    assert dropped == 1
    assert set(kept["episode_id"]) == {"E1"}


def test_particion_temporal_respeta_los_anos(tmp_path):
    ep, ve = _write_tables(
        tmp_path,
        [_episode("E1", 2019, [6.0]), _episode("E2", 2025, [6.0]), _episode("E3", 2026, [6.0])],
        [_vehicle("E1", "a"), _vehicle("E2", "b"), _vehicle("E3", "c")],
    )
    splits = split_by_time(load_episode_tables(ep, ve))

    assert set(splits["train"]["episode_id"]) == {"E1"}
    assert set(splits["val"]["episode_id"]) == {"E2"}
    assert set(splits["test"]["episode_id"]) == {"E3"}
    assert_no_episode_leakage(splits)


def test_particion_temporal_rechaza_anos_solapados(tmp_path):
    ep, ve = _write_tables(tmp_path, [_episode("E1", 2024, [6.0])], [_vehicle("E1", "a")])
    joined = load_episode_tables(ep, ve)
    with pytest.raises(ValueError, match="se solapan"):
        split_by_time(joined, train_years=(2024,), val_years=(2024,), test_years=(2026,))


def test_particion_por_hash_nunca_parte_un_episodio(tmp_path):
    """El defecto que más infla las métricas: repartir filas del mismo episodio
    entre entrenamiento y prueba."""
    episodes = [_episode(f"E{i}", 2024, [6.0, 3.0]) for i in range(60)]
    vehicles = [_vehicle(f"E{i}", f"{i}-{j}") for i in range(60) for j in range(5)]
    ep, ve = _write_tables(tmp_path, episodes, vehicles)

    splits = split_by_episode_hash(load_episode_tables(ep, ve))

    assert_no_episode_leakage(splits)
    assert sum(len(d) for d in splits.values()) == 300
    assert all(len(d) > 0 for d in splits.values())


def test_particion_por_hash_es_estable_entre_llamadas(tmp_path):
    episodes = [_episode(f"E{i}", 2024, [6.0]) for i in range(40)]
    vehicles = [_vehicle(f"E{i}", f"{i}-0") for i in range(40)]
    ep, ve = _write_tables(tmp_path, episodes, vehicles)
    joined = load_episode_tables(ep, ve)

    first = split_by_episode_hash(joined)
    second = split_by_episode_hash(joined)
    for name in ("train", "val", "test"):
        assert list(first[name]["episode_id"]) == list(second[name]["episode_id"])


def test_el_resumen_cuenta_episodios_filas_y_diferidos(tmp_path):
    ep, ve = _write_tables(
        tmp_path,
        [_episode("E1", 2024, [6.0])],
        [_vehicle("E1", "a"), _vehicle("E1", "b", truck="SIN_CAMION")],
    )
    splits = split_by_time(load_episode_tables(ep, ve))
    summary = {s.name: s for s in summarize_splits(splits)}

    assert summary["train"].n_episodes == 1
    assert summary["train"].n_rows == 2
    assert summary["train"].n_deferred_rows == 1
    assert summary["train"].deferred_pct == 50.0


def test_la_fuga_de_episodios_se_detecta():
    df = pd.DataFrame([{"episode_id": "E1", "iso_year": 2024, "loaded": True}])
    with pytest.raises(AssertionError, match="E1"):
        assert_no_episode_leakage({"train": df, "test": df})
