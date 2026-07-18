"""
src/loading/scenarios.py

Turns data/features/vehicles_in_scope.parquet into labeled training
instances for the imitation-learning "student" model:

    group by (iso_year, iso_week, canton)          <- the natural episode unit
        -> drop groups with N < FLOOR (=5)          <- see conversation: below this,
                                                         the current fleet policy makes
                                                         a binding decision ~impossible
        -> stratified subsample to <= MAX_N (=20)    <- labeler's practical budget
        -> synthetic truck fleet (n_trucks, capacities)
        -> labeler.assign_vehicles()                 <- exact optimal assignment
    -> one row per episode  (data/episodes/episodes.parquet)
    -> one row per (episode, vehicle)  (data/episodes/episode_vehicles.parquet)

Reproducibility: every random draw for an episode (which vehicles get
subsampled, the truck fleet, and the labeler's within-class tie-breaking)
comes from ONE `random.Random` seeded from a stable hash of the episode key
(iso_year, iso_week, canton) -- NOT Python's built-in `hash()`, which is
randomized per-process by PYTHONHASHSEED and would silently break
reproducibility across runs. Same episode key always -> same training
example, on any machine, any run.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import pandas as pd

from src.loading.labeler import Vehicle, assign_vehicles

FLOOR_N = 5          # below this, see module docstring -- decided in conversation
MAX_N = 20           # labeler's practical per-episode budget (see 06_feasibility.md)
N_TRUCKS_RANGE = (1, 4)
CAP_RANGE = (3.0, 9.0)


def episode_id(iso_year: int, iso_week: int, canton) -> str:
    return f"{iso_year}-W{int(iso_week):02d}-{canton}"


def episode_seed(iso_year: int, iso_week: int, canton) -> int:
    """Stable seed derived from the episode key -- see module docstring for
    why this can't just be Python's `hash()`."""
    key = episode_id(iso_year, iso_week, canton)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def stratified_subsample(group: pd.DataFrame, max_n: int, rng: random.Random) -> tuple[pd.DataFrame, int]:
    """If len(group) > max_n, sample down to max_n preserving class
    proportions (largest-remainder rounding so counts sum exactly to
    max_n). Returns (sampled_frame, n_excluded)."""
    n = len(group)
    if n <= max_n:
        return group, 0

    class_counts = group["clase"].value_counts()
    raw = class_counts / n * max_n
    target = raw.astype(int)
    remainder = max_n - int(target.sum())
    if remainder > 0:
        fracs = (raw - target).sort_values(ascending=False)
        for c in fracs.index[:remainder]:
            target[c] += 1

    parts = []
    for clase, k in target.items():
        if k <= 0:
            continue
        pool_idx = group.index[group["clase"] == clase].tolist()
        rng.shuffle(pool_idx)
        parts.append(group.loc[pool_idx[:k]])
    sampled = pd.concat(parts) if parts else group.iloc[0:0]
    return sampled, n - len(sampled)


def generate_fleet(rng: random.Random) -> list[float]:
    n_trucks = rng.randint(*N_TRUCKS_RANGE)
    return [round(rng.uniform(*CAP_RANGE), 1) for _ in range(n_trucks)]


@dataclass
class ScenarioSummary:
    n_groups_total: int
    n_below_floor: int
    n_episodes_built: int


def build_and_label_episode(
    iso_year: int, iso_week: int, canton, group: pd.DataFrame, time_budget_s: float = 5.0
) -> tuple[dict, list[dict]]:
    """Build one episode (subsample + synthetic fleet) and label it.

    Returns (episode_record, vehicle_records) -- see build_scenarios.py for
    how these get assembled into the two output tables.
    """
    n_original = len(group)
    seed = episode_seed(iso_year, iso_week, canton)
    rng = random.Random(seed)  # one RNG stream per episode, consumed in order below

    sampled, n_excluded_subsample = stratified_subsample(group, MAX_N, rng)
    fleet = generate_fleet(rng)
    labeler_seed = rng.randrange(2**31)  # fresh draw, passed to assign_vehicles' own RNG

    vehicles = [Vehicle(uid=row.uid, clase=row.clase, cu=row.cu) for row in sampled.itertuples()]
    result = assign_vehicles(vehicles, fleet, time_budget_s=time_budget_s, seed=labeler_seed)

    eid = episode_id(iso_year, iso_week, canton)
    episode_record = {
        "episode_id": eid,
        "iso_year": iso_year,
        "iso_week": iso_week,
        "canton": canton,
        "n_original": n_original,
        "n_sampled": len(sampled),
        "n_excluded_subsample": n_excluded_subsample,
        "n_trucks": len(fleet),
        "truck_capacities": fleet,
        "n_loaded": result.n_loaded,
        "n_deferred": result.n_deferred,
        "cu_utilized": result.cu_utilized,
        "optimal": result.optimal,
        "search_time_ms": result.search_time_ms,
        "nodes_explored": result.nodes_explored,
        "seed": seed,
    }

    vehicle_records = []
    for row in sampled.itertuples():
        truck = result.assignment[row.uid]
        vehicle_records.append({
            "episode_id": eid,
            "uid": row.uid,
            "codigo_vehiculo": row.codigo_vehiculo,
            "clase": row.clase,
            "cu": row.cu,
            "canton": canton,
            "truck": truck,
            "loaded": truck != "SIN_CAMION",
        })

    return episode_record, vehicle_records


def build_all_episodes(
    df: pd.DataFrame, limit: int | None = None, time_budget_s: float = 5.0
) -> tuple[pd.DataFrame, pd.DataFrame, ScenarioSummary]:
    """Group the full feature dataset into episodes and label every one.

    `limit`: stop after this many episodes -- for quick local testing, since
    the full run is ~35k episodes (~30 min, see 06_feasibility.md).
    """
    groups = df.groupby(["iso_year", "iso_week", "canton"], sort=True)
    n_groups_total, n_below_floor = 0, 0
    episode_records, vehicle_records = [], []

    for (iso_year, iso_week, canton), group in groups:
        n_groups_total += 1
        if len(group) < FLOOR_N:
            n_below_floor += 1
            continue
        ep, vehs = build_and_label_episode(iso_year, iso_week, canton, group, time_budget_s)
        episode_records.append(ep)
        vehicle_records.extend(vehs)
        if limit and len(episode_records) >= limit:
            break

    episodes_df = pd.DataFrame(episode_records)
    vehicles_df = pd.DataFrame(vehicle_records)
    summary = ScenarioSummary(
        n_groups_total=n_groups_total,
        n_below_floor=n_below_floor,
        n_episodes_built=len(episode_records),
    )
    return episodes_df, vehicles_df, summary
