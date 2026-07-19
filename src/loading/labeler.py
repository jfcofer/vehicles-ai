"""
src/loading/labeler.py

Exhaustive (exact) teacher labeler for the vehicle-to-truck loading problem.

Scope decision (see reports/03_proposals/fleet_routing/02_scope.md):
No external solver (OR-Tools, PuLP, MIP libraries, etc.) is used. This is a
from-scratch exact search over the assignment space, written and owned by
the team.

Problem
-------
Given N vehicles (each with a CU weight, determined by its CLASE) and n
trucks (each with a capacity, possibly different per truck and per episode),
assign each vehicle to exactly one truck or to "no truck" (deferred), such
that no truck's total assigned CU exceeds its capacity, maximizing -- in
this strict priority order:

    1. number of vehicles loaded (not deferred)
    2. total CU utilized across all trucks (tie-breaker among solutions
       that load the same number of vehicles)

This mirrors the objective stated in the course project (Tarea #3, Sec. I):
maximize vehicles transported, then maximize space utilization.

Design notes
------------
- CU values are scaled to integers via exact fractions (fractions.Fraction),
  using the LCM of all denominators present in a given episode. A *fixed*
  decimal scale (e.g. always x10) silently corrupts repeating fractions like
  2/3 (round(2/3 * 10) = 7, but 12 * 7 != 12 * (2/3) * 10) -- this produced a
  wrong, non-obviously-wrong answer during development and is exactly the
  kind of bug that must not exist in the label source of truth. Fractions
  make the integer scaling exact regardless of which CU values are chosen.

- Search space: vehicles are grouped by CLASE first. Within a class every
  vehicle has the same CU, so for the *optimization* they are interchange-
  able -- CANTON and other per-vehicle attributes do not affect the capacity
  constraint (loading-only scope, no routing). This collapses the search
  from "which of N individual vehicles" (permutation-sized) to "how many of
  each of the (few) classes" (count-tuple-sized), which is the same
  simplification the PDF itself uses in Sec. I (solving for S and D counts,
  not for which specific SUV/Sedan). The state space is then the tuple of
  remaining counts per class, memoized across trucks -- an exact dynamic
  program, not a heuristic.
  Concretely, for 17 vehicles across 4 classes this cut search time from
  ~1.3s / ~2M nodes (item-level branch-and-bound) to a few milliseconds.
  Individual vehicles within a class are re-attached to their truck
  deterministically (sorted by uid) after the class-count solution is found.

- A wall-clock time budget bounds worst-case runtime for batch labeling over
  many historical weeks. If the budget is hit, the best assignment found so
  far is returned with `optimal=False`, so downstream code can filter or
  flag non-certified labels instead of silently trusting them.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Vehicle:
    uid: str
    clase: str
    cu: float | Fraction  # e.g. 1.4, or Fraction(2, 3) for exactness


def _as_fraction(x: float | Fraction) -> Fraction:
    """Snap a float CU to the nearest simple fraction (denominator <= 1000).

    Pass a Fraction directly (e.g. Fraction(2, 3)) when the value is not
    exactly representable in decimal, to skip the snapping heuristic
    entirely.
    """
    if isinstance(x, Fraction):
        return x
    return Fraction(x).limit_denominator(1000)


@dataclass
class LabelResult:
    assignment: dict[str, str]  # uid -> "CAMION_1" | ... | "SIN_CAMION"
    n_loaded: int
    n_deferred: int
    cu_utilized: float
    truck_capacities: list[float]
    truck_loads: list[float]
    optimal: bool  # False if the time budget was hit before proving optimality
    search_time_ms: float
    nodes_explored: int


def assign_vehicles(
    vehicles: list[Vehicle],
    truck_capacities: list[float],
    time_budget_s: float = 5.0,
    seed: int | None = None,
) -> LabelResult:
    """Exact (or best-effort, time-bounded) assignment of vehicles to trucks.

    `seed`: within a class, which specific vehicles get the "loaded" slots
    is arbitrary -- they're interchangeable for the optimizer (same CU).
    With `seed=None` (default), ties break deterministically by uid, which
    is convenient for unit tests but means the SAME vehicles always get
    excluded whenever a similar class-mix recurs -- a spurious, learnable
    pattern for a model trained on this output (see scenarios.py). Pass a
    seed (e.g. derived from the episode key) when generating training data,
    so tie-breaking varies across episodes without sacrificing
    reproducibility -- the same seed always gives the same result.
    """
    start = time.perf_counter()
    deadline = start + time_budget_s
    nodes = 0
    timed_out = False

    n = len(vehicles)
    n_trucks = len(truck_capacities)

    if n == 0 or n_trucks == 0:
        return LabelResult(
            assignment={v.uid: "SIN_CAMION" for v in vehicles},
            n_loaded=0,
            n_deferred=n,
            cu_utilized=0.0,
            truck_capacities=truck_capacities,
            truck_loads=[0.0] * n_trucks,
            optimal=True,
            search_time_ms=(time.perf_counter() - start) * 1000,
            nodes_explored=0,
        )

    # --- Group by class: vehicles of the same class share the same CU and
    # are interchangeable for the optimizer (see module docstring). --------
    by_class: dict[str, list[Vehicle]] = defaultdict(list)
    for v in vehicles:
        by_class[v.clase].append(v)
    classes = sorted(by_class.keys())
    k = len(classes)
    counts = tuple(len(by_class[c]) for c in classes)

    cu_fracs = [_as_fraction(by_class[c][0].cu) for c in classes]
    cap_fracs = [_as_fraction(c) for c in truck_capacities]

    # Exact integer scale: LCM of every denominator present this episode.
    denom = 1
    for f in cu_fracs + cap_fracs:
        denom = math.lcm(denom, f.denominator)

    def _scale(f: Fraction) -> int:
        scaled = f * denom
        assert scaled.denominator == 1, "internal scaling error"
        return scaled.numerator

    cu_scaled = [_scale(f) for f in cu_fracs]
    cap_scaled = [_scale(f) for f in cap_fracs]

    # --- Exact DP over (truck_index, remaining_counts_per_class). ----------
    memo: dict[tuple[int, tuple[int, ...]], tuple[int, int, list[tuple[int, ...]]]] = {}

    def enumerate_loadouts(remaining: tuple[int, ...], capacity: int):
        """All feasible (loadout, count, cu) combos for one truck.

        loadout[i] = how many of classes[i] this truck takes. Bounded by
        `remaining` (what's left to assign) and `capacity` (truck's CU).
        """
        loadout = [0] * k

        def rec(i: int, cap_left: int):
            if i == k:
                yield tuple(loadout), sum(loadout), capacity - cap_left
                return
            max_x = remaining[i]
            if cu_scaled[i] > 0:
                max_x = min(max_x, cap_left // cu_scaled[i])
            for x in range(max_x, -1, -1):
                loadout[i] = x
                yield from rec(i + 1, cap_left - x * cu_scaled[i])
            loadout[i] = 0

        yield from rec(0, capacity)

    def solve(truck_idx: int, remaining: tuple[int, ...]):
        nonlocal nodes, timed_out
        if timed_out or truck_idx == n_trucks or sum(remaining) == 0:
            return 0, 0, []

        key = (truck_idx, remaining)
        cached = memo.get(key)
        if cached is not None:
            return cached

        best_loaded, best_cu, best_loadouts = 0, 0, []
        for loadout, cnt, cu in enumerate_loadouts(remaining, cap_scaled[truck_idx]):
            nodes += 1
            if nodes % 4096 == 0 and time.perf_counter() > deadline:
                timed_out = True
                break

            new_remaining = tuple(r - x for r, x in zip(remaining, loadout))
            sub_loaded, sub_cu, sub_loadouts = solve(truck_idx + 1, new_remaining)
            total_loaded, total_cu = cnt + sub_loaded, cu + sub_cu
            if (total_loaded, total_cu) > (best_loaded, best_cu):
                best_loaded, best_cu = total_loaded, total_cu
                best_loadouts = [loadout] + sub_loadouts

        memo[key] = (best_loaded, best_cu, best_loadouts)
        return memo[key]

    n_loaded, cu_used, per_truck_loadouts = solve(0, counts)

    # --- Re-attach specific vehicle uids. -----------------------------------
    # Deterministic (sorted by uid) by default; seeded shuffle when `seed` is
    # given -- see docstring. Either way, order is fixed before popping, so
    # results are reproducible for a given call.
    if seed is None:
        queues = {c: sorted(by_class[c], key=lambda v: v.uid) for c in classes}
    else:
        rng = random.Random(seed)
        queues = {}
        for c in classes:
            shuffled = sorted(by_class[c], key=lambda v: v.uid)  # stable base order first
            rng.shuffle(shuffled)
            queues[c] = shuffled
    result_assign: dict[str, str] = {}
    truck_loads_scaled = [0] * n_trucks
    for truck_idx, loadout in enumerate(per_truck_loadouts):
        for ci, x in enumerate(loadout):
            for _ in range(x):
                v = queues[classes[ci]].pop(0)
                result_assign[v.uid] = f"CAMION_{truck_idx + 1}"
                truck_loads_scaled[truck_idx] += cu_scaled[ci]
    for c in classes:
        for v in queues[c]:
            result_assign[v.uid] = "SIN_CAMION"

    elapsed_ms = (time.perf_counter() - start) * 1000
    return LabelResult(
        assignment=result_assign,
        n_loaded=n_loaded,
        n_deferred=n - n_loaded,
        cu_utilized=cu_used / denom,
        truck_capacities=truck_capacities,
        truck_loads=[t / denom for t in truck_loads_scaled],
        optimal=not timed_out,
        search_time_ms=elapsed_ms,
        nodes_explored=nodes,
    )
