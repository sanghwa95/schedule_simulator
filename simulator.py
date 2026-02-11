# simulator.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ---------------------------
# Activity model
# ---------------------------
@dataclass(frozen=True)
class Activity:
    name: str
    min_h: float
    max_h: float
    value_per_h: float
    fatigue_per_h: float


# ---------------------------
# Time helpers
# ---------------------------
def hhmm(m: int) -> str:
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


# ---------------------------
# Fixed schedule constants
# ---------------------------
WORK_START = 9 * 60
WORK_END = 18 * 60

BREAKFAST_START, BREAKFAST_END = 8 * 60, 9 * 60
LUNCH_START, LUNCH_END = 12 * 60, 13 * 60
DINNER_START, DINNER_END = 18 * 60, 19 * 60


# ---------------------------
# Balanced-only knobs
# ---------------------------
OT_UTILITY_PER_H = 0.4
OT_FATIGUE_PER_H = 12.0


# ---------------------------
# Grid utilities
# ---------------------------
def mark(grid: List[str], s: int, e: int, label: str, slot: int) -> None:
    """Mark [s, e) with label in slot-sized grid."""
    for i in range(s // slot, (e + slot - 1) // slot):
        if 0 <= i < len(grid):
            grid[i] = label


def compress(grid: List[str], slot: int) -> List[Tuple[int, int, str]]:
    """Convert slot grid into blocks, skipping FREE."""
    blocks: List[Tuple[int, int, str]] = []
    if not grid:
        return blocks

    cur = grid[0]
    start = 0
    for i in range(1, len(grid) + 1):
        if i == len(grid) or grid[i] != cur:
            if cur != "FREE":
                blocks.append((start * slot, i * slot, cur))
            if i < len(grid):
                cur = grid[i]
                start = i
    return blocks


def fill_free_slots(grid: List[str], alloc_slots: Dict[str, int], rng: random.Random) -> None:
    """
    Place activities into FREE slots.
    Larger activities first, prefer contiguous blocks; otherwise scatter.

    FIX: remove "early-time bias":
    - When multiple contiguous FREE blocks exist, choose one uniformly at random.
    - When scattering, shuffle FREE indices before filling.
    """

    def free_indices() -> List[int]:
        return [i for i, v in enumerate(grid) if v == "FREE"]

    def find_blocks(length: int) -> List[int]:
        """
        Return all possible start indices of contiguous FREE runs with >= length.
        Example: FREE FREE FREE FREE and length=2 -> starts at 0,1,2
        """
        free = set(free_indices())
        starts: List[int] = []

        i = 0
        n = len(grid)
        while i < n:
            if i in free:
                j = i
                while j < n and j in free:
                    j += 1
                run_len = j - i
                if run_len >= length:
                    for s in range(i, j - length + 1):
                        starts.append(s)
                i = j
            else:
                i += 1
        return starts

    # Larger allocations first
    for name, slots in sorted(alloc_slots.items(), key=lambda x: -x[1]):
        if slots <= 0:
            continue

        candidates = find_blocks(slots)
        start = rng.choice(candidates) if candidates else None

        if start is not None:
            for i in range(start, start + slots):
                if 0 <= i < len(grid) and grid[i] == "FREE":
                    grid[i] = name
        else:
            free = [i for i, v in enumerate(grid) if v == "FREE"]
            rng.shuffle(free)
            for i in free[:slots]:
                grid[i] = name


# ---------------------------
# Sleep -> utility & fatigue modifiers
# ---------------------------
def productivity_multiplier(sleep_h: float) -> float:
    """
    All non-money utility drops with less sleep.
    8h -> 1.00
    Clamp to [0.35, 1.05]
    """
    if sleep_h >= 8:
        return min(1.05, 1.0 + (sleep_h - 8) * 0.02)

    d = 8 - sleep_h
    m = 1.0 - 0.12 * d - 0.04 * (d ** 2)
    return max(0.35, m)


def fatigue_multiplier(sleep_h: float) -> float:
    """
    Sleep-deprived fatigue amplification for ALL activities.
    8h -> 1.00
    """
    if sleep_h >= 8:
        return 1.0
    d = 8 - sleep_h
    return 1.0 + 0.22 * d + 0.04 * (d ** 2)


# ---------------------------
# Fatigue & score
# ---------------------------
def sleep_fatigue(sleep_h: float) -> float:
    """ADD to fatigue (positive = worse). Target sleep = 8h."""
    if sleep_h < 8:
        d = 8 - sleep_h
        return d * 12.0 + (d ** 2) * 5.0
    return -min(5.0, (sleep_h - 8) * 1.5)


def fatigue_penalty(f: float) -> float:
    """Penalty curve: mild under threshold, non-linear beyond."""
    if f > 30:
        return (f - 30) ** 1.2
    return max(0.0, f) * 0.18


def score_day(
    alloc: Dict[str, float],
    activities: Dict[str, Activity],
    sleep_h: float,
    overtime_min: int,
    ot_rate: int,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns:
      score, money, value, fatigue, penalty, prod, fat_mult
    """
    overtime_h = overtime_min / 60.0

    # money is real cash: fixed by rate, NOT sleep-adjusted
    money = overtime_h * ot_rate

    # sleep-dependent modifiers applied to ALL non-money utility + all activity fatigue
    prod = productivity_multiplier(sleep_h)
    fat_mult = fatigue_multiplier(sleep_h)

    value = 0.0
    fatigue = 0.0

    for name, h in alloc.items():
        a = activities[name]
        value += (a.value_per_h * h) * prod
        fatigue += (a.fatigue_per_h * h) * fat_mult

    # overtime non-money utility & fatigue (money already counted separately)
    if overtime_h > 0:
        value += (OT_UTILITY_PER_H * overtime_h) * prod
        fatigue += (OT_FATIGUE_PER_H * overtime_h) * fat_mult

    fatigue += sleep_fatigue(sleep_h)
    penalty = fatigue_penalty(fatigue)

    score = money / 10000.0 + value - penalty
    return score, money, value, fatigue, penalty, prod, fat_mult


def schedule_signature(
    wake: int,
    overtime_min: int,
    alloc: Dict[str, float],
    blocks: List[Tuple[int, int, str]],
) -> str:
    alloc_key = tuple(sorted((k, round(v, 2)) for k, v in alloc.items() if v > 0))
    blocks_key = tuple((s, e, lab) for s, e, lab in blocks)
    return str((wake, overtime_min, alloc_key, blocks_key))


# ---------------------------
# Public API for FastAPI
# ---------------------------
def run_simulation(
    iters: int = 10000,
    seed: int = 1,
    top: int = 5,
    wake_min: int = 360,
    wake_max: int = 540,
    ot_rate: int = 30000,
    ot_max: int = 4,
    slot: int = 30,
    scatter_sample_n: int = 800,
    hist_bins: int = 20,
) -> Dict[str, Any]:
    """
    Web-friendly wrapper: returns JSON-serializable dict.
    - Keeps your original behavior (breakfast always appears by clamping wake_max <= 08:00).
    - Removes all prints.
    - Adds stats for charts:
      (C) score histogram
      (D) scatter sample (money vs fatigue)
    """

    rng = random.Random(seed)

    activities: Dict[str, Activity] = {
        "study": Activity("study", 0, 6, 2.5, 2.3),
        "hobby": Activity("hobby", 0, 3, 2.0, 0.3),
        "exercise": Activity("exercise", 0, 2, 2.5, -1.0),
        "rest": Activity("rest", 0, 4, 1.6, -0.6),
    }

    # clamp wake-max so breakfast always appears (wake-up <= 08:00)
    effective_wake_max = min(wake_max, BREAKFAST_START)
    if effective_wake_max < wake_min:
        effective_wake_max = wake_min  # degenerate range fallback

    best_rows: List[
        Tuple[
            float,  # score
            int,  # wake
            Dict[str, float],  # alloc
            float,  # money
            float,  # value
            float,  # fatigue
            float,  # penalty
            float,  # prod
            float,  # fat_mult
            List[Tuple[int, int, str]],  # blocks
        ]
    ] = []

    seen = set()
    grid_len = 24 * 60 // slot

    # (D) reservoir samples for scatter: (money, fatigue, score)
    scatter_samples: List[Tuple[float, float, float]] = []
    stream_i = 0  # how many items have been seen in the stream

    for _ in range(iters):
        wake_up = rng.randint(wake_min, effective_wake_max)
        sleep_h = wake_up / 60.0

        # overtime: ALWAYS in 1-hour blocks (0..ot_max)
        ot_hours = rng.randint(0, ot_max)
        overtime_min = ot_hours * 60

        # build grid
        grid = ["FREE"] * grid_len

        # 1) work first
        mark(grid, WORK_START, WORK_END, "work", slot)

        # 2) meals override work where they overlap
        mark(grid, BREAKFAST_START, BREAKFAST_END, "breakfast", slot)
        mark(grid, LUNCH_START, LUNCH_END, "lunch", slot)
        mark(grid, DINNER_START, DINNER_END, "dinner", slot)

        # 3) sleep overrides everything before wake-up
        mark(grid, 0, wake_up, "sleep", slot)

        # 4) overtime after dinner
        mark(grid, DINNER_END, min(24 * 60, DINNER_END + overtime_min), "overtime work", slot)

        # available FREE time (slot units -> convert to 0.5h units)
        free_slots = sum(1 for v in grid if v == "FREE")
        free_half_hours = free_slots
        if slot != 30:
            free_half_hours = int((free_slots * slot) // 30)

        # allocate time in 0.5h units
        alloc: Dict[str, float] = {k: 0.0 for k in activities}
        remaining = free_half_hours
        keys = list(activities.keys())

        while remaining > 0:
            k = rng.choice(keys)
            if alloc[k] + 0.5 <= activities[k].max_h:
                alloc[k] += 0.5
                remaining -= 1
            else:
                if all(alloc[name] + 0.5 > activities[name].max_h for name in keys):
                    break

        # place allocated activities into FREE slots (fixed: remove early bias)
        alloc_slots: Dict[str, int] = {k: int(round((h * 60) / slot)) for k, h in alloc.items()}
        fill_free_slots(grid, alloc_slots, rng)

        # evaluate
        score, money, value, fatigue, penalty, prod, fat_mult = score_day(
            alloc, activities, sleep_h, overtime_min, ot_rate
        )

        # ---- (D) 그래프용 샘플 저장 (reservoir sampling) ----
        if scatter_sample_n > 0:
            if len(scatter_samples) < scatter_sample_n:
                scatter_samples.append((money, fatigue, score, sleep_h))
            else:
                # stream_i is current index (0-based) of the stream
                j = rng.randint(0, stream_i)
                if j < scatter_sample_n:
                    scatter_samples[j] = (money, fatigue, score, sleep_h)
        stream_i += 1

        blocks = compress(grid, slot)
        sig = schedule_signature(wake_up, overtime_min, alloc, blocks)
        if sig in seen:
            continue
        seen.add(sig)

        best_rows.append(
            (score, wake_up, alloc, money, value, fatigue, penalty, prod, fat_mult, blocks)
        )

    best_rows.sort(key=lambda x: x[0], reverse=True)

    # Build JSON helpers
    def blocks_to_json(blocks: List[Tuple[int, int, str]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s, e, label in blocks:
            out.append(
                {
                    "start_min": s,
                    "end_min": e,
                    "start": hhmm(s),
                    "end": hhmm(e),
                    "label": label,
                }
            )
        return out

    def row_to_json(row) -> Dict[str, Any]:
        score, wake, alloc, money, value, fatigue, penalty, prod, fat_mult, blocks = row
        sleep_h = wake / 60.0
        return {
            "score": round(score, 4),
            "wake_min": wake,
            "wake": hhmm(wake),
            "sleep_h": round(sleep_h, 3),
            "money_won": int(round(money)),
            "value": round(value, 4),
            "fatigue": round(fatigue, 4),
            "penalty": round(penalty, 4),
            "multipliers": {"utility": round(prod, 4), "fatigue": round(fat_mult, 4)},
            "alloc_hours": {k: round(v, 2) for k, v in alloc.items() if v > 0},
            "timeline": blocks_to_json(blocks),
        }

    # ---- (C) Score 히스토그램 계산 (정렬 후, return 전) ----
    scores_all = [r[0] for r in best_rows]
    if scores_all:
        s_min = min(scores_all)
        s_max = max(scores_all)
    else:
        s_min, s_max = 0.0, 1.0

    bins = max(5, int(hist_bins))
    if s_max == s_min:
        s_max = s_min + 1e-6

    bin_w = (s_max - s_min) / bins
    edges = [s_min + i * bin_w for i in range(bins + 1)]
    counts = [0] * bins

    for s in scores_all:
        idx = int((s - s_min) / bin_w)
        if idx == bins:  # max edge
            idx = bins - 1
        counts[idx] += 1

    hist = {
        "min": s_min,
        "max": s_max,
        "bins": bins,
        "edges": edges,
        "counts": counts,
    }

    # ---- (D) scatter JSON ----
    scatter = [
        {"money": float(m), "fatigue": float(f), "score": float(s), "sleep_h": float(sh),}
        for (m, f, s, sh) in scatter_samples
    ]

    top_n = min(top, len(best_rows))
    worst_rows = list(reversed(best_rows[-top_n:])) if top_n > 0 else []

    return {
        "settings": {
            "iters": iters,
            "seed": seed,
            "top": top,
            "slot_min": slot,
            "wake_min": wake_min,
            "wake_max": wake_max,
            "effective_wake_max": effective_wake_max,
            "work": {"start": hhmm(WORK_START), "end": hhmm(WORK_END)},
            "meals": {
                "breakfast": {"start": hhmm(BREAKFAST_START), "end": hhmm(BREAKFAST_END)},
                "lunch": {"start": hhmm(LUNCH_START), "end": hhmm(LUNCH_END)},
                "dinner": {"start": hhmm(DINNER_START), "end": hhmm(DINNER_END)},
            },
            "overtime": {"rate_won_per_h": ot_rate, "max_h": ot_max, "scheduled_after": hhmm(DINNER_END)},
            "ot_non_money": {"utility_per_h": OT_UTILITY_PER_H, "fatigue_per_h": OT_FATIGUE_PER_H},
            "charts": {"scatter_sample_n": scatter_sample_n, "hist_bins": hist_bins},
        },
        "best": [row_to_json(r) for r in best_rows[:top_n]],
        "worst": [row_to_json(r) for r in worst_rows],
        "stats": {
            "score_hist": hist,
            "money_fatigue_scatter": scatter,
        },
        "meta": {
            "unique_schedules": len(best_rows),
            "note": "Schedules are deduplicated by (wake, overtime, alloc, blocks).",
        },
    }

