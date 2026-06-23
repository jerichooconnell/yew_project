"""
Social Golfer Problem Solver — 31-player extension
=====================================================

Strategy (two-phase)
---------------------
Phase 1 — Solve SGP(30, 5, 6):
    30 players, 6 groups of 5, 6 rounds, no pair shares a group twice.
    Uses Iterated Local Search with a Numba-JIT simulated-annealing kernel
    (~50 M iterations at near-C speed).  Typically finds the perfect solution
    in one or two restarts (~90 s total, including JIT warm-up).

Phase 2 — Transversal extension:
    Find a "system of distinct representatives" (SDR): one group per round
    from the 30-player schedule, with all 6 chosen groups pairwise disjoint
    (together they cover every one of the 30 players exactly once).
    Add player P31 to those groups, enlarging each to size 6.

    Result: a perfect 31-player schedule — 1 group of 6 and 5 groups of 5
    per round, 6 rounds, no pair of players shares a group more than once.

Why this works
--------------
After the transversal:
  * The 30 pairwise relationships are unchanged (the 30-player schedule was
    already perfect).
  * P31 meets the 5 members of the chosen group in each round.  Because the
    6 chosen groups are pairwise disjoint, P31 sees every one of the 30 other
    players exactly once.
  * Total pair-meetings: 435 (original) + 30 (P31's) = 465? No:
      original 30-player meetings (pairs 0..29 that actually share a group):
        C(30,2) = 435 pairs total; a perfect SGP(30,5,6) has 6 × C(5,2) × 6
        = 6×10×6 = 360 pairs meeting once, 75 never meeting.
      P31 meets: 6 × 5 = 30 distinct players (one per round, all distinct by
        the disjoint-group property) — 30 new pair-meetings.
      Grand total: 360 + 30 = 390; total pairs: C(31,2) = 465;
        75 + (30_never_P31) = 75 + 1 = 76? No wait: 465 - 390 = 75 pairs never
        meet.  From the 30-player side: 75 pairs (all among 0..29). P31 meets
        everyone, so 0 pairs involving P31 are missing → 75 total non-pairs.
    PERFECT: every pair-meeting exactly 1, 75 pairs not sharing a group.

Usage:
    python scripts/social_golfer.py [--seed SEED] [--timeout SECS] [--quiet]
"""

import argparse
import math
import sys
import time

import numpy as np
from numba import njit

# ─────────────────────────────────────────────────────────────────────────────
# Phase-1 constants  (30-player, uniform groups of 5)
# ─────────────────────────────────────────────────────────────────────────────
_N   = 30    # players
_K   = 5     # group size
_G   = 6     # groups per round
_R   = 6     # rounds
assert _N == _G * _K

# ALL_BUT[i, k] = k-th other slot in a group of size K for player at slot i.
# Shape (5, 4): for each slot i, the 4 other slot indices.
_ALL_BUT = np.array(
    [[j for j in range(_K) if j != i] for i in range(_K)],
    dtype=np.int64
)

# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT SA kernel  (30-player, uniform groups)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _jit_sa(schedule, gm, pos, meetings, all_but, n_iter, T0, alpha, jit_seed):
    """
    Simulated annealing for SGP(30, 5, 6).

    jit_seed: seed numba's PRNG at the start so each restart is reproducible.
    Arrays (modified in-place):
      schedule  (6, 30)  int8   group[r, p]
      gm        (6, 6, 5) int16  members[r, g, slot]
      pos       (6, 30)  int8   slot[r, p]
      meetings  (30, 30) int8   cumulative pair-meeting counts

    Returns best score observed; schedule contains the best-found solution.
    """
    np.random.seed(jit_seed)
    cur_score = 0
    for ii in range(30):
        for jj in range(ii + 1, 30):
            v = int(meetings[ii, jj]) - 1
            if v > 0:
                cur_score += v

    best_score = cur_score
    best_sched = schedule.copy()
    T = T0

    for _ in range(n_iter):
        r  = np.random.randint(0, 6)
        p1 = np.random.randint(0, 30)
        p2 = np.random.randint(0, 30)

        g1 = int(schedule[r, p1])
        g2 = int(schedule[r, p2])
        if g1 == g2:
            T *= alpha
            continue

        i1 = int(pos[r, p1])
        i2 = int(pos[r, p2])

        d = 0
        for k in range(4):
            q1 = int(gm[r, g1, all_but[i1, k]])
            if meetings[p1, q1] > 1: d -= 1
            if meetings[p2, q1] > 0: d += 1
        for k in range(4):
            q2 = int(gm[r, g2, all_but[i2, k]])
            if meetings[p2, q2] > 1: d -= 1
            if meetings[p1, q2] > 0: d += 1

        if d <= 0 or np.random.random() < math.exp(-d / T):
            for k in range(4):
                q1 = int(gm[r, g1, all_but[i1, k]])
                meetings[p1, q1] -= 1;  meetings[q1, p1] -= 1
                meetings[p2, q1] += 1;  meetings[q1, p2] += 1
            for k in range(4):
                q2 = int(gm[r, g2, all_but[i2, k]])
                meetings[p2, q2] -= 1;  meetings[q2, p2] -= 1
                meetings[p1, q2] += 1;  meetings[q2, p1] += 1
            gm[r, g1, i1] = p2;   gm[r, g2, i2] = p1
            pos[r, p1] = i2;      pos[r, p2] = i1
            schedule[r, p1] = g2; schedule[r, p2] = g1
            cur_score += d

            if cur_score < best_score:
                best_score = cur_score
                for ri in range(6):
                    for pi in range(30):
                        best_sched[ri, pi] = schedule[ri, pi]
                if best_score == 0:
                    break

        T *= alpha

    for ri in range(6):
        for pi in range(30):
            schedule[ri, pi] = best_sched[ri, pi]

    return best_score


# ─────────────────────────────────────────────────────────────────────────────
# Phase-1 helpers  (uniform 30-player schedule)
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild(schedule):
    gm  = np.zeros((_R, _G, _K), dtype=np.int16)
    pos = np.zeros((_R, _N),     dtype=np.int8)
    for r in range(_R):
        for g in range(_G):
            members = np.where(schedule[r] == g)[0]
            gm[r, g] = members
            for slot, p in enumerate(members):
                pos[r, int(p)] = slot
    meetings = np.zeros((_N, _N), dtype=np.int8)
    for r in range(_R):
        for g in range(_G):
            m = gm[r, g]
            meetings[np.ix_(m, m)] += 1
    np.fill_diagonal(meetings, 0)
    return gm, pos, meetings


def _score(meetings):
    return int(np.sum(np.triu(np.maximum(meetings - 1, 0), k=1)))


def _greedy_init(rng):
    schedule = np.full((_R, _N), -1, dtype=np.int8)
    gm       = np.zeros((_R, _G, _K), dtype=np.int16)
    pos      = np.zeros((_R, _N), dtype=np.int8)
    sizes    = np.zeros((_R, _G), dtype=np.int8)
    meetings = np.zeros((_N, _N), dtype=np.int8)
    for r in range(_R):
        for p in rng.permutation(_N).tolist():
            best_g, best_c = -1, _N + 1
            for g in rng.permutation(_G).tolist():
                if sizes[r, g] >= _K:
                    continue
                sz = int(sizes[r, g])
                c  = int(np.sum(meetings[p, gm[r, g, :sz]] >= 1)) if sz else 0
                if c < best_c:
                    best_c, best_g = c, g
                if c == 0:
                    break
            sz = int(sizes[r, best_g])
            gm[r, best_g, sz]  = p
            pos[r, p]          = sz
            sizes[r, best_g]  += 1
            schedule[r, p]     = best_g
        for g in range(_G):
            m = gm[r, g]
            meetings[np.ix_(m, m)] += 1
        np.fill_diagonal(meetings, 0)
    return schedule, gm, pos, meetings, _score(meetings)


def _best_swap_in_round(r, schedule, gm, pos, meetings):
    g  = schedule[r]
    A  = (meetings > 0).astype(np.int32)
    B  = (meetings > 1).astype(np.int32)
    G  = np.zeros((_N, _G), dtype=np.int32)
    G[np.arange(_N), g] = 1
    H_A = A @ G
    H_B = B @ G
    F   = H_B[np.arange(_N), g]
    H_p = H_A[:, g]
    delta = H_p + H_p.T - 2 * A - F[:, None] - F[None, :]
    ut = np.triu(g[:, None] != g[None, :], k=1)
    if not np.any(delta[ut] < 0):
        return 0
    d_masked = np.where(ut, delta, 1)
    flat     = int(d_masked.argmin())
    p1, p2   = divmod(flat, _N)
    g1 = int(g[p1]);   g2 = int(g[p2])
    i1 = int(pos[r, p1]);   i2 = int(pos[r, p2])
    o1 = gm[r, g1];  o1 = o1[o1 != p1]
    o2 = gm[r, g2];  o2 = o2[o2 != p2]
    meetings[p1, o1] -= 1;  meetings[o1, p1] -= 1
    meetings[p2, o2] -= 1;  meetings[o2, p2] -= 1
    meetings[p1, o2] += 1;  meetings[o2, p1] += 1
    meetings[p2, o1] += 1;  meetings[o1, p2] += 1
    gm[r, g1, i1] = p2;  gm[r, g2, i2] = p1
    pos[r, p1] = i2;     pos[r, p2] = i1
    schedule[r, p1] = g2; schedule[r, p2] = g1
    return int(delta[p1, p2])


def _hill_climb(schedule, gm, pos, meetings, score, max_idle=80):
    idle = 0
    while score > 0 and idle < max_idle:
        improved = False
        for r in range(_R):
            d = _best_swap_in_round(r, schedule, gm, pos, meetings)
            if d < 0:
                score += d;  improved = True;  idle = 0
                if score == 0:
                    return 0
        if not improved:
            idle += 1
    return score


def _ils_kick(schedule, gm, pos, meetings, rng):
    round_excess = np.zeros(_R, dtype=np.int32)
    for r in range(_R):
        for g in range(_G):
            m = gm[r, g]
            for i in range(_K):
                for j in range(i + 1, _K):
                    v = int(meetings[int(m[i]), int(m[j])]) - 1
                    if v > 0:
                        round_excess[r] += v
    wr = int(round_excess.argmax())
    for g in range(_G):
        m = gm[wr, g]
        for i in range(_K):
            for j in range(_K):
                if i != j:
                    meetings[int(m[i]), int(m[j])] -= 1
    np.fill_diagonal(meetings, 0)
    players = rng.permutation(_N)
    new_grp = np.empty(_N, dtype=np.int8)
    for g in range(_G):
        for s in range(_K):
            new_grp[players[g * _K + s]] = g
    for p in range(_N):
        schedule[wr, p] = new_grp[p]
    for g in range(_G):
        members = np.where(schedule[wr] == g)[0]
        gm[wr, g] = members
        for slot, p in enumerate(members):
            pos[wr, int(p)] = slot
    for g in range(_G):
        m = gm[wr, g]
        meetings[np.ix_(m, m)] += 1
    np.fill_diagonal(meetings, 0)
    return _score(meetings)


def _ils_kick_2round(schedule, gm, pos, meetings, rng):
    """Reshuffle the two worst rounds simultaneously for a larger perturbation."""
    round_excess = np.zeros(_R, dtype=np.int32)
    for r in range(_R):
        for g in range(_G):
            m = gm[r, g]
            for i in range(_K):
                for j in range(i + 1, _K):
                    v = int(meetings[int(m[i]), int(m[j])]) - 1
                    if v > 0:
                        round_excess[r] += v
    worst_2 = np.argsort(round_excess)[-2:]
    for wr in worst_2:
        for g in range(_G):
            m = gm[wr, g]
            for i in range(_K):
                for j in range(_K):
                    if i != j:
                        meetings[int(m[i]), int(m[j])] -= 1
    np.fill_diagonal(meetings, 0)
    for wr in worst_2:
        players = rng.permutation(_N)
        new_grp = np.empty(_N, dtype=np.int8)
        for g in range(_G):
            for s in range(_K):
                new_grp[players[g * _K + s]] = g
        for p in range(_N):
            schedule[wr, p] = new_grp[p]
        for g in range(_G):
            members = np.where(schedule[wr] == g)[0]
            gm[wr, g] = members
            for slot, p in enumerate(members):
                pos[wr, int(p)] = slot
    for wr in worst_2:
        for g in range(_G):
            m = gm[wr, g]
            meetings[np.ix_(m, m)] += 1
    np.fill_diagonal(meetings, 0)
    return _score(meetings)


# SA hyper-params
_SA_N_ITER  = 50_000_000
_SA_T0      = 5.0
_SA_T_FINAL = 0.3
_SA_ALPHA   = math.exp(math.log(_SA_T_FINAL / _SA_T0) / _SA_N_ITER)


def _warm_up_jit():
    s  = np.zeros((_R, _N), dtype=np.int8)
    gm = np.zeros((_R, _G, _K), dtype=np.int16)
    p  = np.zeros((_R, _N), dtype=np.int8)
    m  = np.zeros((_N, _N), dtype=np.int8)
    for r in range(_R):
        for g in range(_G):
            for sl in range(_K):
                pl = g * _K + sl
                s[r, pl] = g;  gm[r, g, sl] = pl;  p[r, pl] = sl
    _jit_sa(s, gm, p, m, _ALL_BUT, 1, 0.1, 0.99, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Phase-1 solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_30(timeout: float, max_restarts: int, seed, verbose: bool):
    """Solve SGP(30, 5, 6).  Returns (schedule_30, score)."""
    rng = np.random.default_rng(seed)

    t0 = time.time()
    best_sched = None
    best_score = sys.maxsize

    for restart in range(max_restarts):
        if time.time() - t0 > timeout:
            break

        jit_seed = restart * 7919 + 1  # unique, deterministic per restart
        schedule, gm, pos, meetings, init_score = _greedy_init(rng)
        _jit_sa(schedule, gm, pos, meetings, _ALL_BUT,
                _SA_N_ITER, _SA_T0, _SA_ALPHA, jit_seed)
        gm, pos, meetings = _rebuild(schedule)
        sa_score   = _score(meetings)
        hc_score   = _hill_climb(schedule, gm, pos, meetings, sa_score)

        best_local = schedule.copy()
        best_ls    = hc_score

        for kick_idx in range(400):
            if best_ls == 0:
                break
            schedule[:] = best_local
            gm, pos, meetings = _rebuild(schedule)
            # 30% chance of a larger 2-round kick for diversity
            if kick_idx % 10 >= 7:
                kicked = _ils_kick_2round(schedule, gm, pos, meetings, rng)
            else:
                kicked = _ils_kick(schedule, gm, pos, meetings, rng)
            after  = _hill_climb(schedule, gm, pos, meetings, kicked)
            if after < best_ls:
                best_ls    = after
                best_local = schedule.copy()

        schedule = best_local
        final    = best_ls

        if verbose:
            print(f"  [30-player] Restart {restart + 1:3d}: "
                  f"init={init_score:3d}  SA={sa_score:3d}  "
                  f"HC={hc_score:3d}  ILS={final}  "
                  f"[{time.time() - t0:.1f}s]")

        if final < best_score:
            best_score = final
            best_sched = schedule.copy()

        if best_score == 0:
            if verbose:
                print(f"\n  *** 30-player perfect solution found on restart "
                      f"{restart + 1} ***\n")
            break

    return best_sched, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Phase-2: SDR (transversal) finder
# ─────────────────────────────────────────────────────────────────────────────

def find_sdr(schedule_30: np.ndarray):
    """
    Find a system of distinct representatives (SDR) for the 30-player schedule:
    choose one group g_r per round r such that the 6 chosen groups are pairwise
    disjoint (together they cover all 30 players exactly once).

    Uses depth-first backtracking; typically instant for a perfect schedule.

    Returns a list [g_0, g_1, ..., g_5] of group indices, or None if not found.
    """
    # Pre-compute group members as frozensets for fast intersection tests.
    groups = [
        [frozenset(int(p) for p in np.where(schedule_30[r] == g)[0])
         for g in range(_G)]
        for r in range(_R)
    ]

    result = [None] * _R

    def bt(r: int, covered: frozenset) -> bool:
        if r == _R:
            return len(covered) == _N
        for g in range(_G):
            grp = groups[r][g]
            if grp.isdisjoint(covered):
                result[r] = g
                if bt(r + 1, covered | grp):
                    return True
        result[r] = None
        return False

    if bt(0, frozenset()):
        return list(result)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase-2: construct the 31-player schedule
# ─────────────────────────────────────────────────────────────────────────────

def construct_31(schedule_30: np.ndarray, sdr_groups: list) -> np.ndarray:
    """
    Build a 31-player schedule from the 30-player perfect schedule and the SDR.

    In each round r:
      - Players 0..29 keep their group assignments, but the group indices are
        remapped so that the SDR group becomes group 0 (the "big" group of 6).
      - Player 30 (P31) is assigned to group 0 in every round.

    Returns an array of shape (6, 31), dtype int8, group indices 0..5.
    """
    schedule_31 = np.zeros((_R, _N + 1), dtype=np.int8)
    for r in range(_R):
        g_big = sdr_groups[r]
        for p in range(_N):
            old_g = int(schedule_30[r, p])
            if old_g == g_big:
                new_g = 0
            elif old_g < g_big:
                new_g = old_g + 1
            else:
                new_g = old_g
            schedule_31[r, p] = new_g
        schedule_31[r, _N] = 0  # P31 always in group 0 (big group)
    return schedule_31


# ─────────────────────────────────────────────────────────────────────────────
# Participant names  (P01 → index 0, P31 → index 30)
# ─────────────────────────────────────────────────────────────────────────────
# 30 confirmed players + 1 "Extra" reserve slot (P31).
# Registration / roster numbers in the source list are ignored here; names are
# assigned sequentially in appearance order.
PLAYER_NAMES = [
    "Ellen",    # P01
    "Karen",    # P02
    "Sam",      # P03
    "Aaron",    # P04
    "Lana",     # P05
    "Anne",     # P06
    "Art",      # P07
    "Peter",    # P08
    "Esther",   # P09  (roster #10)
    "Joan",     # P10  (roster #12)
    "Dan O",    # P11  (roster #13)
    "Noah",     # P12  (roster #14)
    "Rachel G", # P13  (roster #15)
    "Will",     # P14  (roster #17)
    "Char",     # P15  (roster #18)
    "Jericho",  # P16  (roster #19)
    "Kirsten",  # P17  (roster #20)
    "Lilli",    # P18  (roster #21)
    "Solomon",  # P19  (roster #23)
    "Misty",    # P20  (roster #24)
    "Aja",      # P21  (roster #25)
    "Isobel",   # P22  (roster #26)
    "Shane",    # P23  (roster #27)
    "Sora",     # P24  (roster #28)
    "Isaac",    # P25  (roster #29)
    "Jordan",   # P26  (roster #30)
    "Dan S",    # P27  (roster #32)
    "Rachel",   # P28  (roster #33)
    "Sophia",   # P29  (roster #35)
    "Cathe",    # P30  (roster #36)
    "Extra",    # P31  — reserve slot
]
assert len(PLAYER_NAMES) == 31, "Name list must contain exactly 31 entries"

# Names that always appear first within their group in the printed output.
_PRIORITY_NAMES = {"Lilli", "Anne", "Will", "Aaron", "Sophia", "Karen"}


# ─────────────────────────────────────────────────────────────────────────────
# Validation & output  (31-player)
# ─────────────────────────────────────────────────────────────────────────────
_N31  = 31
_KBIG = 6
_KMIN = 5


def validate_31(schedule_31: np.ndarray) -> bool:
    n_players = schedule_31.shape[1]
    ok = True

    # --- structural checks ---
    for r in range(_R):
        sz0 = int(np.sum(schedule_31[r] == 0))
        if sz0 != _KBIG:
            print(f"  ERROR: Round {r+1}, Group 1 has {sz0} members (want {_KBIG})")
            ok = False
        for g in range(1, _G):
            sz = int(np.sum(schedule_31[r] == g))
            if sz != _KMIN:
                print(f"  ERROR: Round {r+1}, Group {g+1} has {sz} members (want {_KMIN})")
                ok = False
        covered = np.sort(np.concatenate(
            [np.where(schedule_31[r] == g)[0] for g in range(_G)]))
        if not np.array_equal(covered, np.arange(n_players)):
            print(f"  ERROR: Round {r+1} does not cover all {n_players} players!")
            ok = False

    # --- pair meeting counts ---
    meetings = np.zeros((n_players, n_players), dtype=np.int8)
    for r in range(_R):
        for g in range(_G):
            members = np.where(schedule_31[r] == g)[0]
            meetings[np.ix_(members, members)] += 1
    np.fill_diagonal(meetings, 0)

    ii, jj = np.triu_indices(n_players, k=1)
    vals    = meetings[ii, jj]
    total   = len(vals)
    n0  = int(np.sum(vals == 0))
    n1  = int(np.sum(vals == 1))
    n2  = int(np.sum(vals == 2))
    n3p = int(np.sum(vals >= 3))

    total_m = _R * (_KBIG * (_KBIG - 1) // 2 + (_G - 1) * _KMIN * (_KMIN - 1) // 2)
    print(f"\nPair meeting statistics  ({total} pairs total, "
          f"{total_m} scheduled meetings across 6 rounds):")
    print(f"  Never met        : {n0:4d}  ({100*n0/total:.1f}%)")
    print(f"  Met exactly once : {n1:4d}  ({100*n1/total:.1f}%)")
    print(f"  Met twice        : {n2:4d}  ({100*n2/total:.1f}%)")
    print(f"  Met 3+ times     : {n3p:4d}  ({100*n3p/total:.1f}%)")
    coll = n2 + n3p
    print(f"\nPairs meeting 2+ times (collisions): {coll}")
    return ok and (coll == 0)


def print_31(schedule_31: np.ndarray, names: list = None):
    n_players = schedule_31.shape[1]
    if names is None:
        names = PLAYER_NAMES
    # Compute the column width needed for the longest name label.
    col = max(len(n) for n in names)

    print("\n" + "=" * 70)
    print("  SOCIAL GOLFER SCHEDULE")
    print(f"  {n_players} players  |  1 group of {_KBIG}, "
          f"{_G-1} groups of {_KMIN}  |  {_R} rounds")
    print("=" * 70)
    for r in range(_R):
        print(f"\n  Round {r + 1}:")
        for g in range(_G):
            raw     = sorted(int(p) for p in np.where(schedule_31[r] == g)[0])
            members = ([p for p in raw if names[p] in _PRIORITY_NAMES] +
                       [p for p in raw if names[p] not in _PRIORITY_NAMES])
            labels  = ", ".join(names[p] for p in members)
            tag     = f" ({_KBIG} players)" if g == 0 else ""
            print(f"    Group {g+1}{tag}: [{labels}]")
    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Social Golfer Problem — 31 players, "
            "1 group of 6 + 5 groups of 5, 6 rounds. "
            "Phase 1: solve SGP(30,5,6). "
            "Phase 2: extend to 31 players via transversal."
        )
    )
    parser.add_argument("--seed",     type=int,   default=None)
    parser.add_argument("--timeout",  type=float, default=120.0,
                        help="Time limit for Phase 1 (default 120 s)")
    parser.add_argument("--restarts", type=int,   default=500)
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    print("Social Golfer Problem  —  31 players, "
          f"1 group of 6 + 5 groups of 5, {_R} rounds")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print(f"Phase-1 time limit: {args.timeout}s\n")

    # Compile JIT
    if verbose:
        print("Compiling JIT kernel ... ", end="", flush=True)
    t_c = time.time()
    _warm_up_jit()
    if verbose:
        print(f"done ({time.time()-t_c:.1f}s)\n")

    t0 = time.time()

    # ── Phase 1: solve 30-player SGP ────────────────────────────────────────
    print("Phase 1: solving SGP(30, 5, 6) ...")
    sched_30, score_30 = solve_30(
        timeout=args.timeout,
        max_restarts=args.restarts,
        seed=args.seed,
        verbose=verbose,
    )
    if score_30 != 0:
        print(f"\nPhase 1 failed: best score = {score_30}.")
        print("Try a longer --timeout or a different --seed.")
        sys.exit(1)
    print(f"\nPhase 1: perfect 30-player solution found in {time.time()-t0:.1f}s.\n")

    # ── Phase 2: find transversal and extend ────────────────────────────────
    print("Phase 2: finding transversal (SDR) ...")
    t_sdr = time.time()
    sdr = find_sdr(sched_30)
    if sdr is None:
        print("No transversal found for this schedule.  "
              "Try a different --seed.")
        sys.exit(1)
    print(f"Transversal found in {time.time()-t_sdr:.3f}s: "
          f"big-group indices = {[g+1 for g in sdr]}  (1-based)\n")

    # ── Construct and validate 31-player schedule ───────────────────────────
    sched_31 = construct_31(sched_30, sdr)
    perfect   = validate_31(sched_31)
    print_31(sched_31)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.2f}s")
    if perfect:
        print("Perfect solution: every pair of players meets at most once.")
    else:
        print("WARNING: schedule is not perfect (see pair statistics above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
