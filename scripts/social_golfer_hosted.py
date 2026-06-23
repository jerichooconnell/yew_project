"""
Social Golfer Problem — Hosted Table Rotation
==============================================
30 players total:
  - 6 hosts (Will, Anne, Karen, Sophia, Aaron, Lilli): each stays at their
    own table for every round.
  - 24 rotating players: visit a different table each round.

Structure per round: 6 tables × (1 host + 4 rotators) = 30 players.

The rotator assignment is equivalent to SGP(24, 4, 6): 24 players, 6 groups
of 4, 6 rounds, minimising repeated pairs within a group.

Theory
------
Total rotator pair-meetings: 6 rounds × 6 groups × C(4,2) = 216.
Total pairs among 24 rotators: C(24,2) = 276.
Round bound: (24-1)/(4-1) = 7.67 ≥ 6, so a perfect solution exists:
  → 216 distinct rotator pairs share a table exactly once, 60 never do.

Each host meets all 24 rotators exactly once (4/round × 6 rounds = 24).
No two hosts ever share a table.

Tables
------
  Table 1: Will     Table 2: Anne     Table 3: Karen
  Table 4: Sophia   Table 5: Aaron    Table 6: Lilli

Rotating players (24):
  Ellen, Sam, Lana, Art, Peter, Esther, Joan, Dan O, Noah, Rachel G,
  Char, Jericho, Kirsten, Solomon, Misty, Aja, Isobel, Shane, Sora,
  Isaac, Jordan, Dan S, Rachel, Cathe

Usage:
    python scripts/social_golfer_hosted.py [--seed SEED] [--timeout SECS]
"""

import argparse
import math
import sys
import time

import numpy as np
from numba import njit

# ─────────────────────────────────────────────────────────────────────────────
# Problem constants  (rotators only)
# ─────────────────────────────────────────────────────────────────────────────
_N = 24   # rotating players
_K = 4    # rotators per table per round
_G = 6    # tables (= number of hosts)
_R = 6    # rounds
assert _N == _G * _K

# ALL_BUT[i, k] = k-th other slot index in a group of 4, for slot i.
# Shape (4, 3).
_ALL_BUT = np.array(
    [[j for j in range(_K) if j != i] for i in range(_K)],
    dtype=np.int64,
)

# ─────────────────────────────────────────────────────────────────────────────
# Participant lists
# ─────────────────────────────────────────────────────────────────────────────
HOSTS = ["Will", "Anne", "Karen", "Sophia", "Aaron", "Lilli"]  # table 1-6

ROTATORS = [
    "Ellen",    # R00
    "Sam",      # R01
    "Lana",     # R02
    "Art",      # R03
    "Peter",    # R04
    "Esther",   # R05
    "Joan",     # R06
    "Dan O",    # R07
    "Noah",     # R08
    "Rachel G", # R09
    "Char",     # R10
    "Jericho",  # R11
    "Kirsten",  # R12
    "Solomon",  # R13
    "Misty",    # R14
    "Aja",      # R15
    "Isobel",   # R16
    "Shane",    # R17
    "Sora",     # R18
    "Isaac",    # R19
    "Jordan",   # R20
    "Dan S",    # R21
    "Rachel",   # R22
    "Cathe",    # R23
]
assert len(ROTATORS) == _N

# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT SA kernel  (SGP 24, 4, 6)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _jit_sa(schedule, gm, pos, meetings, visits, all_but, n_iter, T0, alpha, jit_seed):
    """
    Simulated annealing for SGP(24, 4, 6).

    Arrays (modified in-place):
      schedule  (6, 24)  int8    group[r, p]
      gm        (6, 6, 4) int16  members[r, g, slot]
      pos       (6, 24)  int8    slot[r, p]
      meetings  (24, 24) int8    cumulative rotator pair-meeting counts
      visits    (24, 6)  int8    how many times rotator p visits table t

    Returns best score (rotator-pair + host-revisit); schedule holds the
    best-found solution on exit.
    """
    np.random.seed(jit_seed)
    cur_score = 0
    for ii in range(24):          # rotator-pair violations × 10
        for jj in range(ii + 1, 24):
            v = int(meetings[ii, jj]) - 1
            if v > 0:
                cur_score += 10 * v
    for p in range(24):           # host-revisit violations × 1
        for t in range(6):
            v = int(visits[p, t]) - 1
            if v > 0:
                cur_score += v

    best_score = cur_score
    best_sched = schedule.copy()
    T = T0

    for _ in range(n_iter):
        r  = np.random.randint(0, 6)
        p1 = np.random.randint(0, 24)
        p2 = np.random.randint(0, 24)

        g1 = int(schedule[r, p1])
        g2 = int(schedule[r, p2])
        if g1 == g2:
            T *= alpha
            continue

        i1 = int(pos[r, p1])
        i2 = int(pos[r, p2])

        d_rr = 0
        for k in range(3):   # K-1 = 3 group-mates
            q1 = int(gm[r, g1, all_but[i1, k]])
            if meetings[p1, q1] > 1: d_rr -= 1
            if meetings[p2, q1] > 0: d_rr += 1
        for k in range(3):
            q2 = int(gm[r, g2, all_but[i2, k]])
            if meetings[p2, q2] > 1: d_rr -= 1
            if meetings[p1, q2] > 0: d_rr += 1
        d_hr = 0
        if visits[p1, g1] > 1: d_hr -= 1
        if visits[p1, g2] >= 1: d_hr += 1
        if visits[p2, g2] > 1: d_hr -= 1
        if visits[p2, g1] >= 1: d_hr += 1
        d = 10 * d_rr + d_hr

        if d <= 0 or np.random.random() < math.exp(-d / T):
            for k in range(3):
                q1 = int(gm[r, g1, all_but[i1, k]])
                meetings[p1, q1] -= 1;  meetings[q1, p1] -= 1
                meetings[p2, q1] += 1;  meetings[q1, p2] += 1
            for k in range(3):
                q2 = int(gm[r, g2, all_but[i2, k]])
                meetings[p2, q2] -= 1;  meetings[q2, p2] -= 1
                meetings[p1, q2] += 1;  meetings[q2, p1] += 1
            visits[p1, g1] -= 1;  visits[p1, g2] += 1
            visits[p2, g2] -= 1;  visits[p2, g1] += 1
            gm[r, g1, i1] = p2;   gm[r, g2, i2] = p1
            pos[r, p1] = i2;      pos[r, p2] = i1
            schedule[r, p1] = g2; schedule[r, p2] = g1
            cur_score += d

            if cur_score < best_score:
                best_score = cur_score
                for ri in range(6):
                    for pi in range(24):
                        best_sched[ri, pi] = schedule[ri, pi]
                if best_score == 0:
                    break

        T *= alpha

    for ri in range(6):
        for pi in range(24):
            schedule[ri, pi] = best_sched[ri, pi]

    return best_score


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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
    visits   = np.zeros((_N, _G), dtype=np.int8)
    for r in range(_R):
        for g in range(_G):
            m = gm[r, g]
            meetings[np.ix_(m, m)] += 1
            visits[m, g] += 1
    np.fill_diagonal(meetings, 0)
    return gm, pos, meetings, visits


def _score(meetings):
    return int(np.sum(np.triu(np.maximum(meetings - 1, 0), k=1)))


def _host_score(visits):
    """Number of (rotator, table) pairs where the rotator visits more than once."""
    return int(np.sum(np.maximum(visits - 1, 0)))


def _total_score(meetings, visits):
    return _score(meetings) + _host_score(visits)


def _weighted_score(meetings, visits):
    """Internal optimisation objective: 10 × rotator-pair + host-revisit."""
    return 10 * _score(meetings) + _host_score(visits)


def _greedy_init(rng):
    schedule = np.full((_R, _N), -1, dtype=np.int8)
    gm       = np.zeros((_R, _G, _K), dtype=np.int16)
    pos      = np.zeros((_R, _N), dtype=np.int8)
    sizes    = np.zeros((_R, _G), dtype=np.int8)
    meetings = np.zeros((_N, _N), dtype=np.int8)
    visits   = np.zeros((_N, _G), dtype=np.int8)
    for r in range(_R):
        for p in rng.permutation(_N).tolist():
            best_g, best_c = -1, _N + 1
            for g in rng.permutation(_G).tolist():
                if sizes[r, g] >= _K:
                    continue
                sz = int(sizes[r, g])
                c  = int(np.sum(meetings[p, gm[r, g, :sz]] >= 1)) if sz else 0
                c += int(visits[p, g] >= 1)   # penalise revisiting a host table
                if c < best_c:
                    best_c, best_g = c, g
                if c == 0:
                    break
            sz = int(sizes[r, best_g])
            gm[r, best_g, sz] = p
            pos[r, p]         = sz
            sizes[r, best_g] += 1
            schedule[r, p]    = best_g
            visits[p, best_g] += 1
        for g in range(_G):
            m = gm[r, g]
            meetings[np.ix_(m, m)] += 1
        np.fill_diagonal(meetings, 0)
    return schedule, gm, pos, meetings, visits, _weighted_score(meetings, visits)


def _best_swap_in_round(r, schedule, gm, pos, meetings, visits):
    g   = schedule[r]
    A   = (meetings > 0).astype(np.int32)
    B   = (meetings > 1).astype(np.int32)
    G   = np.zeros((_N, _G), dtype=np.int32)
    G[np.arange(_N), g] = 1
    H_A = A @ G
    H_B = B @ G
    F   = H_B[np.arange(_N), g]
    H_p = H_A[:, g]
    delta_rr   = H_p + H_p.T - 2 * A - F[:, None] - F[None, :]
    # host-rotator component: penalise a rotator revisiting a table
    exceed     = (visits[np.arange(_N), g] > 1).astype(np.int32)
    already_at = (visits >= 1).astype(np.int32)[:, g]  # [p1,p2] = visits[p1,g[p2]]>=1
    delta_hr   = -exceed[:, None] - exceed[None, :] + already_at + already_at.T
    delta      = 10 * delta_rr + delta_hr  # rr violations count 10× more
    ut = np.triu(g[:, None] != g[None, :], k=1)
    if not np.any(delta[ut] < 0):
        return 0
    d_masked = np.where(ut, delta, 1)
    flat     = int(d_masked.argmin())
    p1, p2   = divmod(flat, _N)
    g1 = int(g[p1]);  g2 = int(g[p2])
    i1 = int(pos[r, p1]);  i2 = int(pos[r, p2])
    o1 = gm[r, g1];  o1 = o1[o1 != p1]
    o2 = gm[r, g2];  o2 = o2[o2 != p2]
    meetings[p1, o1] -= 1;  meetings[o1, p1] -= 1
    meetings[p2, o2] -= 1;  meetings[o2, p2] -= 1
    meetings[p1, o2] += 1;  meetings[o2, p1] += 1
    meetings[p2, o1] += 1;  meetings[o1, p2] += 1
    visits[p1, g1] -= 1;  visits[p1, g2] += 1
    visits[p2, g2] -= 1;  visits[p2, g1] += 1
    gm[r, g1, i1] = p2;  gm[r, g2, i2] = p1
    pos[r, p1] = i2;     pos[r, p2] = i1
    schedule[r, p1] = g2; schedule[r, p2] = g1
    return int(delta[p1, p2])


def _hill_climb(schedule, gm, pos, meetings, visits, score, max_idle=80):
    idle = 0
    while score > 0 and idle < max_idle:
        improved = False
        for r in range(_R):
            d = _best_swap_in_round(r, schedule, gm, pos, meetings, visits)
            if d < 0:
                score += d;  improved = True;  idle = 0
                if score == 0:
                    return 0
        if not improved:
            idle += 1
    return score


def _ils_kick(schedule, gm, pos, meetings, visits, rng):
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
        visits[m, g] -= 1
    np.fill_diagonal(meetings, 0)
    # visit-aware greedy: prefer tables this rotator hasn't visited yet
    group_sizes = np.zeros(_G, dtype=np.int8)
    new_grp     = np.empty(_N, dtype=np.int8)
    for p in rng.permutation(_N).tolist():
        best_g, best_c = -1, _N + 1
        for g in rng.permutation(_G).tolist():
            if group_sizes[g] >= _K:
                continue
            c = int(visits[p, g] >= 1)
            if c < best_c:
                best_c, best_g = c, g
            if c == 0:
                break
        new_grp[p] = best_g
        group_sizes[best_g] += 1
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
        visits[m, g] += 1
    np.fill_diagonal(meetings, 0)
    return _weighted_score(meetings, visits)


def _ils_kick_2round(schedule, gm, pos, meetings, visits, rng):
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
            visits[m, g] -= 1
    np.fill_diagonal(meetings, 0)
    # visit-aware greedy for each kicked round, updating visits between them
    for wr in worst_2:
        group_sizes = np.zeros(_G, dtype=np.int8)
        new_grp     = np.empty(_N, dtype=np.int8)
        for p in rng.permutation(_N).tolist():
            best_g, best_c = -1, _N + 1
            for g in rng.permutation(_G).tolist():
                if group_sizes[g] >= _K:
                    continue
                c = int(visits[p, g] >= 1)
                if c < best_c:
                    best_c, best_g = c, g
                if c == 0:
                    break
            new_grp[p] = best_g
            group_sizes[best_g] += 1
        for p in range(_N):
            schedule[wr, p] = new_grp[p]
        for g in range(_G):
            members = np.where(schedule[wr] == g)[0]
            gm[wr, g] = members
            for slot, p in enumerate(members):
                pos[wr, int(p)] = slot
        for g in range(_G):
            visits[gm[wr, g], g] += 1   # update before assigning the next round
    for wr in worst_2:
        for g in range(_G):
            m = gm[wr, g]
            meetings[np.ix_(m, m)] += 1
    np.fill_diagonal(meetings, 0)
    return _weighted_score(meetings, visits)


# SA hyper-params  — two-phase schedule
# Phase 1 (rr exploration): T  62 → 5;  exp(-100/62)=0.20 lets rr uphill occur
# Phase 2 (hr refinement):  T   5 → 0.3; exp(-100/5)≈0 freezes rr, hr converges
_SA_N_ITER1  = 25_000_000
_SA_T0_1     = 62.0
_SA_TF_1     = 5.0
_SA_ALPHA1   = math.exp(math.log(_SA_TF_1  / _SA_T0_1)  / _SA_N_ITER1)
_SA_N_ITER2  = 25_000_000
_SA_T0_2     = 5.0
_SA_TF_2     = 0.3
_SA_ALPHA2   = math.exp(math.log(_SA_TF_2  / _SA_T0_2)  / _SA_N_ITER2)


def _warm_up_jit():
    s  = np.zeros((_R, _N), dtype=np.int8)
    gm = np.zeros((_R, _G, _K), dtype=np.int16)
    p  = np.zeros((_R, _N), dtype=np.int8)
    m  = np.zeros((_N, _N), dtype=np.int8)
    v  = np.zeros((_N, _G), dtype=np.int8)
    for r in range(_R):
        for g in range(_G):
            for sl in range(_K):
                pl = g * _K + sl
                s[r, pl] = g;  gm[r, g, sl] = pl;  p[r, pl] = sl
    _jit_sa(s, gm, p, m, v, _ALL_BUT, 1, 0.1, 0.99, 0)


def _best_hr_swap_in_round(r, schedule, gm, pos, meetings, visits):
    """Best rr-neutral-or-improving swap that also improves hr."""
    g          = schedule[r]
    A          = (meetings > 0).astype(np.int32)
    B          = (meetings > 1).astype(np.int32)
    G_mat      = np.zeros((_N, _G), dtype=np.int32)
    G_mat[np.arange(_N), g] = 1
    H_A = A @ G_mat;  H_B = B @ G_mat
    F   = H_B[np.arange(_N), g];  H_p = H_A[:, g]
    delta_rr   = H_p + H_p.T - 2 * A - F[:, None] - F[None, :]
    exceed     = (visits[np.arange(_N), g] > 1).astype(np.int32)
    already_at = (visits >= 1).astype(np.int32)[:, g]
    delta_hr   = -exceed[:, None] - exceed[None, :] + already_at + already_at.T
    ut   = np.triu(g[:, None] != g[None, :], k=1)
    mask = ut & (delta_rr <= 0) & (delta_hr < 0)
    if not np.any(mask):
        return 0
    d_hr_m = np.where(mask, delta_hr, 1)
    flat   = int(d_hr_m.argmin())
    p1, p2 = divmod(flat, _N)
    if delta_hr[p1, p2] >= 0:
        return 0
    g1 = int(g[p1]);  g2 = int(g[p2])
    i1 = int(pos[r, p1]);  i2 = int(pos[r, p2])
    o1 = gm[r, g1];  o1 = o1[o1 != p1]
    o2 = gm[r, g2];  o2 = o2[o2 != p2]
    meetings[p1, o1] -= 1;  meetings[o1, p1] -= 1
    meetings[p2, o2] -= 1;  meetings[o2, p2] -= 1
    meetings[p1, o2] += 1;  meetings[o2, p1] += 1
    meetings[p2, o1] += 1;  meetings[o1, p2] += 1
    visits[p1, g1] -= 1;  visits[p1, g2] += 1
    visits[p2, g2] -= 1;  visits[p2, g1] += 1
    gm[r, g1, i1] = p2;  gm[r, g2, i2] = p1
    pos[r, p1] = i2;     pos[r, p2] = i1
    schedule[r, p1] = g2; schedule[r, p2] = g1
    return int(delta_hr[p1, p2])


def _hr_hill_climb(schedule, gm, pos, meetings, visits, score_hr, max_idle=80):
    """Reduce hr violations using only rr-safe (delta_rr ≤ 0) swaps."""
    idle = 0
    while score_hr > 0 and idle < max_idle:
        improved = False
        for r in range(_R):
            d = _best_hr_swap_in_round(r, schedule, gm, pos, meetings, visits)
            if d < 0:
                score_hr += d;  improved = True;  idle = 0
                if score_hr == 0:
                    return 0
        if not improved:
            idle += 1
    return score_hr


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve(timeout: float, max_restarts: int, seed, verbose: bool):
    """Solve SGP(24, 4, 6).  Returns (schedule, score)."""
    rng = np.random.default_rng(seed)
    t0  = time.time()
    best_sched = None
    best_score = sys.maxsize

    for restart in range(max_restarts):
        if time.time() - t0 > timeout:
            break

        jit_seed = restart * 7919 + 1
        schedule, gm, pos, meetings, visits, init_w = _greedy_init(rng)
        # Phase 1 SA: explore rr landscape  (T 62 → 5)
        _jit_sa(schedule, gm, pos, meetings, visits, _ALL_BUT,
                _SA_N_ITER1, _SA_T0_1, _SA_ALPHA1, jit_seed)
        # Phase 2 SA: converge hr while rr is frozen  (T 5 → 0.3)
        gm, pos, meetings, visits = _rebuild(schedule)
        _jit_sa(schedule, gm, pos, meetings, visits, _ALL_BUT,
                _SA_N_ITER2, _SA_T0_2, _SA_ALPHA2, jit_seed + 13)
        gm, pos, meetings, visits = _rebuild(schedule)
        sa_w     = _weighted_score(meetings, visits)
        hc_score = _hill_climb(schedule, gm, pos, meetings, visits, sa_w)

        best_local = schedule.copy()
        best_ls    = hc_score

        for kick_idx in range(400):
            if best_ls == 0:
                break
            schedule[:] = best_local
            gm, pos, meetings, visits = _rebuild(schedule)
            if kick_idx % 10 >= 7:
                kicked = _ils_kick_2round(schedule, gm, pos, meetings, visits, rng)
            else:
                kicked = _ils_kick(schedule, gm, pos, meetings, visits, rng)
            after = _hill_climb(schedule, gm, pos, meetings, visits, kicked)
            if after < best_ls:
                best_ls    = after
                best_local = schedule.copy()

        # Secondary hr pass: rr-preserving hr cleanup
        schedule = best_local
        gm, pos, meetings, visits = _rebuild(schedule)
        hr_after = _host_score(visits)
        _hr_hill_climb(schedule, gm, pos, meetings, visits, hr_after)
        final = _weighted_score(meetings, visits)

        if verbose:
            rr_d = _score(meetings);  hr_d = _host_score(visits)
            print(f"  Restart {restart + 1:3d}:  "
                  f"rr={rr_d} hr={hr_d} total={rr_d+hr_d}  "
                  f"[{time.time() - t0:.1f}s]")

        if final < best_score:
            best_score = final
            best_sched = schedule.copy()

        if best_score == 0:
            if verbose:
                print(f"\n  *** Perfect solution found on restart {restart + 1} ***\n")
            break

    return best_sched, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Validation & output
# ─────────────────────────────────────────────────────────────────────────────

def validate(schedule) -> bool:
    """Check structure and pair-meeting counts for both rotators and hosts."""
    ok = True
    for r in range(_R):
        for g in range(_G):
            sz = int(np.sum(schedule[r] == g))
            if sz != _K:
                print(f"  ERROR round {r+1} table {g+1}: {sz} rotators (want {_K})")
                ok = False
        covered = np.sort(np.concatenate(
            [np.where(schedule[r] == g)[0] for g in range(_G)]))
        if not np.array_equal(covered, np.arange(_N)):
            print(f"  ERROR round {r+1}: rotators not fully covered")
            ok = False

    _, _, meetings, visits = _rebuild(schedule)

    # Rotator–rotator pair statistics
    ii, jj = np.triu_indices(_N, k=1)
    vals    = meetings[ii, jj]
    total   = len(vals)
    n0  = int(np.sum(vals == 0))
    n1  = int(np.sum(vals == 1))
    n2  = int(np.sum(vals == 2))
    n3p = int(np.sum(vals >= 3))
    rr_coll = n2 + n3p

    print(f"\nRotator\u2013rotator pair statistics  ({total} pairs, "
          f"{_R * _G * (_K*(_K-1)//2)} scheduled meetings):")
    print(f"  Never shared a table : {n0:4d}  ({100*n0/total:.1f}%)")
    print(f"  Shared exactly once  : {n1:4d}  ({100*n1/total:.1f}%)")
    print(f"  Shared twice         : {n2:4d}  ({100*n2/total:.1f}%)")
    print(f"  Shared 3+ times      : {n3p:4d}  ({100*n3p/total:.1f}%)")
    print(f"  Collisions (2+ times): {rr_coll}")

    # Host–rotator visit statistics
    hv_total = _N * _G
    hv_flat  = visits.flatten()
    hv0  = int(np.sum(hv_flat == 0))
    hv1  = int(np.sum(hv_flat == 1))
    hv2p = int(np.sum(hv_flat >= 2))
    hr_coll = int(np.sum(np.maximum(visits - 1, 0)))

    print(f"\nHost\u2013rotator visit statistics  "
          f"({_N} rotators \u00d7 {_G} tables = {hv_total} pairs):")
    print(f"  Never visited        : {hv0:4d}  ({100*hv0/hv_total:.1f}%)")
    print(f"  Visited exactly once : {hv1:4d}  ({100*hv1/hv_total:.1f}%)")
    print(f"  Visited 2+ times     : {hv2p:4d}  ({100*hv2p/hv_total:.1f}%)")
    print(f"  Revisit collisions   : {hr_coll}")

    total_coll = rr_coll + hr_coll
    print(f"\nTotal score: {total_coll}  "
          f"(rotator-pair: {rr_coll}, host-revisit: {hr_coll})")
    return ok and total_coll == 0


def print_schedule(schedule):
    print("\n" + "=" * 62)
    print("  HOSTED TABLE ROTATION SCHEDULE")
    print(f"  6 tables × (1 host + 4 rotators)  |  {_R} rounds")
    print("  Hosts are stationary; rotators change table each round.")
    print("=" * 62)
    for r in range(_R):
        print(f"\n  Round {r + 1}:")
        for g in range(_G):
            rotators = sorted(int(p) for p in np.where(schedule[r] == g)[0])
            rnames   = ", ".join(ROTATORS[p] for p in rotators)
            print(f"    {HOSTS[g]}'s table: {HOSTS[g]} | {rnames}")
    print("\n" + "=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hosted table rotation: 6 stationary hosts, 24 rotating players, "
            "6 rounds of 4 per table.  Solves SGP(24, 4, 6)."
        )
    )
    parser.add_argument("--seed",     type=int,   default=None)
    parser.add_argument("--timeout",  type=float, default=60.0,
                        help="Time limit in seconds (default 60)")
    parser.add_argument("--restarts", type=int,   default=500)
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    print("Hosted Table Rotation  —  SGP(24, 4, 6)")
    print(f"Hosts: {', '.join(HOSTS)}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Time limit: {args.timeout}s\n")

    if verbose:
        print("Compiling JIT kernel ... ", end="", flush=True)
    t_c = time.time()
    _warm_up_jit()
    if verbose:
        print(f"done ({time.time()-t_c:.1f}s)\n")

    t0 = time.time()
    sched, score = solve(
        timeout=args.timeout,
        max_restarts=args.restarts,
        seed=args.seed,
        verbose=verbose,
    )

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.2f}s  —  total score: {score}")

    perfect = validate(sched)
    print_schedule(sched)

    if perfect:
        print("Perfect: every pair of rotators shares a table at most once,")
        print("and every rotator visits each host's table exactly once.")
    else:
        print("Note: perfect solution not found within budget.")
        print("Try a longer --timeout or different --seed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
