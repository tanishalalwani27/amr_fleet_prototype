"""
Priority scheme used to break ties in distributed conflict resolution.

No robot has authority over another -- priority is just a comparable
number that every robot can compute locally (from information it already
has: its own state, and the peer state it received over comms) and agree
on deterministically, without any negotiation protocol or central
arbiter.

The catch is that peer state arrives over a lossy channel with latency, so
it is always slightly STALE. Every design choice below exists to make two
robots reach the same verdict anyway; without that, both can conclude they
won and neither yields, which is exactly how two AMRs drive into each other.
"""

import math
from dataclasses import dataclass


@dataclass
class PriorityContext:
    remaining_distance: float
    wait_time: float          # how long this robot has been stuck/yielding
    robot_id: str


# Priority differences smaller than this are treated as a tie and decided by
# robot_id. It has to be comfortably larger than the drift that staleness
# introduces: at ~1.2 m/s a peer's remaining_distance can be ~0.5 m old by the
# time it is used, which moves the base term by ~0.01. A tie-break threshold of
# 1e-9 (what this used to be) is meaningless against that noise -- two robots
# would disagree, both think they won, and neither would replan.
PRIORITY_EPS = 0.05

# Granularity of the anti-starvation aging steps, in seconds of waiting.
AGING_STEP_S = 2.0
AGING_STEP_VALUE = 0.5
AGING_MAX = 5.0


def compute_priority(ctx: PriorityContext) -> float:
    """
    Higher value = higher priority (wins conflicts, doesn't replan).

    - Robots closer to their destination get slight priority (they have
      less to lose by continuing, and it reduces total system time).
    - Robots that have already been waiting/yielding a long time get an
      escalating priority boost, which guarantees no robot can be starved
      forever (a basic distributed livelock/deadlock-avoidance measure).
      The boost is quantized rather than continuous so that a peer's
      wait_time being up to a broadcast interval stale cannot change it.
    - robot_id is used as the tie-breaker whenever the two priorities are
      close, which is what makes the verdict reproducible from imperfect
      information: ids never change and are never stale.
    """
    base = 1.0 / (1.0 + ctx.remaining_distance)
    aging_boost = min(math.floor(ctx.wait_time / AGING_STEP_S) * AGING_STEP_VALUE,
                      AGING_MAX)
    return base + aging_boost


def wins(my_ctx: PriorityContext, other_ctx: PriorityContext) -> bool:
    """Deterministic comparator: True if `my_ctx` robot has priority over
    `other_ctx` robot for a detected conflict. Both robots can evaluate this
    independently and will agree, because anything within PRIORITY_EPS falls
    through to a stable id tie-break instead of being decided by stale floats.

    Strictly antisymmetric: exactly one of wins(a, b) and wins(b, a) is True."""
    mp = compute_priority(my_ctx)
    op = compute_priority(other_ctx)
    if abs(mp - op) > PRIORITY_EPS:
        return mp > op
    return my_ctx.robot_id > other_ctx.robot_id
