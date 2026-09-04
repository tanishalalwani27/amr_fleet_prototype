"""
Distributed, CBS-inspired conflict resolution.

Classic Centralized CBS finds ALL conflicts in the whole fleet and branches
on a constraint tree to resolve them jointly. We can't do that here
(requirement #5: no central fleet-wide planner). Instead each robot runs a
local, single-agent slice of the same idea:

  1. Detect conflicts between MY planned trajectory and the trajectories of
     robots I've heard from (conflict_detection.collision_checker).
  2. For each conflict, decide -- using only locally-computable, mutually
     agreed priority (coordination.priority) -- whether *I* am the one who
     must yield.
  3. If I must yield, turn the higher-priority robots' trajectories into
     space-time constraints (motion_planning.hybrid_astar.reservations_from_trajectory)
     and ask my OWN local Hybrid A* planner to replan around them.

This is exactly CBS's "add a constraint, replan the single agent" step,
just decided and executed locally by the constrained agent instead of by
a central search tree.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from amr_fleet.conflict_detection.trajectory import Trajectory
from amr_fleet.conflict_detection.collision_checker import find_conflict, Conflict
from amr_fleet.coordination.priority import PriorityContext, wins
from amr_fleet.motion_planning.hybrid_astar import reservations_from_trajectory


@dataclass
class ResolutionResult:
    must_replan: bool
    conflicts: List[Conflict]
    reservations: set  # union of space-time cells to avoid, if must_replan
    # When the earliest conflict I must yield for actually occurs, or None if
    # I win every conflict. Lets the caller distinguish "replan now" from
    # "this is 6 s away, handle it in the next planning cycle".
    earliest_yield_t: Optional[float] = None


def check_and_resolve(my_id: str, my_priority_ctx: PriorityContext,
                       my_trajectory: Trajectory,
                       peer_trajectories: Dict[str, Trajectory],
                       peer_priority_ctx: Dict[str, PriorityContext],
                       safety_radius: float, now: float,
                       horizon: Optional[float] = None,
                       reservation_time_margin: int = 0) -> ResolutionResult:
    """
    Evaluate my_trajectory against every known peer trajectory.

    Returns a ResolutionResult telling the caller (Robot) whether it must
    replan, and -- if so -- the combined space-time reservation set (drawn
    only from peers that outrank me for at least one conflict) that the
    local Hybrid A* planner should avoid.

    `horizon` gives this a rolling planning window (WHCA*-style): only the
    next `horizon` seconds are reasoned about, and anything beyond is deferred
    to a later cycle, when the peer's fresher broadcast will be available.
    Note the window is applied to conflict detection as well as to the
    reservations, and that this is load-bearing rather than cosmetic -- if a
    conflict 30 s out still set `must_replan` while the reservations only
    covered the next 8 s, the replan would reproduce the very conflict that
    triggered it and the robot would replan every tick forever without ever
    changing its path.
    """
    conflicts = []
    must_replan = False
    reservations = set()
    earliest_yield_t = None
    until = math.inf if horizon is None else now + horizon

    my_window = my_trajectory.truncate_after(until)

    for peer_id, peer_traj in peer_trajectories.items():
        if peer_traj is None or peer_traj.is_empty():
            continue
        peer_window = peer_traj.truncate_after(until)
        if peer_window.is_empty():
            continue
        c = find_conflict(my_window, peer_window, safety_radius)
        if c is None:
            continue
        conflicts.append(c)

        peer_ctx = peer_priority_ctx.get(peer_id)
        if peer_ctx is None:
            # No priority info yet -- fail safe: assume the peer outranks us
            # so we're the one who yields (never assume it's safe to barrel
            # through an unknown robot).
            i_win = False
        else:
            i_win = wins(my_priority_ctx, peer_ctx)

        if not i_win:
            must_replan = True
            if earliest_yield_t is None or c.t < earliest_yield_t:
                earliest_yield_t = c.t
            reservations |= reservations_from_trajectory(
                peer_traj, exclude_before=now, exclude_after=until,
                time_margin_buckets=reservation_time_margin)

    return ResolutionResult(must_replan=must_replan, conflicts=conflicts,
                             reservations=reservations,
                             earliest_yield_t=earliest_yield_t)
