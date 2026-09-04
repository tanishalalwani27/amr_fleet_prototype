"""
Conflict detection: purely geometric/temporal reasoning over Trajectory
objects. This module knows nothing about communication, priorities, or
how a robot resolves a conflict -- it only answers "do these two planned
trajectories come too close to each other, and when?"

This isolation matters: coordination and communication code call into
this module, never the other way around.
"""

from dataclasses import dataclass
from typing import List, Optional
import math

from amr_fleet.conflict_detection.trajectory import Trajectory


@dataclass
class Conflict:
    robot_a: str
    robot_b: str
    t: float           # simulation time the conflict occurs at
    pos_a: tuple
    pos_b: tuple
    distance: float
    kind: str = "vertex"   # "vertex" (too close) or "edge" (swap/crossing)


def _sample_times(traj_a: Trajectory, traj_b: Trajectory, dt: float):
    lo = max(traj_a.points[0].t, traj_b.points[0].t)
    hi = min(traj_a.points[-1].t, traj_b.points[-1].t)
    if hi <= lo:
        return []
    n = max(1, int((hi - lo) / dt))
    return [lo + i * dt for i in range(n + 1)]


def find_conflict(traj_a: Trajectory, traj_b: Trajectory, safety_radius: float,
                   dt: float = 0.1) -> Optional[Conflict]:
    """
    Time-aware collision check between two trajectories. Samples both
    trajectories at a fixed dt over their overlapping time window and
    reports the first time the two robots would be closer than
    `safety_radius` (their combined footprint + margin).

    Also detects "edge" conflicts: robots swapping positions between
    consecutive samples (passing through each other), which vertex-only
    sampling can miss at coarse dt.
    """
    if traj_a.is_empty() or traj_b.is_empty():
        return None

    times = _sample_times(traj_a, traj_b, dt)
    prev = None
    for t in times:
        pa = traj_a.pose_at(t)
        pb = traj_b.pose_at(t)
        if pa is None or pb is None:
            prev = None
            continue
        d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        if d < safety_radius:
            return Conflict(traj_a.robot_id, traj_b.robot_id, t,
                             (pa[0], pa[1]), (pb[0], pb[1]), d, kind="vertex")

        if prev is not None:
            (pt, ppa, ppb) = prev
            # edge/swap conflict: A and B roughly trade places
            swap_dist = math.hypot(ppa[0] - pb[0], ppa[1] - pb[1]) + \
                        math.hypot(ppb[0] - pa[0], ppb[1] - pa[1])
            if swap_dist < safety_radius * 1.5:
                return Conflict(traj_a.robot_id, traj_b.robot_id, t,
                                 (pa[0], pa[1]), (pb[0], pb[1]), d, kind="edge")
        prev = (t, (pa[0], pa[1]), (pb[0], pb[1]))

    return None


def find_all_conflicts(trajectories: List[Trajectory], safety_radius: float,
                        dt: float = 0.1) -> List[Conflict]:
    """All pairwise conflicts among a set of trajectories (used e.g. by the
    simulation engine for metrics, and could be used offline/for testing --
    NOT used as a live fleet-wide planner)."""
    conflicts = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            c = find_conflict(trajectories[i], trajectories[j], safety_radius, dt)
            if c:
                conflicts.append(c)
    return conflicts
