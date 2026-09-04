"""
The warehouse environment: static layout (walls/shelving) and dynamic
obstacles (e.g. humans, forklifts) that do NOT participate in the robot
communication protocol -- robots only know about them via limited-range
"sensing", which is how requirement #9 (dynamic obstacles) is handled
without breaking the decentralization requirement.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from amr_fleet.motion_planning.hybrid_astar import to_cell, to_time_bucket


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x, y, margin=0.0):
        return (self.x0 - margin <= x <= self.x1 + margin and
                self.y0 - margin <= y <= self.y1 + margin)


@dataclass
class DynamicObstacle:
    """A moving obstacle that follows a fixed back-and-forth path at
    constant speed -- simple, but enough to force robots to sense and
    react to something unpredictable from their point of view (they don't
    receive its trajectory over comms, only sense it locally)."""
    obstacle_id: str
    waypoints: List[Tuple[float, float]]
    speed: float = 0.6
    radius: float = 0.3
    _t0: float = 0.0

    def position_at(self, t: float) -> Tuple[float, float]:
        if len(self.waypoints) < 2:
            return self.waypoints[0]
        # total loop length (ping-pong)
        segs = list(zip(self.waypoints, self.waypoints[1:])) + \
            list(zip(self.waypoints[::-1], self.waypoints[::-1][1:]))
        seg_lens = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs]
        total = sum(seg_lens) or 1e-6
        dist = (self.speed * t) % total
        for (a, b), L in zip(segs, seg_lens):
            if dist <= L or L == 0:
                f = 0 if L == 0 else dist / L
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
            dist -= L
        return self.waypoints[0]


class World:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.static_obstacles: List[Rect] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []

    def add_static(self, rect: Rect):
        self.static_obstacles.append(rect)

    def add_dynamic(self, obs: DynamicObstacle):
        self.dynamic_obstacles.append(obs)

    def is_occupied(self, x: float, y: float, margin: float = 0.0) -> bool:
        if x - margin < 0 or y - margin < 0 or x + margin > self.width or y + margin > self.height:
            return True
        for r in self.static_obstacles:
            if r.contains(x, y, margin):
                return True
        return False

    def sensed_dynamic_reservations(self, x: float, y: float, t: float,
                                     sense_radius: float, horizon: float,
                                     dt: float = 0.25, time_margin_buckets: int = 0):
        """
        A robot "senses" nearby dynamic obstacles (within sense_radius of
        its current position) and forward-projects their motion over a
        short horizon to build space-time reservations for its own local
        Hybrid A* planner. This is per-robot local sensing, NOT a shared
        fleet-wide obstacle map.

        Only the obstacle's own cell/bucket is emitted per sample: the planner's
        `_collides_dynamic` already applies the 3x3-cell and ±1-bucket
        discretization margin, so inflating here as well would square the
        effective obstacle size. `time_margin_buckets` is an explicit opt-in
        for deliberately more conservative behaviour.
        """
        out = set()
        for obs in self.dynamic_obstacles:
            ox, oy = obs.position_at(t)
            if math.hypot(ox - x, oy - y) > sense_radius:
                continue
            steps = int(horizon / dt)
            for i in range(steps + 1):
                tt = t + i * dt
                px, py = obs.position_at(tt)
                cell = to_cell(px, py)
                tb = to_time_bucket(tt)
                for ddt in range(-time_margin_buckets, time_margin_buckets + 1):
                    out.add((cell[0], cell[1], tb + ddt))
        return out
