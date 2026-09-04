"""
Trajectory: the shared data contract between motion planning, conflict
detection, coordination and communication.

A Trajectory is a time-stamped sequence of poses. It is the ONLY thing
robots exchange about their intended motion -- nobody exchanges internal
planner state, search trees, etc.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import math


@dataclass
class TrajectoryPoint:
    t: float        # absolute simulation time (seconds)
    x: float
    y: float
    theta: float
    v: float        # signed speed at this point (m/s)


@dataclass
class Trajectory:
    robot_id: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    generation: int = 0   # bumped every time this robot replans

    def is_empty(self):
        return len(self.points) == 0

    def duration(self):
        if self.is_empty():
            return 0.0
        return self.points[-1].t - self.points[0].t

    def length(self):
        """Path length in meters."""
        d = 0.0
        for a, b in zip(self.points, self.points[1:]):
            d += math.hypot(b.x - a.x, b.y - a.y)
        return d

    def pose_at(self, t: float):
        """
        Linearly interpolate (x, y, theta) at absolute time t.
        Returns None if t is outside the trajectory's time range.
        """
        pts = self.points
        if not pts or t < pts[0].t or t > pts[-1].t:
            return None
        if len(pts) == 1:
            p = pts[0]
            return (p.x, p.y, p.theta)
        for a, b in zip(pts, pts[1:]):
            if a.t <= t <= b.t:
                span = (b.t - a.t) or 1e-9
                f = (t - a.t) / span
                x = a.x + f * (b.x - a.x)
                y = a.y + f * (b.y - a.y)
                dtheta = math.atan2(math.sin(b.theta - a.theta), math.cos(b.theta - a.theta))
                theta = a.theta + f * dtheta
                return (x, y, theta)
        return None

    def truncate_after(self, t: float):
        """Return a new Trajectory containing only points with time <= t."""
        pts = [p for p in self.points if p.t <= t]
        return Trajectory(self.robot_id, pts, self.generation)
