"""
The simulation loop. IMPORTANT: this module only sequences per-tick calls
into robots/world/channel/metrics -- it never computes a path for more
than one robot itself. That responsibility stays inside each Robot's own
HybridAStarPlanner instance, preserving requirement #5.
"""

from typing import Dict, List, Optional, Callable

from amr_fleet.simulation.world import World
from amr_fleet.simulation.robot import Robot, RobotStatus
from amr_fleet.simulation.metrics import MetricsCollector
from amr_fleet.communication.channel import CommChannel


class SimulationEngine:
    def __init__(self, world: World, robots: List[Robot], channel: CommChannel,
                 dt: float = 0.2, collision_radius: float = 0.7):
        self.world = world
        self.robots: Dict[str, Robot] = {r.id: r for r in robots}
        self.channel = channel
        self.dt = dt
        self.collision_radius = collision_radius
        self.metrics = MetricsCollector()
        self.t = 0.0
        self.history: List[Dict] = []   # per-tick snapshot, used by the visualizer
        self._scheduled_link_events: List[tuple] = []  # (t, a, b, down_bool)

    def schedule_link_event(self, t: float, robot_a: str, robot_b: str, down: bool):
        """Schedule a communication link failure/recovery at time t, to
        demonstrate requirement #11 (fleet keeps operating despite a lost
        connection)."""
        self._scheduled_link_events.append((t, robot_a, robot_b, down))

    def _apply_scheduled_link_events(self):
        remaining = []
        for (t, a, b, down) in self._scheduled_link_events:
            if t <= self.t:
                self.channel.set_link_down(a, b, down)
            else:
                remaining.append((t, a, b, down))
        self._scheduled_link_events = remaining

    def initialize(self):
        for r in self.robots.values():
            r.plan_initial(self.world, self.t)

    def step(self):
        self._apply_scheduled_link_events()
        self.channel.tick(self.t)

        all_ids = list(self.robots.keys())

        # 1. receive whatever comms have arrived
        for r in self.robots.values():
            r.receive(self.channel, self.t)

        # 2. each robot independently checks conflicts & replans if needed
        for r in self.robots.values():
            r.coordinate(self.world, self.t)

        # 3. each robot advances along its own trajectory
        for r in self.robots.values():
            r.advance(self.t, self.dt)

        # 4. each robot broadcasts its (possibly updated) plan
        for r in self.robots.values():
            r.broadcast(self.channel, all_ids, self.t)

        # 5. metrics
        self.metrics.record_positions(self.t, self.robots, self.collision_radius)
        self.metrics.record_deadlocks(self.t, self.robots)

        self.history.append({
            "t": self.t,
            "robots": {rid: {"pose": r.pose, "status": r.status.name,
                              "trajectory": r.trajectory,
                              "yield_count": r.yield_count,
                              "conflict_count": r.conflict_count,
                              "wait_steps": r.wait_steps}
                       for rid, r in self.robots.items()},
            "dynamic_obstacles": {o.obstacle_id: o.position_at(self.t)
                                    for o in self.world.dynamic_obstacles},
        })

        self.t += self.dt

    def all_done(self) -> bool:
        return all(r.status in (RobotStatus.DONE, RobotStatus.FAILED)
                   for r in self.robots.values())

    def run(self, max_time: float = 120.0, on_step: Optional[Callable] = None):
        self.initialize()
        while self.t < max_time and not self.all_done():
            self.step()
            if on_step:
                on_step(self)
        return self.metrics.summary(self.robots, self.channel, self.t)
