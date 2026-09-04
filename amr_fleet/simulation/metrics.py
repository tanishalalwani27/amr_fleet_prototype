"""
Metrics collection, kept entirely separate from simulation mechanics so it
can be swapped out (e.g. for CSV export, dashboards) without touching the
engine or any robot logic.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricsCollector:
    collision_events: List[tuple] = field(default_factory=list)   # (t, id_a, id_b, dist)
    deadlock_events: List[tuple] = field(default_factory=list)    # (t, id)
    _deadlocked_currently: set = field(default_factory=set)

    def record_positions(self, t: float, robots, collision_radius: float):
        ids = list(robots.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = robots[ids[i]], robots[ids[j]]
                if a.status.name == "DONE" or b.status.name == "DONE":
                    continue
                d = math.hypot(a.pose[0] - b.pose[0], a.pose[1] - b.pose[1])
                if d < collision_radius:
                    self.collision_events.append((t, a.id, b.id, d))

    def record_deadlocks(self, t: float, robots):
        for rid, r in robots.items():
            is_dl = (r.status.name == "DEADLOCKED")
            if is_dl and rid not in self._deadlocked_currently:
                self.deadlock_events.append((t, rid))
                self._deadlocked_currently.add(rid)
            elif not is_dl and rid in self._deadlocked_currently:
                self._deadlocked_currently.discard(rid)

    def summary(self, robots, channel, sim_time: float) -> Dict:
        completed = [r for r in robots.values() if r.finish_time is not None]
        completion_times = [r.finish_time - r.start_time for r in completed]
        path_lengths = {r.id: r.path_length for r in robots.values()}
        replans = {r.id: r.replanning_count for r in robots.values()}

        return {
            "collisions": len(self.collision_events),
            "collision_events": self.collision_events,
            "deadlocks": len(self.deadlock_events),
            "deadlock_events": self.deadlock_events,
            "tasks_completed": len(completed),
            "tasks_total": len(robots),
            "avg_completion_time_s": (sum(completion_times) / len(completion_times)
                                       if completion_times else None),
            "completion_times_s": {r.id: (r.finish_time - r.start_time)
                                    for r in completed},
            "path_length_m": path_lengths,
            "total_replanning_count": sum(replans.values()),
            "replanning_count_per_robot": replans,
            "total_yield_count": sum(r.yield_count for r in robots.values()),
            "yield_count_per_robot": {r.id: r.yield_count for r in robots.values()},
            "total_wait_steps": sum(r.wait_steps for r in robots.values()),
            "wait_steps_per_robot": {r.id: r.wait_steps for r in robots.values()},
            "total_conflict_count": sum(r.conflict_count for r in robots.values()),
            "conflict_count_per_robot": {r.id: r.conflict_count for r in robots.values()},
            "avg_comm_latency_s": channel.average_latency(),
            "comm_loss_rate": channel.loss_rate(),
            "throughput_tasks_per_min": (len(completed) / (sim_time / 60.0)
                                          if sim_time > 0 else 0.0),
            "sim_time_s": sim_time,
        }

    def print_report(self, robots, channel, sim_time: float):
        s = self.summary(robots, channel, sim_time)
        print("=" * 60)
        print("FLEET METRICS REPORT")
        print("=" * 60)
        print(f"Simulated time:            {s['sim_time_s']:.1f} s")
        print(f"Tasks completed:           {s['tasks_completed']}/{s['tasks_total']}")
        print(f"Collisions detected:       {s['collisions']}")
        print(f"Deadlock events:           {s['deadlocks']}")
        if s['avg_completion_time_s'] is not None:
            print(f"Avg task completion time:  {s['avg_completion_time_s']:.2f} s")
        print(f"Total replanning events:   {s['total_replanning_count']}")
        print(f"Total yield events:        {s['total_yield_count']}")
        print(f"Total wait steps:          {s['total_wait_steps']}")
        print(f"Total conflicts detected:  {s['total_conflict_count']}")
        for rid, n in s['replanning_count_per_robot'].items():
            print(f"    {rid}: {n} replans, {s['yield_count_per_robot'][rid]} yields, "
                  f"{s['wait_steps_per_robot'][rid]} waits, "
                  f"path length {s['path_length_m'][rid]:.2f} m")
        print(f"Avg comm latency:          {s['avg_comm_latency_s']*1000:.1f} ms")
        print(f"Comm loss rate:            {s['comm_loss_rate']*100:.1f} %")
        print(f"Throughput:                {s['throughput_tasks_per_min']:.2f} tasks/min")
        print("=" * 60)
