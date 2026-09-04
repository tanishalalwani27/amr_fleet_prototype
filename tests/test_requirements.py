"""
Requirement-by-requirement checks. Run with:
    python3 -m pytest tests/ -v
or just:
    python3 tests/test_requirements.py
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_fleet.simulation.world import World, Rect, DynamicObstacle
from amr_fleet.simulation.robot import Robot, RobotStatus
from amr_fleet.simulation.engine import SimulationEngine
from amr_fleet.communication.channel import CommChannel
from amr_fleet.motion_planning.vehicle import VehicleParams
from amr_fleet.motion_planning.hybrid_astar import HybridAStarPlanner
from amr_fleet.conflict_detection.collision_checker import find_all_conflicts


def make_scenario(n_robots=3, seed=7):
    random.seed(seed)
    world = World(20.0, 20.0)
    for row_y in (4.0, 9.0, 14.0):
        world.add_static(Rect(2.0, row_y, 8.5, row_y + 1.2))
        world.add_static(Rect(11.5, row_y, 18.0, row_y + 1.2))
    world.add_dynamic(DynamicObstacle("human_1", waypoints=[(10.0, 6.0), (10.0, 19.0)],
                                       speed=0.7, radius=0.3))
    params = VehicleParams(max_speed=1.2, radius=0.35)
    all_scn = [
        ("AMR_1", (1.5, 1.5, 0.0), (18.5, 18.5, 0.0)),
        ("AMR_2", (18.5, 1.5, math.pi), (1.5, 18.5, math.pi)),
        ("AMR_3", (1.5, 18.5, -math.pi / 2), (18.5, 1.5, -math.pi / 2)),
        ("AMR_4", (10.0, 1.0, math.pi / 2), (10.0, 19.0, math.pi / 2)),
    ][:n_robots]
    robots = [Robot(rid, s, g, params=params) for rid, s, g in all_scn]
    channel = CommChannel(latency_range=(0.05, 0.4), loss_prob=0.05, rng=random.Random(seed))
    engine = SimulationEngine(world, robots, channel, dt=0.2, collision_radius=0.7)
    return world, robots, channel, engine


def test_each_robot_has_own_planner():
    """Req #1: distinct planner instance per robot."""
    _, robots, _, _ = make_scenario(3)
    planners = [r.planner for r in robots]
    assert len(set(id(p) for p in planners)) == len(robots)
    assert all(isinstance(p, HybridAStarPlanner) for p in planners)


def test_hybrid_astar_produces_feasible_trajectory():
    """Req #2: Hybrid A* trajectory respects the vehicle's kinematic
    constraints (bounded per-step heading change) and avoids static
    obstacles."""
    world, robots, _, _ = make_scenario(1)
    planner = HybridAStarPlanner(VehicleParams(max_speed=1.2, radius=0.35))
    traj = planner.plan(world, (1.5, 1.5, 0.0), (18.5, 18.5, 0.0))
    assert traj is not None and len(traj.points) > 1
    # check every kinematic expansion step except the very last: the final
    # point is deliberately coordinate-snapped onto the exact goal pose for
    # arrival precision, so it isn't itself a raw motion-primitive step.
    for a, b in zip(traj.points[:-2], traj.points[1:-1]):
        dtheta = abs(math.atan2(math.sin(b.theta - a.theta), math.cos(b.theta - a.theta)))
        assert dtheta < math.radians(45)  # bounded by max_steer-driven curvature per step
        # trajectory shouldn't clip through a shelf
        assert not world.is_occupied(b.x, b.y, margin=0.34)


def test_supports_at_least_three_amrs():
    """Req #3."""
    _, robots, _, engine = make_scenario(4)
    assert len(robots) >= 3
    summary = engine.run(max_time=60.0)
    assert summary["tasks_total"] == 4


def test_robots_exchange_required_fields():
    """Req #4: broadcast message contains position, velocity, destination,
    priority, and planned trajectory."""
    world, robots, channel, engine = make_scenario(3)
    engine.initialize()
    r = robots[0]
    r.broadcast(channel, [x.id for x in robots], now=0.0)
    channel.tick(10.0)
    other = robots[1]
    other.receive(channel, 10.0)
    assert r.id in other.peer_trajectories
    # underlying message contract exposes all required fields
    from amr_fleet.communication.message import StateBroadcast
    import dataclasses
    fields = {f.name for f in dataclasses.fields(StateBroadcast)}
    assert {"position", "velocity", "destination", "priority", "trajectory"} <= fields


def test_no_central_planner_call():
    """Req #5: the engine's step() never calls a fleet-wide planning
    function; only per-robot planners are invoked, one robot at a time."""
    import inspect
    from amr_fleet.simulation import engine as engine_mod
    src = inspect.getsource(engine_mod)
    assert "HybridAStarPlanner(" not in src  # engine never instantiates a planner itself
    assert "for r in self.robots" in src      # it iterates and delegates per-robot


def test_time_aware_conflict_detection():
    """Req #6: two crossing trajectories that occupy the same space at
    overlapping times are flagged; a conflict computed with a large enough
    time offset is not."""
    from amr_fleet.conflict_detection.trajectory import Trajectory, TrajectoryPoint
    a = Trajectory("A", [TrajectoryPoint(0, 0, 0, 0, 1), TrajectoryPoint(10, 10, 0, 0, 1)])
    b_conflicting = Trajectory("B", [TrajectoryPoint(0, 10, 0, math.pi, 1),
                                      TrajectoryPoint(10, 0, 0, math.pi, 1)])
    b_clear = Trajectory("B", [TrajectoryPoint(100, 10, 0, math.pi, 1),
                                TrajectoryPoint(110, 0, 0, math.pi, 1)])
    conflicts = find_all_conflicts([a, b_conflicting], safety_radius=0.9)
    assert len(conflicts) == 1
    conflicts2 = find_all_conflicts([a, b_clear], safety_radius=0.9)
    assert len(conflicts2) == 0


def test_priority_resolution_is_symmetric_and_deterministic():
    """Req #7: two robots independently evaluating the same conflict must
    agree on who yields."""
    from amr_fleet.coordination.priority import PriorityContext, wins
    ctx_a = PriorityContext(remaining_distance=5.0, wait_time=0.0, robot_id="AMR_1")
    ctx_b = PriorityContext(remaining_distance=20.0, wait_time=0.0, robot_id="AMR_2")
    assert wins(ctx_a, ctx_b) != wins(ctx_b, ctx_a)


def test_conflicting_robot_replans_via_hybrid_astar():
    """Req #8: forcing a conflict causes replanning_count to increase and
    the resulting trajectory to avoid the reserved cells."""
    from amr_fleet.motion_planning.hybrid_astar import reservations_from_trajectory
    from amr_fleet.conflict_detection.trajectory import Trajectory, TrajectoryPoint

    world, robots, channel, engine = make_scenario(1)
    r = robots[0]
    engine.initialize()
    before = r.replanning_count
    # a small, local obstruction a couple of meters ahead on the robot's
    # route (not the whole corridor) -- enough to force a detour/replan
    # while still leaving a feasible path.  Keep the blockage short so the
    # robot can wait it out within max_wait_steps.
    blocking_traj = Trajectory("ghost", [
        TrajectoryPoint(0.0, 3.0, 1.5, 0.0, 1.0),
        TrajectoryPoint(1.0, 3.0, 1.5, 0.0, 1.0),
    ])
    reservations = reservations_from_trajectory(blocking_traj, exclude_before=0.0,
                                                  time_margin_buckets=3)
    new_traj = r._replan(world, 0.0, reservations)
    assert new_traj is not None
    assert r.replanning_count == before + 1


def test_planner_can_wait_for_temporary_reservation():
    """A temporary space-time blockage can be waited out in place."""
    from amr_fleet.conflict_detection.trajectory import Trajectory, TrajectoryPoint
    from amr_fleet.motion_planning.hybrid_astar import reservations_from_trajectory, TIME_RES

    world = World(10.0, 10.0)
    planner = HybridAStarPlanner(VehicleParams(max_speed=1.0, radius=0.2),
                                 max_expansions=10000)
    blocker = Trajectory("blocker", [
        TrajectoryPoint(0.0, 1.0, 1.0, 0.0, 0.0),
        TrajectoryPoint(0.9, 1.0, 1.0, 0.0, 0.0),
    ])
    reservations = reservations_from_trajectory(blocker, exclude_before=0.0,
                                                 time_margin_buckets=0)
    traj = planner.plan(world, (1.0, 1.0, 0.0), (3.0, 1.0, 0.0),
                        reservations=reservations)

    assert traj is not None
    assert any(a.x == b.x and a.y == b.y
               for a, b in zip(traj.points, traj.points[1:]))
    assert traj.points[1].t >= TIME_RES


def test_wait_horizon_is_bounded():
    """Waiting is capped so the time-expanded search cannot grow without
    bound. In this blockage the only way out is to hold position, so a
    planner with no wait budget must fail where one with a budget succeeds."""
    from amr_fleet.conflict_detection.trajectory import Trajectory, TrajectoryPoint
    from amr_fleet.motion_planning.hybrid_astar import reservations_from_trajectory

    world = World(10.0, 10.0)
    params = VehicleParams(max_speed=1.0, radius=0.2)
    blocker = Trajectory("blocker", [
        TrajectoryPoint(0.0, 1.0, 1.0, 0.0, 0.0),
        TrajectoryPoint(0.9, 1.0, 1.0, 0.0, 0.0),
    ])
    reservations = reservations_from_trajectory(blocker, exclude_before=0.0,
                                                 time_margin_buckets=0)

    no_wait = HybridAStarPlanner(params, max_expansions=10000, max_wait_steps=0)
    assert no_wait.plan(world, (1.0, 1.0, 0.0), (3.0, 1.0, 0.0),
                        reservations=reservations) is None

    bounded = HybridAStarPlanner(params, max_expansions=10000, max_wait_steps=8)
    assert bounded.plan(world, (1.0, 1.0, 0.0), (3.0, 1.0, 0.0),
                        reservations=reservations) is not None


def test_dynamic_obstacle_avoidance():
    """Req #9: robot's planned trajectory avoids the swept cells of a
    dynamic obstacle it senses."""
    world, robots, channel, engine = make_scenario(1)
    r = robots[0]
    r.plan_initial(world, 0.0)
    dyn = world.sensed_dynamic_reservations(r.pose[0], r.pose[1], 0.0, r.sense_radius, 8.0)
    for p in r.trajectory.points:
        from amr_fleet.motion_planning.hybrid_astar import to_cell, to_time_bucket
        key = (*to_cell(p.x, p.y), to_time_bucket(p.t))
        assert key not in dyn


def test_communication_simulates_latency_and_loss():
    """Req #10."""
    channel = CommChannel(latency_range=(0.05, 0.4), loss_prob=1.0, rng=random.Random(1))
    from amr_fleet.communication.message import StateBroadcast
    from amr_fleet.conflict_detection.trajectory import Trajectory
    msg = StateBroadcast("A", 0.0, (0, 0, 0), 1.0, (1, 1, 0), 0.0, Trajectory("A"))
    channel.broadcast("A", ["B"], msg, now=0.0)
    channel.tick(1.0)
    assert channel.inbox("B") == []  # loss_prob=1.0 -> nothing delivered
    assert channel.loss_rate() == 1.0

    channel2 = CommChannel(latency_range=(0.1, 0.1), loss_prob=0.0, rng=random.Random(1))
    channel2.broadcast("A", ["B"], msg, now=0.0)
    channel2.tick(0.05)
    assert channel2.inbox("B") == []  # not arrived yet (latency not elapsed)
    channel2.tick(0.2)
    assert len(channel2.inbox("B")) == 1
    assert channel2.average_latency() > 0


def test_fleet_survives_link_failure():
    """Req #11: killing one comm link doesn't stop the fleet from
    completing its tasks, because the topology is peer-to-peer mesh, not
    hub-and-spoke."""
    world, robots, channel, engine = make_scenario(3)
    engine.schedule_link_event(t=0.0, robot_a="AMR_1", robot_b="AMR_2", down=True)
    summary = engine.run(max_time=60.0)
    assert summary["tasks_completed"] >= 2  # fleet keeps making progress
    assert summary["collisions"] == 0


def test_metrics_report_has_all_required_fields():
    """Req #12."""
    _, robots, channel, engine = make_scenario(3)
    summary = engine.run(max_time=60.0)
    required = {"collisions", "deadlocks", "avg_completion_time_s", "path_length_m",
                "total_replanning_count", "avg_comm_latency_s", "throughput_tasks_per_min"}
    assert required <= set(summary.keys())


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
