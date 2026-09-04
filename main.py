"""
Scenario setup and entry point.

Run:
    python3 main.py

This builds a small warehouse with shelving (static obstacles), one
moving human/forklift (dynamic obstacle), and 4 AMRs whose paths are
designed to conflict, then runs the decentralized simulation, prints the
metrics report, and renders a video + a metrics summary chart.
"""

import math
import os
import random

from amr_fleet.simulation.world import World, Rect, DynamicObstacle
from amr_fleet.simulation.robot import Robot
from amr_fleet.simulation.engine import SimulationEngine
from amr_fleet.communication.channel import CommChannel
from amr_fleet.motion_planning.vehicle import VehicleParams
from amr_fleet.visualization.plotter import animate_simulation, plot_metrics_summary


def build_world() -> World:
    world = World(width=20.0, height=20.0)
    # warehouse shelving rows, with an aisle gap in the middle so paths
    # genuinely have to cross / merge -- this is what forces conflicts.
    for row_y in (4.0, 9.0, 14.0):
        world.add_static(Rect(2.0, row_y, 8.5, row_y + 1.2))
        world.add_static(Rect(11.5, row_y, 18.0, row_y + 1.2))

    # one dynamic obstacle patrolling the central aisle (e.g. a person),
    # offset from any robot's start pose so it isn't co-located at t=0
    world.add_dynamic(DynamicObstacle(
        obstacle_id="human_1",
        waypoints=[(10.0, 6.0), (10.0, 19.0)],
        speed=0.7, radius=0.3))
    return world


def build_robots() -> list:
    params = VehicleParams(max_speed=1.2, radius=0.35)
    scenarios = [
        ("AMR_1", (1.5, 1.5, 0.0), (18.5, 18.5, 0.0)),
        ("AMR_2", (18.5, 1.5, math.pi), (1.5, 18.5, math.pi)),
        ("AMR_3", (1.5, 18.5, -math.pi / 2), (18.5, 1.5, -math.pi / 2)),
        ("AMR_4", (10.0, 1.0, math.pi / 2), (10.0, 19.0, math.pi / 2)),
    ]
    return [Robot(rid, start, goal, params=params) for rid, start, goal in scenarios]


def main():
    random.seed(7)
    world = build_world()
    robots = build_robots()

    # latency up to 400ms, 5% packet loss -- deliberately imperfect comms
    channel = CommChannel(latency_range=(0.05, 0.4), loss_prob=0.05, rng=random.Random(7))

    engine = SimulationEngine(world, robots, channel, dt=0.2, collision_radius=0.7)

    # Demonstrate requirement #11: kill the AMR_1<->AMR_2 link for a while;
    # the rest of the mesh (and AMR_1/AMR_2's own local replanning) keeps
    # the fleet operating regardless.
    engine.schedule_link_event(t=5.0, robot_a="AMR_1", robot_b="AMR_2", down=True)
    engine.schedule_link_event(t=15.0, robot_a="AMR_1", robot_b="AMR_2", down=False)

    print("Running simulation...")
    summary = engine.run(max_time=90.0)

    engine.metrics.print_report(engine.robots, engine.channel, engine.t)

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    print("Rendering visualization...")
    try:
        animate_simulation(engine, os.path.join(out_dir, "fleet_simulation.gif"), fps=12, stride=2)
    except Exception as e:
        print(f"(video render skipped: {e})")
    try:
        plot_metrics_summary(summary, os.path.join(out_dir, "metrics_summary.png"))
    except Exception as e:
        print(f"(metrics plot skipped: {e})")
    print(f"Done. Outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
