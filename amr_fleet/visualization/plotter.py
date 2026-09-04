"""
Visualization. Reads only the recorded `engine.history` (+ static world
layout) -- it has no access to, and makes no calls into, any planner,
comms channel, or coordination logic. Purely a renderer.
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


ROBOT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

# status colour used for the robot marker face; edge colour stays the robot id
STATUS_COLORS = {
    "PLANNING": "#9b59b6",   # purple
    "MOVING": "#2ecc71",      # green
    "YIELDING": "#f39c12",    # amber
    "DEADLOCKED": "#e74c3c",  # red
    "DONE": "#3498db",        # blue
    "FAILED": "#7f8c8d",      # gray
}


def _draw_world(ax, world):
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    for r in world.static_obstacles:
        ax.add_patch(patches.Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0,
                                        facecolor="#555555", edgecolor="black", zorder=1))


def animate_simulation(engine, out_path: str, fps: int = 15, stride: int = 1):
    """Render the recorded simulation to an mp4/gif at out_path."""
    world = engine.world
    history = engine.history[::stride]
    robot_ids = list(engine.robots.keys())
    colors = {rid: ROBOT_COLORS[i % len(ROBOT_COLORS)] for i, rid in enumerate(robot_ids)}

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_world(ax, world)

    robot_artists = {}
    trail_artists = {}
    plan_artists = {}
    trails = {rid: ([], []) for rid in robot_ids}
    for rid in robot_ids:
        # planned trajectory (dashed line, current intent)
        (plan,) = ax.plot([], [], "--", color=colors[rid], alpha=0.5, linewidth=1.5)
        plan_artists[rid] = plan
        # executed trail
        (trail,) = ax.plot([], [], "-", color=colors[rid], alpha=0.35, linewidth=2)
        trail_artists[rid] = trail
        # robot marker: face colour = status, edge colour = robot id colour
        (patch,) = ax.plot([], [], marker=(3, 0, 0), markersize=16,
                            markerfacecolor=STATUS_COLORS["PLANNING"],
                            markeredgecolor=colors[rid], markeredgewidth=2,
                            linestyle="None", label=rid)
        robot_artists[rid] = patch
        gx, gy = engine.robots[rid].goal[0], engine.robots[rid].goal[1]
        ax.plot(gx, gy, marker="*", markersize=14, color=colors[rid], markeredgecolor="black")

    dyn_artists = {}
    for obs in world.dynamic_obstacles:
        (patch,) = ax.plot([], [], "o", color="black", markersize=10)
        dyn_artists[obs.obstacle_id] = patch

    title = ax.set_title("t = 0.0 s")
    # two legends: robot ids and status colours
    robot_legend = ax.legend(handles=[robot_artists[rid] for rid in robot_ids],
                              loc="upper right", fontsize=8, title="Robot")
    ax.add_artist(robot_legend)
    status_handles = [patches.Patch(color=c, label=s) for s, c in STATUS_COLORS.items()]
    ax.legend(handles=status_handles, loc="lower right", fontsize=7, title="Status")

    def update(frame_idx):
        snap = history[frame_idx]
        title.set_text(f"t = {snap['t']:.1f} s")
        for rid, data in snap["robots"].items():
            x, y, theta = data["pose"]
            robot_artists[rid].set_data([x], [y])
            robot_artists[rid].set_marker((3, 0, math.degrees(theta) - 90))
            robot_artists[rid].set_markerfacecolor(STATUS_COLORS.get(data["status"], "#7f8c8d"))
            trails[rid][0].append(x)
            trails[rid][1].append(y)
            trail_artists[rid].set_data(trails[rid][0], trails[rid][1])
            traj = data.get("trajectory")
            if traj is not None and not traj.is_empty():
                xs = [p.x for p in traj.points]
                ys = [p.y for p in traj.points]
                plan_artists[rid].set_data(xs, ys)
            else:
                plan_artists[rid].set_data([], [])
        for oid, (x, y) in snap["dynamic_obstacles"].items():
            dyn_artists[oid].set_data([x], [y])
        return (list(robot_artists.values()) + list(trail_artists.values()) +
                list(plan_artists.values()) + list(dyn_artists.values()))

    anim = animation.FuncAnimation(fig, update, frames=len(history), blit=False,
                                    interval=1000 / fps)
    if out_path.endswith(".gif"):
        anim.save(out_path, writer="pillow", fps=fps)
    else:
        anim.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)


def plot_metrics_summary(summary: dict, out_path: str):
    """Static bar-chart summary of the metrics report, for a quick visual
    read without opening the video."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    rids = list(summary["path_length_m"].keys())

    ax = axes[0, 0]
    ax.bar(rids, [summary["path_length_m"][r] for r in rids], color="#1f77b4")
    ax.set_title("Path length per robot (m)")

    ax = axes[0, 1]
    ax.bar(rids, [summary["replanning_count_per_robot"][r] for r in rids], color="#d62728")
    ax.set_title("Replanning count per robot")

    ax = axes[0, 2]
    labels = ["Collisions", "Deadlocks"]
    vals = [summary["collisions"], summary["deadlocks"]]
    ax.bar(labels, vals, color=["#e74c3c", "#f39c12"])
    ax.set_title("Safety events")

    ax = axes[1, 0]
    ax.bar(rids, [summary["yield_count_per_robot"][r] for r in rids], color="#9467bd")
    ax.set_title("Yield events per robot")

    ax = axes[1, 1]
    ax.bar(rids, [summary["wait_steps_per_robot"][r] for r in rids], color="#2ca02c")
    ax.set_title("Wait steps per robot")

    ax = axes[1, 2]
    ax.axis("off")
    text = f"Tasks completed: {summary['tasks_completed']}/{summary['tasks_total']}\n"
    if summary['avg_completion_time_s']:
        text += f"Avg completion time: {summary['avg_completion_time_s']:.2f} s\n"
    text += (
        f"Total yields: {summary['total_yield_count']}\n"
        f"Total waits: {summary['total_wait_steps']}\n"
        f"Total conflicts: {summary['total_conflict_count']}\n"
        f"Throughput: {summary['throughput_tasks_per_min']:.2f} tasks/min\n"
        f"Avg comm latency: {summary['avg_comm_latency_s']*1000:.1f} ms\n"
        f"Comm loss rate: {summary['comm_loss_rate']*100:.1f} %\n"
        f"Sim time: {summary['sim_time_s']:.1f} s"
    )
    ax.text(0.05, 0.5, text, fontsize=11, va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
