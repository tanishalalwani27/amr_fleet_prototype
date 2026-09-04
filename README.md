# Edge-AI Distributed Fleet Coordination for AMRs — Prototype

A working, decentralized multi-AMR warehouse simulation: each robot plans
its own Hybrid A* trajectory, robots exchange state over a simulated lossy
P2P network, and conflicts are resolved locally using a CBS-inspired
priority protocol — with no central fleet planner anywhere.

## A note on the two reference repos

Neither repo actually does what "repository A" / "repository B" implied,
so here's exactly what was reused and what wasn't:

- **`ehe3/Multi-Agent-Pathfinding`** is a *grid-based* MAPF solver (SAT
  solver + A* + independence detection), not a single-AMR Hybrid A*
  planner. Its useful idea — **detect conflicts, then have the losing
  agent replan around them** — is exactly what `coordination/resolver.py`
  does, just decentralized (each robot decides for itself) instead of a
  central independence-detection loop.
- **`maximaximal/Paracooba`** is a distributed cube-and-conquer *SAT
  solver*, not a robotics repo at all. Its relevant architectural idea —
  **peer-to-peer work distribution with no single coordinating master,
  tolerant of nodes coming and going** — is what `communication/channel.py`
  and the "no central planner" rule in `simulation/engine.py` are modeled
  after.
- Neither repo implements **Hybrid A*** or continuous-space kinematic
  planning at all, so `motion_planning/hybrid_astar.py` is written from
  scratch (standard bicycle-model Hybrid A*, per requirement #2).

I called this out up front rather than silently pretending the repos
matched the ask — happy to swap in actual code from those repos anywhere
you want it more directly reused (e.g. their SAT-based low-level search
in place of Hybrid A*'s BFS heuristic), just say where.

## Architecture (strict separation, per your requirement list)

```
amr_fleet/
  motion_planning/     # Hybrid A* + vehicle kinematics. Owned 1:1 by each robot.
    vehicle.py           bicycle model, motion primitives
    hybrid_astar.py       the planner + space-time reservation conversion
  conflict_detection/  # Pure geometry/time reasoning. No comms, no priorities.
    trajectory.py          shared Trajectory/TrajectoryPoint data contract
    collision_checker.py   time-aware pairwise conflict detection
  communication/       # Simulated P2P radio. No planning logic lives here.
    message.py             StateBroadcast: position, velocity, destination,
                            priority, planned trajectory
    channel.py              per-link latency, packet loss, link-down/up
  coordination/        # Decides WHO must yield and WHAT they must avoid.
    priority.py             deterministic, symmetric priority function
    resolver.py              CBS-inspired: detect -> decide -> constrain
  simulation/           # Orchestration only. Never plans for >1 robot.
    world.py                static map + dynamic (non-communicating) obstacles
    robot.py                one AMR: planner + comms + coordination + motion
    engine.py                tick loop: receive -> coordinate -> move -> broadcast
    metrics.py                collisions, deadlocks, completion time, path
                              length, replans, latency, throughput
  visualization/         # Reads recorded history only. No sim/logic access.
    plotter.py                GIF/MP4 animation + metrics bar charts
main.py                  # scenario setup / entry point
tests/test_requirements.py  # one test per numbered requirement
```

Requirement → code map:

| # | Requirement | Where |
|---|---|---|
| 1 | Each AMR has its own local planner | `Robot.__init__` creates its own `HybridAStarPlanner` instance |
| 2 | Hybrid A* for feasible trajectories | `motion_planning/hybrid_astar.py` (bicycle model, bounded steering) |
| 3 | ≥3 AMRs | `main.py` runs 4; `engine.py` has no robot-count limit |
| 4 | Exchange position/velocity/destination/priority/trajectory | `communication/message.py::StateBroadcast` |
| 5 | No central fleet-wide planner | `engine.py` only ever calls `robot.coordinate()`/`plan_initial()` per robot; `test_no_central_planner_call` asserts this |
| 6 | Time-aware collision checking | `conflict_detection/collision_checker.py::find_conflict` |
| 7 | Distributed priority/CBS-inspired resolution | `coordination/priority.py` + `coordination/resolver.py` |
| 8 | Conflicting robot locally replans via Hybrid A* | `Robot.coordinate()` → `Robot._replan()` → own `HybridAStarPlanner.plan()` |
| 9 | Dynamic obstacles | `simulation/world.py::DynamicObstacle` + sensed (not broadcast) reservations |
| 10 | Simulated latency/loss | `communication/channel.py` (`latency_range`, `loss_prob`) |
| 11 | Fleet survives one dead connection | `channel.set_link_down()` kills one pair's link only; mesh topology, no hub |
| 12 | Metrics | `simulation/metrics.py::MetricsCollector` |

## Running it

```bash
python3 main.py                       # runs the 4-AMR warehouse scenario,
                                       # prints the metrics report, writes
                                       # fleet_simulation.gif + metrics_summary.png
python3 -m pytest tests/ -v           # or: python3 tests/test_requirements.py
```

## How the decentralized conflict resolution actually works

1. Every robot periodically broadcasts a `StateBroadcast` (its own plan)
   over the lossy P2P channel — no server relays or stores it.
2. Each robot independently checks its own planned trajectory against
   whatever peer trajectories it has *currently and recently* received
   (`conflict_detection.find_conflict`).
3. If a conflict is found, both robots involved can compute a shared,
   deterministic `priority` value from public information (remaining
   distance + how long they've been waiting, tie-broken by ID). Whoever
   would lose recomputes its own trajectory with the other robot's
   trajectory converted into temporary space-time "reservations"
   (`reservations_from_trajectory`) — exactly CBS's "add constraint,
   replan single agent" step, just executed unilaterally by the
   constrained robot instead of a central search tree.
4. The `wait_time`-based priority aging guarantees a robot that keeps
   losing conflicts eventually outranks everyone nearby — a basic
   distributed anti-starvation/anti-livelock measure.

## Known limitations / honest caveats in this prototype

- **Wait transitions are bucketed and capped.** The planner can hold position
  when space-time reservations are present, advancing one reservation bucket at
  a time, up to `max_wait_steps` consecutive waits. A robot already standing
  *inside* a reservation is permitted to wait there — it may never move into
  reserved space, only out of it — which is what stops a temporarily blocked
  robot from having no legal transition at all.
- **Priority is *not* actually symmetric over the wire, despite §3 above.**
  `Robot.broadcast` sends `priority=0.0` hardcoded, and `Robot.receive`
  defaults every peer's `wait_time` to `0.0`. So the anti-starvation aging
  term is private to each robot: a robot that has been yielding for 10 s still
  looks like a zero-wait peer to everyone else, and two robots *can* disagree
  about who wins a conflict (both yield, or neither does). Transmitting
  `wait_time`/`priority` in `StateBroadcast` is a small, high-value fix.
- **Dynamic obstacles are sensed with a simple radius + straight
  forward-projection**, not a real sensor model or predictive filter.
- **Priority is recomputed from broadcast state**, not cryptographically
  signed/authenticated — fine for a prototype, not for a real fleet.
- **Deadlock detection is windowed-displacement based**, which is simple
  and works, but a smarter wait-graph / cycle-detection approach would
  catch some deadlocks earlier.

## Roadmap: basic → pro

**Stage 1 (done here): correctness skeleton.** Decentralized Hybrid A*,
time-aware conflict detection, priority-based local replanning, simulated
lossy comms, metrics, visualization. Good for demonstrating every
requirement and for unit-testing each module in isolation.

**Stage 2: planning quality.**
- ~~Add a wait/stay-in-place primitive to Hybrid A*~~ **Done, and covered by
  tests.** The zero-length step is reservation-aware and bounded by
  `max_wait_steps`. The bug it took to get there: a robot whose *current* cell
  was already reserved had no legal transition whatsoever — waiting was
  rejected because its own cell was reserved, and stepping out was rejected
  because the ±1-cell footprint margin still overlapped it — so `plan()`
  returned `None` and the robot went `DEADLOCKED` on a purely temporary
  blockage. Waits are now exempt for the cell the robot already occupies, and
  only for waits: moving into reserved space is still never allowed.
- Replace the Euclidean/BFS heuristic with a proper **Reeds-Shepp /
  non-holonomic-without-obstacles heuristic** (as in the original Hybrid
  A* paper) for faster, higher-quality search in open areas.
- Add a **velocity profile / trapezoidal acceleration model** instead of
  constant-speed segments, so trajectories (and their time-stamps) are
  actually dynamically feasible, not just kinematically feasible.

**Stage 3: coordination robustness.**
- Move from "avoid the whole trajectory" reservations to genuine
  **windowed cooperative replanning** (WHCA*-style): only reserve a
  rolling time horizon so replans are cheaper and less prone to
  over-conservatism. **This is now the top remaining item.** Measured after
  the wait fix — 105 replans across 6 runs of the stock 4-AMR and a congested
  6-AMR scenario — the wait primitive fires almost never (1 wait step total):
  robots always detour spatially instead, because the reservations they detour
  around are inflated several times over. Three specific defects to fix as
  part of this work:
  - `reservations_from_trajectory` samples only a trajectory's discrete
    points, so a sparse plan leaves **unreserved gaps** between them — unsafe
    in the opposite direction, and it is why a blocker sitting still for 0.9 s
    reserves buckets 0 and 2 but not 1.
  - `to_time_bucket` uses `round()`, i.e. **banker's rounding**, so at
    `max_speed=1.2` (0.4167 s per step) consecutive steps can land in the
    *same* bucket and the time axis stops being monotonic. `floor` is correct.
  - The caller's `time_margin_buckets=1` **stacks on top of** the `3×3×3`
    neighbourhood in `_collides_dynamic`, giving an effective ~±1.0 s and
    ~±0.7 m inflation around every peer. Pick one place to own the margin.
- Add **true CBS-style constraint negotiation** between just the two
  conflicting robots (a 2-agent local CBS solve) before falling back to
  unilateral priority-based yielding — better solution quality when both
  robots are reachable.
- **Deadlock/cycle detection** via a distributed wait-for graph
  (each robot tracks who it's yielding to; a cycle triggers a
  coordinated priority perturbation) instead of only displacement-based
  detection.

**Stage 4: communication realism.**
- Model **bandwidth limits and message size** (trajectories are large;
  real radios can't broadcast a full plan every 0.5 s at scale) — send
  deltas/compressed corridors instead of full point lists.
- Add a **gossip/relay protocol** so robots without a direct link can
  still learn of each other's plans via a third robot, instead of only
  direct pairwise links.
- Model **partition scenarios** (a whole subset of the fleet cut off, not
  just one link) and verify sub-fleets keep operating safely and merge
  state cleanly on reconnect.

**Stage 5: production-shape.**
- Swap the toy `World` occupancy grid for a real warehouse map
  format (e.g. exported from a fleet's SLAM map), with inflation layers
  per robot footprint.
- Move metrics from in-memory to a **time-series store** (Prometheus/
  InfluxDB) for live dashboards instead of a post-hoc report.
- Add **task allocation** on top (currently each robot has a fixed single
  destination) — auction-based or market-based decentralized task
  assignment, so this becomes a full pick-and-deliver fleet system.
- Port the local planner to something **real-time-capable in C++/Rust**
  for actual edge hardware, keeping the same module boundaries so this
  Python version stays useful as the reference/testbed implementation.

The wait primitive is done. The highest-value next step is the **windowed
WHCA*-style reservation scheme** in Stage 3 — it is what actually limits
solution quality right now, and the three reservation defects listed there are
prerequisites for it. The **priority-transmission fix** in Known limitations is
much smaller and independently worthwhile, since the protocol's symmetry claim
does not currently hold over the wire.
