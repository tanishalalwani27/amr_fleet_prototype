"""
Hybrid A* — the local, per-robot motion planner.

Each robot runs its OWN instance of this planner. It never plans for any
other robot, and it has no notion of "the fleet". It optionally accepts a
set of space-time reservations (cells other robots are known to occupy at
particular times) so that a robot can replan *around* a conflict without
any central authority computing the fleet-wide solution -- the reservations
are just extra, temporary obstacles from this one robot's point of view.

This keeps requirement #5 (no central fleet-wide planner) intact: the
"conflict-aware" behaviour lives entirely inside a single robot's local
planning call.
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from amr_fleet.motion_planning.vehicle import VehicleParams, bicycle_step, motion_primitives
from amr_fleet.conflict_detection.trajectory import Trajectory, TrajectoryPoint


# ---------------------------------------------------------------------------
# Grid / discretization helpers
# ---------------------------------------------------------------------------

XY_RES = 0.5          # meters per grid cell (for closed-set discretization & collision lookup)
THETA_RES = math.radians(15)  # heading bins
TIME_RES = 0.5         # seconds per time bucket, for space-time reservations


def to_cell(x, y):
    return (int(round(x / XY_RES)), int(round(y / XY_RES)))


def to_theta_bin(theta):
    theta = math.atan2(math.sin(theta), math.cos(theta))
    return int(round(theta / THETA_RES)) % int(round(2 * math.pi / THETA_RES))


def to_time_bucket(t):
    # floor, not round: round() is banker's rounding in Python, which made the
    # bucket sequence non-monotonic for step times that aren't multiples of
    # TIME_RES (e.g. max_speed=1.2 -> 0.4167s per step).
    return int(math.floor(t / TIME_RES))


@dataclass(order=True)
class _Node:
    f: float
    g: float = field(compare=False)
    pose: Tuple[float, float, float] = field(compare=False)
    t: float = field(compare=False)
    parent: Optional["_Node"] = field(compare=False, default=None)
    waits: int = field(compare=False, default=0)  # consecutive waits on this branch


class HybridAStarPlanner:
    """
    A self-contained Hybrid A* planner for ONE robot.

    Static obstacles come from the world's occupancy grid.
    Dynamic constraints come from `reservations`: a set of
    (cell_x, cell_y, time_bucket) tuples that this robot must avoid --
    these represent OTHER robots' broadcast trajectories (with a safety
    margin already baked in by the caller) or moving obstacles.
    """

    def __init__(self, params: VehicleParams = None, n_steer: int = 7,
                 max_expansions: int = 40000, max_wait_steps: int = 8,
                 safety_radius: float = None):
        self.params = params or VehicleParams()
        self.n_steer = n_steer
        self.max_expansions = max_expansions
        self.max_wait_steps = max_wait_steps
        self.safety_radius = (safety_radius if safety_radius is not None
                              else 2 * self.params.radius + 0.2)
        # Two positions whose cells are Chebyshev distance d apart can be as
        # little as (d-1)*XY_RES apart in continuous space, so rejecting
        # everything within m cells guarantees m*XY_RES of separation for any
        # transition we accept. This margin MUST agree with the radius conflict
        # detection uses -- if the planner accepts a path that find_conflict
        # then rejects, every replan reproduces the conflict it was meant to
        # remove and the robot replans every tick without ever changing course.
        self.res_margin = max(1, int(math.ceil(self.safety_radius / XY_RES)))

    # -- collision checks ---------------------------------------------------

    def _collides_static(self, world, x, y):
        return world.is_occupied(x, y, margin=self.params.radius)

    def _collides_dynamic(self, x, y, t, reservations):
        """True if (x, y) at time t falls inside a reservation.

        This function OWNS the discretization margin: it tests a
        (2*res_margin+1)^2 cell neighbourhood and a ±1 bucket time
        neighbourhood. That margin cannot be dropped, because `to_cell` and
        `to_time_bucket` both quantize -- two positions 0.1 m apart can land in
        different cells, and two times 0.02 s apart can land in different
        buckets. Callers must therefore NOT add their own inflation on top of
        it (that is why `reservations_from_trajectory`'s `time_margin_buckets`
        defaults to 0).
        """
        if not reservations:
            return False
        cell = to_cell(x, y)
        tb = to_time_bucket(t)
        m = self.res_margin
        # check neighbouring cells too (footprint safety margin)
        for dx in range(-m, m + 1):
            for dy in range(-m, m + 1):
                for dt in (-1, 0, 1):
                    if (cell[0] + dx, cell[1] + dy, tb + dt) in reservations:
                        return True
        return False

    def _state_key(self, pose, t, include_time=False):
        """Discretize a search state, retaining time for wait transitions."""
        key = (to_cell(pose[0], pose[1]), to_theta_bin(pose[2]))
        if include_time:
            return key + (to_time_bucket(t),)
        return key

    # -- heuristic: BFS distance-to-goal on the occupancy grid (ignores heading) --

    def _build_heuristic(self, world, goal_xy):
        """Dijkstra/BFS over grid cells from the goal, ignoring headings.
        This lets the search find its way around shelving even though the
        raw Euclidean distance would be a poor guide near obstacles."""
        import collections
        gx, gy = to_cell(*goal_xy)
        dist = {(gx, gy): 0.0}
        pq = [(0.0, (gx, gy))]
        while pq:
            d, cell = heapq.heappop(pq)
            if d > dist.get(cell, math.inf):
                continue
            cx, cy = cell
            for dx, dy, cost in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)):
                ncell = (cx + dx, cy + dy)
                wx, wy = ncell[0] * XY_RES, ncell[1] * XY_RES
                if world.is_occupied(wx, wy, margin=self.params.radius):
                    continue
                nd = d + cost * XY_RES
                if nd < dist.get(ncell, math.inf):
                    dist[ncell] = nd
                    heapq.heappush(pq, (nd, ncell))
        return dist

    def _heuristic(self, dist_map, x, y, goal_xy):
        cell = to_cell(x, y)
        if cell in dist_map:
            return dist_map[cell]
        # fall back to euclidean if BFS didn't reach this (e.g. cut off) cell
        return math.hypot(goal_xy[0] - x, goal_xy[1] - y)

    # -- main search ----------------------------------------------------------

    def plan(self, world, start_pose, goal_pose, start_time: float = 0.0,
             reservations: Optional[Set[Tuple[int, int, int]]] = None,
             speed: Optional[float] = None) -> Optional[Trajectory]:
        """
        Plan a kinematically-feasible, time-stamped trajectory from
        start_pose (x,y,theta) to goal_pose (x,y,theta), starting at
        start_time, avoiding static obstacles in `world` and the given
        space-time `reservations`.

        Returns a Trajectory, or None if no path was found.
        """
        speed = speed or self.params.max_speed
        goal_xy = (goal_pose[0], goal_pose[1])
        dist_map = self._build_heuristic(world, goal_xy)
        prims = motion_primitives(self.params, self.n_steer)
        # Waiting is only useful when space-time constraints are present. It
        # advances one reservation bucket without changing the pose.
        if reservations:
            prims.append((0.0, 0.0))

        start = _Node(f=0.0, g=0.0, pose=start_pose, t=start_time, parent=None)
        start.f = self._heuristic(dist_map, *start_pose[:2], goal_xy)
        open_heap = [start]
        best_g: Dict[Tuple, float] = {}

        goal_xy_thresh = XY_RES * 1.5
        goal_theta_thresh = THETA_RES * 3

        # Nothing beyond the last reserved bucket can constrain the robot, so
        # past that point time is dropped from the closed-set key. Keying the
        # (usually long) unconstrained tail by arrival time would make every
        # arrival time a distinct state and exhaust max_expansions before the
        # goal is ever reached.
        res_end_bucket = (max(tb for (_, _, tb) in reservations) + 1
                          if reservations else -1)

        expansions = 0
        while open_heap and expansions < self.max_expansions:
            node = heapq.heappop(open_heap)
            expansions += 1
            x, y, theta = node.pose

            if (math.hypot(x - goal_pose[0], y - goal_pose[1]) < goal_xy_thresh and
                    not self._collides_dynamic(x, y, node.t, reservations)):
                return self._reconstruct(node, goal_pose, speed)

            time_expanded = to_time_bucket(node.t) <= res_end_bucket
            key = self._state_key(node.pose, node.t, include_time=time_expanded)
            if best_g.get(key, math.inf) < node.g - 1e-6:
                continue

            # A robot already inside a reservation may hold position there.
            # Waiting claims no new space -- the footprint is identical, only
            # the time advances -- so it cannot make an existing conflict
            # worse. Without this rule such a robot has no legal transition at
            # all: every neighbouring cell also falls inside the reservation
            # margin, so plan() returns None even for a purely temporary
            # blockage. max_wait_steps stops it sitting there indefinitely.
            may_hold = self._collides_dynamic(x, y, node.t, reservations)

            for steer, step_len in prims:
                npose = bicycle_step(node.pose, steer, step_len, self.params.wheelbase)
                nx, ny, ntheta = npose
                if self._collides_static(world, nx, ny):
                    continue
                is_wait = abs(step_len) < 1e-9
                if is_wait:
                    if node.waits >= self.max_wait_steps:
                        continue
                    nwaits = node.waits + 1
                else:
                    nwaits = 0
                travel_time = TIME_RES if is_wait else abs(step_len) / max(speed, 1e-3)
                nt = node.t + travel_time
                # holding position is exempt; moving into reserved space never is
                if not (is_wait and may_hold) and self._collides_dynamic(nx, ny, nt, reservations):
                    continue
                nkey = self._state_key(npose, nt,
                                       include_time=to_time_bucket(nt) <= res_end_bucket)
                ng = node.g + abs(step_len) + (0.5 if step_len < 0 else 0.0)  # penalize reverse
                if is_wait:
                    ng += TIME_RES * speed
                if ng >= best_g.get(nkey, math.inf) - 1e-6:
                    continue
                best_g[nkey] = ng
                h = self._heuristic(dist_map, nx, ny, goal_xy)
                child = _Node(f=ng + h, g=ng, pose=npose, t=nt, parent=node, waits=nwaits)
                heapq.heappush(open_heap, child)

        return None  # no feasible trajectory found

    def _reconstruct(self, node, goal_pose, speed):
        chain = []
        n = node
        while n is not None:
            chain.append(n)
            n = n.parent
        chain.reverse()
        robot_id = "unassigned"
        points = []
        for i, n in enumerate(chain):
            nxt = chain[i + 1] if i + 1 < len(chain) else None
            moved = nxt is not None and math.hypot(nxt.pose[0] - n.pose[0],
                                                   nxt.pose[1] - n.pose[1]) > 1e-9
            v = speed if moved else 0.0
            points.append(TrajectoryPoint(t=n.t, x=n.pose[0], y=n.pose[1], theta=n.pose[2], v=v))
        # snap final point exactly onto the requested goal pose
        if points:
            last = points[-1]
            points[-1] = TrajectoryPoint(t=last.t, x=goal_pose[0], y=goal_pose[1],
                                          theta=goal_pose[2], v=0.0)
        return Trajectory(robot_id=robot_id, points=points)


def reservations_from_trajectory(traj: Trajectory, exclude_before: float = -math.inf,
                                  exclude_after: float = math.inf,
                                  time_margin_buckets: int = 0) -> Set[Tuple[int, int, int]]:
    """
    Convert another robot's Trajectory into the space-time reservation set
    consumed by HybridAStarPlanner.plan(). This is how one robot's plan
    becomes a temporary "moving obstacle" for another, without any central
    planner ever seeing the whole fleet at once.

    The trajectory is *interpolated*: sampling only its discrete points left
    unreserved gaps whenever a plan was sparse (a robot sitting still for 0.9 s
    reserved the first and last bucket but not the ones between), which let a
    peer route straight through space that was in fact occupied.

    `exclude_before`/`exclude_after` clip the reservations to a time window,
    which is what makes rolling-horizon (WHCA*-style) replanning possible: a
    robot only needs to avoid what it will actually encounter before it next
    gets the chance to replan.

    `time_margin_buckets` is *extra* temporal padding on top of the ±1 bucket
    that `_collides_dynamic` already applies to absorb time discretization, so
    it defaults to 0. Setting it higher makes the robot deliberately more
    conservative about a peer's timing.
    """
    out: Set[Tuple[int, int, int]] = set()

    def reserve(x, y, t):
        if t < exclude_before or t > exclude_after:
            return
        cell = to_cell(x, y)
        tb = to_time_bucket(t)
        for dt in range(-time_margin_buckets, time_margin_buckets + 1):
            out.add((cell[0], cell[1], tb + dt))

    pts = traj.points
    for p in pts:
        reserve(p.x, p.y, p.t)

    # half a bucket per sample guarantees no bucket along a segment is skipped
    sample_dt = TIME_RES * 0.5
    for a, b in zip(pts, pts[1:]):
        span = b.t - a.t
        if span <= sample_dt:
            continue
        n = int(math.ceil(span / sample_dt))
        for i in range(1, n):
            f = i / n
            reserve(a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.t + f * span)
    return out
