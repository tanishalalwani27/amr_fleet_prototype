"""
The AMR agent itself. This is the composition root for a single robot:
it OWNS a local Hybrid A* planner instance, talks to the shared comm
channel, and runs the local slice of the distributed coordination
protocol. There is no God object anywhere that owns multiple robots'
planners -- the SimulationEngine only ever calls methods on individual
Robot instances.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from amr_fleet.motion_planning.vehicle import VehicleParams
from amr_fleet.motion_planning.hybrid_astar import HybridAStarPlanner, TIME_RES
from amr_fleet.conflict_detection.trajectory import Trajectory, TrajectoryPoint
from amr_fleet.coordination.priority import PriorityContext, compute_priority
from amr_fleet.coordination.resolver import check_and_resolve
from amr_fleet.communication.message import StateBroadcast
from amr_fleet.communication.channel import CommChannel


class RobotStatus(Enum):
    PLANNING = auto()
    MOVING = auto()
    YIELDING = auto()      # replanning due to a lost conflict
    DEADLOCKED = auto()
    DONE = auto()
    FAILED = auto()        # planner could not find any feasible trajectory


class Robot:
    def __init__(self, robot_id: str, start_pose, goal_pose,
                 params: Optional[VehicleParams] = None,
                 safety_radius: float = 0.9,
                 sense_radius: float = 4.0,
                 broadcast_interval: float = 0.5,
                 plan_window: float = 8.0,
                 relay_gossip: bool = True,
                 deadlock_window: float = 6.0,
                 deadlock_min_progress: float = 0.3,
                 hold_duration: float = 2.0,
                 replan_interval: float = 1.0,
                 urgent_horizon: float = 2.5):
        self.id = robot_id
        self.pose = start_pose
        self.goal = goal_pose
        self.params = params or VehicleParams()
        self.safety_radius = safety_radius
        self.planner = HybridAStarPlanner(self.params, safety_radius=self.safety_radius)
        self.sense_radius = sense_radius
        self.broadcast_interval = broadcast_interval
        self.plan_window = plan_window
        self.relay_gossip = relay_gossip
        self.deadlock_window = deadlock_window
        self.deadlock_min_progress = deadlock_min_progress
        self.hold_duration = hold_duration
        self.replan_interval = replan_interval
        self.urgent_horizon = urgent_horizon

        self.velocity = 0.0
        self.trajectory: Optional[Trajectory] = None
        self.status = RobotStatus.PLANNING
        self.generation = 0

        # coordination state
        self.blocked_time = 0.0
        self.peer_trajectories: Dict[str, Trajectory] = {}
        self.peer_priority: Dict[str, PriorityContext] = {}
        self.peer_last_heard: Dict[str, float] = {}
        self.peer_sent_time: Dict[str, float] = {}
        # directly-heard foreign broadcasts awaiting one forward, keyed by
        # origin so we relay at most the latest plan per peer
        self._relay_buffer: Dict[str, StateBroadcast] = {}

        # bookkeeping / metrics
        self.replanning_count = 0
        self.path_length = 0.0
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.yield_count = 0        # conflicts where I was the one who lost
        self.conflict_count = 0     # conflicts I detected (won or lost)
        self.wait_steps = 0         # wait transitions my replans committed to
        self.relayed_count = 0      # foreign broadcasts I forwarded
        self.failed_replans = 0     # replan attempts where the planner gave up
        self._last_broadcast_t = -math.inf
        self._last_yield_replan_t = -math.inf
        self._progress_history: List[tuple] = []   # (t, x, y) samples for deadlock check

    # ---------------------------------------------------------------- utils

    def remaining_distance(self) -> float:
        return math.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1])

    def priority_context(self) -> PriorityContext:
        return PriorityContext(remaining_distance=self.remaining_distance(),
                                wait_time=self.blocked_time, robot_id=self.id)

    def reached_goal(self) -> bool:
        return self.remaining_distance() < 0.25

    # ---------------------------------------------------------------- planning

    def plan_initial(self, world, now: float):
        self.start_time = now
        self._replan(world, now, reservations=set())
        if self.trajectory is None:
            self.status = RobotStatus.FAILED
        else:
            self.status = RobotStatus.MOVING

    def _replan(self, world, now: float, reservations: set):
        dyn = world.sensed_dynamic_reservations(self.pose[0], self.pose[1], now,
                                                 self.sense_radius, horizon=8.0)
        traj = self.planner.plan(world, self.pose, self.goal, start_time=now,
                                  reservations=reservations | dyn)
        if traj is not None:
            traj.robot_id = self.id
            self.generation += 1
            traj.generation = self.generation
            self.replanning_count += 1
            self.trajectory = traj
        return traj

    def _hold_position(self, now: float):
        """Discard the current plan and commit to standing still.

        Used when replanning fails. The plan we were executing is known to
        conflict, so carrying on with it drives us into a peer we have already
        detected -- holding is the only safe fallback. The hold is broadcast
        like any other plan, so peers see a stationary robot and can route
        around it. Its finite duration is deliberate: it self-limits how often
        the expensive replan is retried, and it lets us resume the moment a
        path opens up rather than freezing for the rest of the run.
        """
        self.failed_replans += 1
        x, y, theta = self.pose
        n = max(2, int(math.ceil(self.hold_duration / TIME_RES)) + 1)
        self.trajectory = Trajectory(
            self.id,
            [TrajectoryPoint(t=now + i * TIME_RES, x=x, y=y, theta=theta, v=0.0)
             for i in range(n)],
            self.generation)
        self.velocity = 0.0

    # ---------------------------------------------------------------- comms

    def broadcast(self, channel: CommChannel, all_robot_ids: List[str], now: float):
        if now - self._last_broadcast_t < self.broadcast_interval:
            return
        self._last_broadcast_t = now
        recipients = [r for r in all_robot_ids if r != self.id]

        if self.trajectory is not None:
            ctx = self.priority_context()
            msg = StateBroadcast(sender_id=self.id, sent_time=now, position=self.pose,
                                  velocity=self.velocity, destination=self.goal,
                                  priority=compute_priority(ctx),
                                  trajectory=self.trajectory,
                                  wait_time=ctx.wait_time)
            channel.broadcast(self.id, recipients, msg, now)

        # Gossip, one hop only: forward plans we heard directly so that two
        # robots whose own link is down are not blind to each other. This is
        # what makes requirement #11 hold by design rather than by timing
        # luck. Relayed copies carry hops=1, so nobody re-forwards them.
        pending, self._relay_buffer = self._relay_buffer, {}
        for foreign in pending.values():
            # The channel's self-send guard only knows *our* id, not the id the
            # message originated from -- without this filter we would relay a
            # robot's own plan straight back to it.
            to_relay = [r for r in recipients if r != foreign.sender_id]
            if not to_relay:
                continue
            channel.broadcast(self.id, to_relay, foreign.relayed(self.id), now)
            self.relayed_count += 1

    def receive(self, channel: CommChannel, now: float):
        for msg in channel.inbox(self.id):
            # Never treat our own plan as a peer's. A relayed copy of our own
            # broadcast would otherwise land in peer_trajectories under our id
            # and conflict with itself at distance zero -- a contest we always
            # lose on the id tie-break, pinning us in an endless replan loop.
            if msg.sender_id == self.id:
                continue
            # A relayed copy travels an extra hop, so it can arrive after we
            # have already heard the same plan (or a newer one) directly.
            # Never let older information overwrite newer.
            if msg.sent_time <= self.peer_sent_time.get(msg.sender_id, -math.inf):
                continue
            self.peer_sent_time[msg.sender_id] = msg.sent_time
            self.peer_trajectories[msg.sender_id] = msg.trajectory
            self.peer_priority[msg.sender_id] = PriorityContext(
                remaining_distance=math.hypot(msg.destination[0] - msg.position[0],
                                               msg.destination[1] - msg.position[1]),
                wait_time=msg.wait_time,
                robot_id=msg.sender_id)
            self.peer_last_heard[msg.sender_id] = now
            if self.relay_gossip and msg.hops == 0:
                self._relay_buffer[msg.sender_id] = msg

    # ---------------------------------------------------------------- coordination

    def coordinate(self, world, now: float, comm_timeout: float = 3.0):
        """Check my planned trajectory against everything I've heard from
        peers, and replan locally if I must yield. This is the per-robot
        slice of the distributed, CBS-inspired protocol."""
        if self.trajectory is None or self.status in (RobotStatus.DONE, RobotStatus.FAILED):
            return

        # A plan that has fully elapsed without getting us to the goal is not a
        # plan any more. Nothing else in the loop would ever ask us to plan
        # again, so without this we would sit on the stale trajectory forever.
        pts = self.trajectory.points
        if pts and now >= pts[-1].t and not self.reached_goal():
            if self._replan(world, now, reservations=set()) is None:
                self.status = RobotStatus.DEADLOCKED
                self._hold_position(now)
                return

        # drop stale peer data (a peer we haven't heard from in a while may
        # have gone silent / out of range -- don't let it block us forever)
        fresh_peers = {pid: traj for pid, traj in self.peer_trajectories.items()
                        if now - self.peer_last_heard.get(pid, -math.inf) <= comm_timeout}

        result = check_and_resolve(self.id, self.priority_context(), self.trajectory,
                                    fresh_peers, self.peer_priority,
                                    self.safety_radius, now, horizon=self.plan_window)
        self.conflict_count += len(result.conflicts)

        if result.must_replan:
            # Plan commitment (WHCA*-style). Replanning every tick is actively
            # harmful: a robot would discard its plan faster than peers can
            # hear it, so everyone reserves against paths nobody is flying any
            # more and the fleet thrashes without making progress. Commit to
            # the current plan for replan_interval, and break that commitment
            # early only when the conflict I would lose is imminent.
            if (now - self._last_yield_replan_t < self.replan_interval and
                    result.earliest_yield_t - now > self.urgent_horizon):
                return
            self._last_yield_replan_t = now
            self.yield_count += 1
            self.status = RobotStatus.YIELDING
            new_traj = self._replan(world, now, reservations=result.reservations)
            if new_traj is None:
                # No collision-free space-time path exists right now. Stop
                # executing the plan we have just proven conflicts, stand
                # still, and try again once the hold expires -- which is also
                # what lets transient deadlocks clear as peers move on.
                self.status = RobotStatus.DEADLOCKED
                self._hold_position(now)
            else:
                self.wait_steps += sum(
                    1 for a, b in zip(new_traj.points, new_traj.points[1:])
                    if abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9)
        else:
            self.status = RobotStatus.MOVING

    # ---------------------------------------------------------------- motion

    def advance(self, now: float, dt: float):
        if self.trajectory is None or self.status in (RobotStatus.DONE, RobotStatus.FAILED):
            return
        prev_pose = self.pose
        pose = self.trajectory.pose_at(now)
        if pose is None:
            pts = self.trajectory.points
            if pts and now > pts[-1].t:
                # trajectory has fully elapsed: snap exactly onto its final
                # (goal-aligned) point instead of holding the last sampled
                # position, which could be fractionally short of the goal.
                last = pts[-1]
                pose = (last.x, last.y, last.theta)
            else:
                pose = self.pose
        travelled = math.hypot(pose[0] - prev_pose[0], pose[1] - prev_pose[1])
        # derive speed from actual motion rather than assuming cruise speed, so
        # a robot holding position for a peer reports v=0 -- it is stationary,
        # and blocked_time (which drives priority aging) depends on that
        self.velocity = travelled / dt if dt > 0 else 0.0
        self.path_length += travelled
        self.pose = pose

        # Progress tracking. blocked_time feeds priority aging, so it must
        # measure *continuous* time without progress: a running total would
        # let a robot that waited once early on outrank a genuinely blocked
        # peer for the rest of the run.
        self._progress_history.append((now, pose[0], pose[1]))
        self._progress_history = [(t, x, y) for (t, x, y) in self._progress_history
                                   if now - t <= self.deadlock_window]
        t0, x0, y0 = self._progress_history[0]
        span = now - t0
        disp = math.hypot(pose[0] - x0, pose[1] - y0)
        stalled = (self.velocity < 1e-6 or
                   (span >= self.deadlock_window * 0.9 and disp < self.deadlock_min_progress))
        if stalled and not self.reached_goal():
            self.blocked_time += dt
            # a lone wait point inside an otherwise healthy plan zeroes the
            # velocity for one tick, so require a sustained stall before
            # labelling the robot deadlocked
            if self.blocked_time >= self.deadlock_window * 0.5:
                self.status = RobotStatus.DEADLOCKED
        else:
            self.blocked_time = 0.0
            if self.status == RobotStatus.DEADLOCKED:
                self.status = RobotStatus.MOVING

        if self.reached_goal() and self.status != RobotStatus.DONE:
            self.status = RobotStatus.DONE
            self.finish_time = now
