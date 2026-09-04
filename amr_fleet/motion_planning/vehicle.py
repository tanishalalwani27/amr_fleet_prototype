"""
Vehicle kinematics used by the local (per-robot) motion planner.

This is deliberately isolated from everything else: conflict detection,
coordination and communication code never import this module directly.
They only ever see the *output* of planning (a Trajectory object).
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VehicleParams:
    """Physical parameters of an AMR, used to keep planned motion feasible."""

    wheelbase: float = 0.5        # meters, distance between axles (bicycle model)
    max_steer: float = math.radians(35)   # max steering angle (rad)
    step_length: float = 0.5      # meters traveled per primitive/expansion
    max_speed: float = 1.0        # m/s, nominal cruise speed
    radius: float = 0.35          # robot footprint radius, for collision checks
    allow_reverse: bool = False


def bicycle_step(pose, steer, step_length, wheelbase):
    """
    Advance a (x, y, theta) pose by one kinematic-bicycle-model step.

    pose: (x, y, theta) in meters/radians
    steer: steering angle in radians
    step_length: signed arc length to travel (negative = reverse)
    wheelbase: vehicle wheelbase in meters

    Returns the new (x, y, theta) pose. Uses a small-arc approximation
    (exact for constant steering angle over the step).
    """
    x, y, theta = pose
    if abs(steer) < 1e-6:
        nx = x + step_length * math.cos(theta)
        ny = y + step_length * math.sin(theta)
        ntheta = theta
    else:
        turning_radius = wheelbase / math.tan(steer)
        dtheta = step_length / turning_radius
        cx = x - turning_radius * math.sin(theta)
        cy = y + turning_radius * math.cos(theta)
        ntheta = theta + dtheta
        nx = cx + turning_radius * math.sin(ntheta)
        ny = cy - turning_radius * math.cos(ntheta)
    ntheta = math.atan2(math.sin(ntheta), math.cos(ntheta))
    return (nx, ny, ntheta)


def motion_primitives(params: VehicleParams, n_steer=5):
    """
    Discretized set of (steer_angle, step_length) primitives used to expand
    a Hybrid A* node. Symmetric steering angles between -max_steer..+max_steer,
    optionally with a reverse-driving mirror set.
    """
    steers = [
        -params.max_steer + i * (2 * params.max_steer) / (n_steer - 1)
        for i in range(n_steer)
    ]
    prims = [(s, params.step_length) for s in steers]
    if params.allow_reverse:
        prims += [(s, -params.step_length) for s in steers]
    return prims
