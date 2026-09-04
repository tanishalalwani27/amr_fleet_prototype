"""
Message contract exchanged between robots. This is the only information
robots share with each other -- no robot ever queries another robot's
internal planner state.
"""

from dataclasses import dataclass, replace
from typing import Optional

from amr_fleet.conflict_detection.trajectory import Trajectory


@dataclass
class StateBroadcast:
    """A single robot's periodic broadcast of its intent."""
    sender_id: str
    sent_time: float          # simulation time the message was sent
    position: tuple           # (x, y, theta)
    velocity: float           # current speed, m/s
    destination: tuple        # (x, y, theta) final goal
    priority: float           # this robot's current priority value
    trajectory: Trajectory    # full planned trajectory (space-time)
    wait_time: float = 0.0    # how long the sender has been blocked/yielding
    hops: int = 0             # 0 = heard directly, 1 = relayed by a third robot
    relayed_by: Optional[str] = None

    def relayed(self, forwarder_id: str) -> "StateBroadcast":
        """A copy of this message marked as forwarded. `sender_id` and
        `sent_time` are preserved so the recipient can tell whose plan this
        is and how stale it is -- only the delivery metadata changes."""
        return replace(self, hops=self.hops + 1, relayed_by=forwarder_id)

    def copy_delayed(self, arrival_time: float) -> "StateBroadcast":
        # messages are immutable payloads; only metadata about delivery
        # (handled by the channel) changes in transit.
        return self
