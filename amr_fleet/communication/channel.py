"""
Simulated communication layer.

Design choice for requirement #5 / #11 ("no central server may perform
fleet-wide planning" / "fleet continues when one connection is
unavailable"): this is a fully peer-to-peer mesh. There is no broker
process that routes or plans anything -- `CommChannel` is just a shared
medium that model latency and packet loss per LINK (robot pair). Killing
one link only affects that pair; every other pair keeps exchanging
messages, and no single component's failure can stop the fleet.

Robots pull their inbox each tick; the channel never inspects message
contents or makes planning decisions.
"""

import random
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict

from amr_fleet.communication.message import StateBroadcast


@dataclass
class LinkStats:
    sent: int = 0
    delivered: int = 0
    dropped: int = 0
    latencies: List[float] = field(default_factory=list)


class CommChannel:
    """
    Peer-to-peer broadcast medium.

    latency_range: (min, max) seconds of one-way delay applied per message
    loss_prob: probability a given message is dropped in transit
    """

    def __init__(self, latency_range: Tuple[float, float] = (0.05, 0.4),
                 loss_prob: float = 0.05, rng: random.Random = None):
        self.latency_range = latency_range
        self.loss_prob = loss_prob
        self.rng = rng or random.Random(42)
        self._in_flight: List[Tuple[float, str, StateBroadcast]] = []  # heap of (arrival_t, recipient, msg)
        self._down_links: set = set()   # set of frozenset({a,b}) links that are currently down
        self.link_stats: Dict[Tuple[str, str], LinkStats] = defaultdict(LinkStats)
        self._inboxes: Dict[str, List[StateBroadcast]] = defaultdict(list)

    # -- link/network control -------------------------------------------------

    def set_link_down(self, robot_a: str, robot_b: str, down: bool = True):
        """Simulate a broken connection between two specific robots (e.g. a
        radio dead zone). Every other robot pair is unaffected -- there is
        no shared central hub whose failure could take down the fleet."""
        key = frozenset((robot_a, robot_b))
        if down:
            self._down_links.add(key)
        else:
            self._down_links.discard(key)

    def _link_is_down(self, a, b):
        return frozenset((a, b)) in self._down_links

    # -- sending / receiving ----------------------------------------------------

    def broadcast(self, sender_id: str, recipients: List[str], msg: StateBroadcast,
                  now: float):
        """Send msg from sender_id to each recipient, subject to per-link
        latency and loss. This models a broadcast/gossip radio, not a
        server relay: each recipient gets (or doesn't get) the message
        independently."""
        for r in recipients:
            if r == sender_id:
                continue
            stats = self.link_stats[(sender_id, r)]
            stats.sent += 1
            if self._link_is_down(sender_id, r):
                stats.dropped += 1
                continue
            if self.rng.random() < self.loss_prob:
                stats.dropped += 1
                continue
            delay = self.rng.uniform(*self.latency_range)
            arrival = now + delay
            stats.delivered += 1
            stats.latencies.append(delay)
            heapq.heappush(self._in_flight, (arrival, r, msg))

    def tick(self, now: float):
        """Deliver any messages whose arrival time has passed. Call once per
        simulation step before robots read their inboxes."""
        while self._in_flight and self._in_flight[0][0] <= now:
            _, recipient, msg = heapq.heappop(self._in_flight)
            self._inboxes[recipient].append(msg)

    def inbox(self, robot_id: str) -> List[StateBroadcast]:
        """Pop and return all messages delivered to robot_id so far."""
        msgs = self._inboxes[robot_id]
        self._inboxes[robot_id] = []
        return msgs

    def average_latency(self) -> float:
        all_lat = [l for s in self.link_stats.values() for l in s.latencies]
        return sum(all_lat) / len(all_lat) if all_lat else 0.0

    def loss_rate(self) -> float:
        sent = sum(s.sent for s in self.link_stats.values())
        dropped = sum(s.dropped for s in self.link_stats.values())
        return dropped / sent if sent else 0.0
