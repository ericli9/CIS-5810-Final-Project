"""Debounced zone occupancy per tracked object.

A raw "is the box inside the polygon" test flickers: boxes jitter, a hand
occludes an item for three frames, the tracker drops and re-acquires. So an
object has to clear ``confirm_frames`` before it counts as inside a zone and
``exit_frames`` before it counts as out, and a track that vanishes is held for
``lost_frames`` before its zones are released.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .detect import Detection
from .zones import Zone, ZoneSet


@dataclass
class TrackState:
    track_id: int
    cls_votes: Counter = field(default_factory=Counter)
    hits: dict[str, int] = field(default_factory=dict)
    misses: dict[str, int] = field(default_factory=dict)
    inside: set[str] = field(default_factory=set)
    visited_roles: set[str] = field(default_factory=set)
    first_frame: int = 0
    last_frame: int = 0
    absent: int = 0
    last_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_conf: float = 0.0

    @property
    def cls_name(self) -> str:
        """Majority class over the track's life -- steadier than any one frame."""
        return self.cls_votes.most_common(1)[0][0] if self.cls_votes else "unknown"


@dataclass
class ZoneEvent:
    kind: str  # "enter" | "exit"
    camera: str
    zone: Zone
    track: TrackState
    frame_idx: int

    @property
    def role(self) -> str:
        return self.zone.role


class ZoneOccupancy:
    """Turns a stream of tracked detections into confirmed zone enter/exit events."""

    def __init__(
        self,
        camera: str,
        zoneset: ZoneSet,
        coverage_threshold: float = 0.30,
        confirm_frames: int = 4,
        exit_frames: int = 6,
        lost_frames: int = 45,
    ):
        self.camera = camera
        self.zoneset = zoneset
        self.coverage_threshold = coverage_threshold
        self.confirm_frames = confirm_frames
        self.exit_frames = exit_frames
        self.lost_frames = lost_frames
        self.tracks: dict[int, TrackState] = {}

    def update(self, detections: list[Detection], frame_idx: int) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        seen: set[int] = set()

        for det in detections:
            if det.track_id is None:
                continue  # untracked boxes are drawn but never move cart state
            seen.add(det.track_id)
            st = self.tracks.get(det.track_id)
            if st is None:
                st = TrackState(track_id=det.track_id, first_frame=frame_idx)
                self.tracks[det.track_id] = st
            st.cls_votes[det.cls_name] += 1
            st.last_frame = frame_idx
            st.last_box = det.box
            st.last_conf = det.conf
            st.absent = 0

            for zone in self.zoneset:
                cov = zone.coverage(det.box)
                if cov >= self.coverage_threshold:
                    st.hits[zone.name] = st.hits.get(zone.name, 0) + 1
                    st.misses[zone.name] = 0
                    if zone.name not in st.inside and st.hits[zone.name] >= self.confirm_frames:
                        st.inside.add(zone.name)
                        st.visited_roles.add(zone.role)
                        events.append(ZoneEvent("enter", self.camera, zone, st, frame_idx))
                else:
                    st.misses[zone.name] = st.misses.get(zone.name, 0) + 1
                    st.hits[zone.name] = 0
                    if zone.name in st.inside and st.misses[zone.name] >= self.exit_frames:
                        st.inside.discard(zone.name)
                        events.append(ZoneEvent("exit", self.camera, zone, st, frame_idx))

        for track_id in list(self.tracks):
            if track_id in seen:
                continue
            st = self.tracks[track_id]
            st.absent += 1
            if st.absent < self.lost_frames:
                continue
            for zone_name in sorted(st.inside):
                zone = self.zoneset.get(zone_name)
                if zone is not None:
                    events.append(ZoneEvent("exit", self.camera, zone, st, frame_idx))
            st.inside.clear()
            del self.tracks[track_id]

        return events

    def active(self) -> list[TrackState]:
        return [st for st in self.tracks.values() if st.absent == 0]
