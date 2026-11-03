"""What the body remembers seeing: timestamped sightings grouped into objects, with the frame each came from.

Pure numpy. Sightings come from SAM 3.1 masks placed in 3D with head depth. Nothing here reads simulator truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SAME_OBJECT_M = 1.0   # same label within this distance -> same object


@dataclass
class Sighting:
    id: int
    label: str
    score: float
    xyz: np.ndarray
    t: float
    frame: int          # index of the saved head-camera frame
    body_xy: np.ndarray


@dataclass
class Track:
    id: int
    label: str
    sightings: list[int] = field(default_factory=list)
    misses: list[tuple[float, int]] = field(default_factory=list)  # (t, frame) looked right at it and did not see it

    def last_xy(self, mem: "Episodic") -> np.ndarray:
        return mem.sightings[self.sightings[-1]].xyz[:2]


@dataclass
class Episodic:
    sightings: list[Sighting] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)

    def add(self, label: str, score: float, xyz, t: float, frame: int, body_xy) -> tuple[Sighting, Track, bool]:
        """Store one sighting. Returns (sighting, track, is_new_object)."""
        s = Sighting(len(self.sightings), label, float(score), np.asarray(xyz, float), float(t), int(frame),
                     np.asarray(body_xy, float))
        self.sightings.append(s)
        best, best_d = None, SAME_OBJECT_M
        for tr in self.tracks:
            if tr.label != label:
                continue
            d = float(np.linalg.norm(tr.last_xy(self) - s.xyz[:2]))
            if d < best_d:
                best, best_d = tr, d
        if best is None:
            best = Track(len(self.tracks), label)
            self.tracks.append(best)
            best.sightings.append(s.id)
            return s, best, True
        best.sightings.append(s.id)
        return s, best, False

    def expected_in_view(self, cam_pos, cam_mat, fovy_deg: float, depth: np.ndarray, max_range: float = 4.0):
        """Tracks whose last known position is inside this view and not hidden behind something nearer.

        Returns [(track, pixel (u, v))]."""
        h, w = depth.shape
        f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
        R = np.asarray(cam_mat).reshape(3, 3)
        out = []
        for tr in self.tracks:
            p = self.sightings[tr.sightings[-1]].xyz
            pc = R.T @ (p - np.asarray(cam_pos))
            if pc[2] >= -0.3:
                continue
            d = -pc[2]
            if d > max_range:
                continue
            u = int(pc[0] / d * f + w / 2)
            v = int(-pc[1] / d * f + h / 2)
            if not (8 <= u < w - 8 and 8 <= v < h - 8):
                continue
            if depth[v, u] < d - 0.5:  # something nearer is in the way
                continue
            out.append((tr, (u, v)))
        return out

    def table(self) -> str:
        """Plain-text memory for the language model: one line per sighting."""
        lines = ["sighting_id | time_s | object | confidence | x_m | y_m | object_id"]
        for tr in self.tracks:
            for sid in tr.sightings:
                s = self.sightings[sid]
                lines.append(f"{s.id} | {s.t:.1f} | {s.label} | {s.score:.2f} | {s.xyz[0]:.1f} | {s.xyz[1]:.1f} | {tr.id}")
            for t, fr in tr.misses:
                lines.append(f"- | {t:.1f} | {tr.label} NOT seen at its last known place | - | - | - | {tr.id}")
        return "\n".join(lines)
