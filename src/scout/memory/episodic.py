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
    status: str = "ok"  # "ok", "retracted" (a one-off sighting that a second look did not confirm), "gone" (was there, now is not)

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
            if tr.label != label or tr.status != "ok":
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
            if tr.status != "ok":
                continue
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
        """Plain-text memory for the language model: one line per remembered object, plus misses and verdicts."""
        lines = ["object_id | object | first_seen_s | last_seen_s | times_seen | best_confidence | x_m | y_m | sighting_ids (first, best, last)"]
        for tr in self.tracks:
            ss = [self.sightings[i] for i in tr.sightings]
            best = max(ss, key=lambda s: s.score)
            ids = sorted({ss[0].id, best.id, ss[-1].id})
            lines.append(f"{tr.id} | {tr.label} | {ss[0].t:.1f} | {ss[-1].t:.1f} | {len(ss)} | {best.score:.2f} | "
                         f"{best.xyz[0]:.1f} | {best.xyz[1]:.1f} | {ids}")
            for t, fr in tr.misses:
                lines.append(f"{tr.id} | {tr.label} NOT seen at its last known place at t={t:.1f}s")
            if tr.status == "retracted":
                lines.append(f"{tr.id} | {tr.label}: a second look found nothing there, probably a misrecognition")
            elif tr.status == "gone":
                lines.append(f"{tr.id} | {tr.label}: was seen here repeatedly, but a deliberate check found it gone")
        return "\n".join(lines)
