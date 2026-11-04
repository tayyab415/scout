"""Turn what the body knows into a prompt, and the model's reply into a choice.

The prompt holds exact coordinates, but only for things the body has itself seen.
"""
from __future__ import annotations

import json
import math
import re

from ..agent import Body
from ..memory.explored import Frontier, Memory
from ..world.layout import GOAL_TYPES

SYSTEM = (
    "You are the navigation mind of a person walking through a house they have never been in. "
    "You only know what they have seen so far. Coordinates are in metres: +x is east, +y is north. "
    "Use common sense about how houses are laid out (bedrooms off corridors, bathrooms next to bedrooms, "
    "kitchens next to dining and living areas) to decide where to look next. Answer with JSON only."
)

ROOM_NAMES = {"LivingRoom": "living room", "Kitchen": "kitchen", "Bedroom": "bedroom", "Bathroom": "bathroom"}


def compass(dx: float, dy: float) -> str:
    a = math.degrees(math.atan2(dy, dx)) % 360
    return ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"][int((a + 22.5) // 45) % 8]


def describe(memory: Memory, body: Body, frontiers: list[tuple[Frontier, float]], goal: str | None,
             budget_left: float) -> str:
    lay = memory.layout
    known_rooms = memory.rooms_seen()
    known_ids = {rid for rid, _, _ in known_rooms}
    lines = []
    if goal:
        lines.append(f"TASK: find the {GOAL_TYPES.get(goal, goal)}. You have not seen one yet.")
    else:
        lines.append("TASK: explore this house so you can later find anything in it. "
                     "See as many different rooms and pieces of furniture as you can with little walking.")
    lines.append(f"You are at ({body.xy[0]:.1f}, {body.xy[1]:.1f}). You have walked {body.walked:.0f} m; "
                 f"{budget_left:.0f} m of walking left.")
    lines.append("Rooms you have looked into:" if known_rooms else "Rooms you have looked into: none yet.")
    for rid, rtype, c in known_rooms:
        lines.append(f"- {ROOM_NAMES.get(rtype, rtype)} around ({c[0]:.1f}, {c[1]:.1f})")
    seen = sorted(memory.sightings.values(), key=lambda s: s.type)
    if seen:
        lines.append("Things you have seen:")
        for s in seen:
            room = next((ROOM_NAMES.get(t, t) for rid, t, _ in known_rooms if rid == s.room), None)
            where = f" in the {room}" if room else ""
            lines.append(f"- {GOAL_TYPES.get(s.type, s.type)} at ({s.xy[0]:.1f}, {s.xy[1]:.1f}){where}")
    lines.append("Unexplored openings you can walk to:")
    for k, (f, d) in enumerate(frontiers):
        r = lay.room_at(f.xy)
        area = f"edge of the {ROOM_NAMES.get(r.type, r.type)}" if r is not None and r.id in known_ids else "leads into unseen space"
        dx, dy = f.xy - body.xy
        lines.append(f"[{k}] at ({f.xy[0]:.1f}, {f.xy[1]:.1f}), {d:.1f} m walk to the {compass(dx, dy)}, "
                     f"{f.size * lay.grid.res:.1f} m of unseen edge, {area}")
    lines.append('First think about where this is likely to be, then reply as '
                 '{"why": "<one or two sentences>", "choice": <opening number>}')
    return "\n".join(lines)


def parse(text: str, n: int) -> tuple[int | None, str]:
    """Pull the chosen opening out of a reply. Returns (index or None, reason)."""
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        c = obj.get("choice")
        if isinstance(c, str) and c.strip().lstrip("[").rstrip("]").isdigit():
            c = int(c.strip().lstrip("[").rstrip("]"))
        if isinstance(c, int) and 0 <= c < n:
            return c, str(obj.get("why", ""))[:300]
    m = re.search(r'"?choice"?\s*[:=]\s*\[?(\d+)', text)
    if m and int(m.group(1)) < n:
        return int(m.group(1)), ""
    return None, ""
