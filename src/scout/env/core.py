"""The Scout environment: a Meta Motivo humanoid in an unseen ProcTHOR house, a head RGB-D camera, SAM 3.1 eyes, and
an evaluator that keeps the truth. Needs a GPU machine with humenv, metamotivo, mujoco (EGL) and sam3.

Contract (what an agent gets, `ScoutEnv.observe`):
  t           simulation time in seconds (one clock for everything, 30 physics-control steps per second)
  rgb, depth  head camera, 512 x 512, 75 deg vertical FOV, tilted 20 deg down; depth is ideal simulated depth (m)
  cam_pos, cam_mat   exact head-camera pose (disclosed: perfect odometry and head pose from the simulator)
  xy, yaw, head_z    body position, heading and head height (same source)
  dets        SAM 3.1 detections for the fixed object vocabulary (when asked for)
Actions: `step(z)` with a Motivo skill vector, or `skill(name)`. The agent never sees the evaluator's truth.

Evaluator side (`ScoutEnv.truth_*`, `move_object`): stable object ids, current positions after moves, and per-look
visibility (which object instances cover >= 0.5% of the head image, from the simulator's segmentation).
"""
from __future__ import annotations

import copy
import math

import numpy as np

PROMPTS = ["bed", "sofa", "television", "dining table", "fridge", "kitchen counter", "chair", "armchair",
           "floor lamp", "painting", "laptop", "toilet", "sink", "dresser", "bathtub", "house plant"]
# THOR body-name prefixes for each category (evaluator only)
CATEGORY_PREFIX = {
    "bed": ["bed"], "toilet": ["crapper", "toilet"], "sofa": ["sofa", "couch"], "television": ["television", "tv"],
    "dining table": ["diningtable"], "fridge": ["refrigerator", "fridge"], "bathtub": ["bathtub", "tub"], "sink": ["sink"],
    "dresser": ["chestofdrawers", "dresser"], "armchair": ["armchair"], "kitchen counter": ["countertop"],
    "laptop": ["laptop"], "house plant": ["houseplant"], "floor lamp": ["floorlamp"], "painting": ["painting"],
    "chair": ["chair"],
}
MOVABLE = ["television", "armchair", "floor lamp", "laptop", "sofa", "dresser", "chair"]
FOVY, SIZE, EYE_PITCH = 75, 512, 20
VISIBLE_FRAC = 0.005


class ScoutEnv:
    def __init__(self, house: str, seed: int = 0, root: str = "/data/house", skills: str = "/data/scout/z_motion.pt",
                 eyes=None):
        import mujoco
        from ..body.motivo import MotivoBody
        from ..world.layout import Layout
        self.mj, self.house, self.seed = mujoco, house, seed
        self.body = MotivoBody(root, house, skills, device="cuda", eye_pitch_deg=EYE_PITCH)
        self.body.M.vis.global_.offwidth, self.body.M.vis.global_.offheight = 1280, 960
        self.M, self.D = self.body.M, self.body.D
        self.Z = self.body.Z
        self.eyes = eyes
        self.layout = Layout.load(f"{root}/layouts/{house}.npz")  # evaluator side
        self.k = 0
        self._instances()
        self.moves = []

    # --- evaluator: stable object instances --------------------------------------------------------------------
    def _instances(self):
        """Group THOR bodies into object instances with a category and a stable id (evaluator only)."""
        mj, M = self.mj, self.M
        names = [mj.mj_id2name(M, mj.mjtObj.mjOBJ_BODY, b) or "" for b in range(M.nbody)]
        pre = {p: c for c, ps in CATEGORY_PREFIX.items() for p in ps}
        groups = {}
        for b, n in enumerate(names):
            if not n.startswith("h_"):
                continue
            parts = n[2:].split("_")
            cat = pre.get(parts[0].lower())
            if not cat or len(parts) < 3:
                continue
            key = "_".join(parts[:3])  # prefix_hash_index: one placed object
            groups.setdefault(key, (cat, []))[1].append(b)
        self.instances = {}
        for i, (key, (cat, bodies)) in enumerate(sorted(groups.items())):
            self.instances[i] = dict(id=i, key=key, category=cat, bodies=bodies)
        self.body_instance = np.full(M.nbody, -1)
        for i, inst in self.instances.items():
            self.body_instance[inst["bodies"]] = i
        self.names = names

    def truth_objects(self) -> list[dict]:
        """Current position of every object instance (after any evaluator moves)."""
        out = []
        for i, inst in self.instances.items():
            p = np.mean([self.D.xpos[b][:2] for b in inst["bodies"]], 0)
            out.append(dict(id=i, category=inst["category"], xy=p.round(3).tolist()))
        return out

    def visible_instances(self) -> dict:
        """Which object instances cover >= VISIBLE_FRAC of the head image right now: {instance_id: fraction}."""
        if not hasattr(self, "_seg_rd"):  # its own renderer: sharing one with depth rendering garbles the ids
            self._seg_rd = self.mj.Renderer(self.M, SIZE, SIZE)
            self._seg_rd.enable_segmentation_rendering()
        rd = self._seg_rd
        rd.update_scene(self.D, camera="eye")
        seg = rd.render()
        gid = seg[..., 0]
        isg = (seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
        inst = np.full(gid.shape, -1)
        inst[isg] = self.body_instance[self.M.geom_bodyid[gid[isg]]]
        ids, counts = np.unique(inst[inst >= 0], return_counts=True)
        frac = counts / inst.size
        return {int(i): round(float(f), 4) for i, f in zip(ids, frac) if f >= VISIBLE_FRAC}

    def move_object(self, instance_id: int, to_xy) -> dict:
        """Evaluator intervention: slide one object instance to a new floor position (disclosed in results)."""
        inst = self.instances[instance_id]
        src = np.mean([self.D.xpos[b][:2] for b in inst["bodies"]], 0)
        delta = np.asarray(to_xy, float) - src
        top = [b for b in inst["bodies"] if self.M.body_parentid[b] not in inst["bodies"]]
        for b in top:
            self.M.body_pos[b][:2] += delta
        self.mj.mj_forward(self.M, self.D)
        rec = dict(t=round(self.t, 2), instance=instance_id, category=inst["category"], from_xy=src.round(3).tolist(),
                   to_xy=np.asarray(to_xy, float).round(3).tolist())
        self.moves.append(rec)
        return rec

    # --- the agent's interface -----------------------------------------------------------------------------------
    @property
    def t(self) -> float:
        return self.k / 30.0

    def reset(self, start_xy=None, start_yaw=None):
        rng = np.random.default_rng(self.seed)
        cells = np.argwhere(self.layout.grid.free)
        xy = self.layout.grid.pos(cells[rng.integers(len(cells))]) if start_xy is None else np.asarray(start_xy, float)
        yaw = float(rng.uniform(-math.pi, math.pi)) if start_yaw is None else float(start_yaw)
        self.body.place(xy, yaw)
        for _ in range(20):
            self.body.step("move-ego-0-0")
        self.k = 0
        self.start = dict(xy=xy.round(3).tolist(), yaw=round(yaw, 3))
        return self.observe()

    def step(self, z) -> bool:
        """One control step (1/30 s). Returns False if the physics became invalid."""
        try:
            self.body.step_z(z)
        except ValueError:
            self.k += 1
            return False
        self.k += 1
        return True

    def observe(self, detect: bool = False) -> dict:
        rgb, depth, cp, cm = self.body.head_rgbd(SIZE)
        obs = dict(t=self.t, rgb=rgb, depth=depth, cam_pos=cp, cam_mat=cm, xy=self.body.xy, yaw=self.body.yaw,
                   head_z=self.body.head_z, dets=None)
        if detect and self.eyes is not None:
            obs["dets"] = self.eyes.detect(rgb, PROMPTS)
        return obs

    def observer_view(self, h: int = 540, w: int = 720) -> np.ndarray:
        """A stable viewer camera (fixed heading, high angle) with walls and ceilings cut away. Viewers only."""
        return self.body.follow_view(135.0, h, w, distance=6.0, elevation=-58)

    # --- branching ----------------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        D = self.D
        return dict(qpos=D.qpos.copy(), qvel=D.qvel.copy(), act=D.act.copy(), ctrl=D.ctrl.copy(), time=D.time,
                    k=self.k, body_pos=self.M.body_pos.copy(), moves=copy.deepcopy(self.moves))

    def restore(self, s: dict) -> None:
        D = self.D
        self.M.body_pos[:] = s["body_pos"]
        D.qpos[:], D.qvel[:], D.ctrl[:] = s["qpos"], s["qvel"], s["ctrl"]
        if D.act.size:
            D.act[:] = s["act"]
        D.time, self.k, self.moves = s["time"], s["k"], copy.deepcopy(s["moves"])
        self.mj.mj_forward(self.M, D)
