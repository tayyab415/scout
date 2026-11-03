"""The Meta Motivo M-1 body inside a house: place it, walk it with named skills, film it.

Needs a GPU machine with humenv, metamotivo, torch and MuJoCo EGL rendering (see scripts/modal_body.py).
Skill vectors (z) come from Motivo's reward inference and are stored in a torch file name -> tensor.
"""
from __future__ import annotations

import math

import numpy as np

from ..world.house import CEILING_GROUP, build_body_world


class MotivoBody:
    def __init__(self, root: str, house: str, skills_path: str, device: str = "cuda", eye_pitch_deg: float = 0.0):
        import mujoco
        import torch
        from humenv import make_humenv
        from metamotivo.fb_cpr.huggingface import FBcprModel

        self.torch, self.mj, self.device = torch, mujoco, device
        self.model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(device)
        self.Z = {k: v.to(device) for k, v in torch.load(skills_path).items()}
        self.axis = self._calibrate(make_humenv)
        self.env, _ = make_humenv(num_envs=1, state_init="Default")
        self.u = self.env.unwrapped
        self.M = build_body_world(root, house, eye_pitch_deg=eye_pitch_deg)
        self.u.model, self.u.data = self.M, mujoco.MjData(self.M)
        self.D = self.u.data
        self.pelvis = mujoco.mj_name2id(self.M, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
        self.head = mujoco.mj_name2id(self.M, mujoco.mjtObj.mjOBJ_BODY, "Head")
        self._rd = {}

    # --- state --------------------------------------------------------------------------------------
    def _obs(self):
        return self.torch.tensor(self.u.get_obs()["proprio"].reshape(1, -1), dtype=self.torch.float32, device=self.device)

    def _calibrate(self, make_humenv):
        """Which pelvis axis points the way the body walks (done once in an empty world)."""
        env, _ = make_humenv(num_envs=1, state_init="Default")
        u = env.unwrapped
        env.reset(seed=0)
        pid = self.mj.mj_name2id(u.model, self.mj.mjtObj.mjOBJ_BODY, "Pelvis")
        p0, R0 = u.data.xpos[pid].copy(), u.data.xmat[pid].reshape(3, 3).copy()
        o = self.torch.tensor(u.get_obs()["proprio"].reshape(1, -1), dtype=self.torch.float32, device=self.device)
        for _ in range(60):
            a = self.model.act(o, self.Z["move-ego-0-1"], mean=True).float().cpu().numpy().ravel()
            env.step(a)
            o = self.torch.tensor(u.get_obs()["proprio"].reshape(1, -1), dtype=self.torch.float32, device=self.device)
        d = u.data.xpos[pid][:2] - p0[:2]
        d /= np.linalg.norm(d) + 1e-9
        return max(((s, c) for c in range(3) for s in (1, -1)), key=lambda k: k[0] * R0[:2, k[1]] @ d)

    @property
    def xy(self) -> np.ndarray:
        return self.D.xpos[self.pelvis][:2].copy()

    @property
    def yaw(self) -> float:
        R = self.D.xmat[self.pelvis].reshape(3, 3)
        v = self.axis[0] * R[:2, self.axis[1]]
        return math.atan2(v[1], v[0])

    @property
    def head_z(self) -> float:
        return float(self.D.xpos[self.head][2])

    def place(self, xy, yaw: float) -> None:
        """Stand the body at xy facing yaw (rotates the T-pose about z, then corrects for the walking axis)."""
        self.env.reset(seed=0)
        q = self.D.qpos.copy()
        q[0:2] = xy
        self.u.set_physics(q, np.zeros(self.M.nv))
        err = yaw - self.yaw
        qy = np.array([math.cos(err / 2), 0, 0, math.sin(err / 2)])
        qq = np.zeros(4)
        self.mj.mju_mulQuat(qq, qy, q[3:7])
        q[3:7] = qq
        self.u.set_physics(q, np.zeros(self.M.nv))

    def step(self, skill: str) -> None:
        a = self.model.act(self._obs(), self.Z[skill], mean=True).float().cpu().numpy().ravel()
        self.env.step(a)

    def step_z(self, z) -> None:
        """One 30 Hz control step with an arbitrary skill vector (for blending and mixing)."""
        a = self.model.act(self._obs(), z.reshape(1, -1), mean=True).float().cpu().numpy().ravel()
        self.env.step(a)

    def head_rgbd(self, size: int = 512):
        """Head camera colour, depth (metres along the view axis), camera position and rotation matrix."""
        rd = self._renderer(size, size)
        rd.update_scene(self.D, camera="eye")
        rgb = rd.render().copy()
        rd.enable_depth_rendering()
        rd.update_scene(self.D, camera="eye")
        depth = rd.render().copy()
        rd.disable_depth_rendering()
        cid = self.mj.mj_name2id(self.M, self.mj.mjtObj.mjOBJ_CAMERA, "eye")
        return rgb, depth, self.D.cam_xpos[cid].copy(), self.D.cam_xmat[cid].reshape(3, 3).copy()

    # --- cameras ------------------------------------------------------------------------------------
    def _renderer(self, h: int, w: int):
        if (h, w) not in self._rd:
            self._rd[(h, w)] = self.mj.Renderer(self.M, h, w)
        return self._rd[(h, w)]

    def head_view(self, size: int = 360) -> np.ndarray:
        rd = self._renderer(size, size)
        rd.update_scene(self.D, camera="eye")  # default groups 0-2: visuals and the ceiling
        return rd.render().copy()

    def follow_view(self, azimuth_deg: float, h: int = 480, w: int = 640, distance: float = 4.6,
                    elevation: float = -62) -> np.ndarray:
        rd = self._renderer(h, w)
        cam = self.mj.MjvCamera()
        cam.lookat[:] = self.D.xpos[self.pelvis] + np.array([0, 0, 0.1])
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth_deg, elevation
        opt = self.mj.MjvOption()
        opt.geomgroup[CEILING_GROUP] = 0
        rd.update_scene(self.D, camera=cam, scene_option=opt)
        return rd.render().copy()
