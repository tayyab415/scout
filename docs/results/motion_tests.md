# Why the body moves badly: three tests on frozen Motivo (empty world)

Run 2026-11-02 on Modal CPUs. Results: `runs/motion/results.json`. Clips: `runs/motion/*.mp4`.

1. **Turning.** Reward-inferred yaw skills on the pelvis's vertical (y) axis, 6 runs x 6 s each, no falls in any run.
   Asking for 0.5 or 1 rad/s gives a body that stands (1-5 deg in 6 s). Asking for 2 rad/s gives a slow real turn:
   +32 deg left, -51 deg right in 6 s (about 5-8 deg/s). Sidestepping turns 119 deg but travels 8.5 m.
2. **Instant switching vs 0.5 s blending** of the skill vector, 20 runs x 15 s, a new random skill every 0.5 s.
   No falls either way. Head rotation speed drops from 110 to 69 deg/s with blending; mean head tilt is unchanged
   (19 vs 20 deg).
3. **Mixing forward with a sidestep.** Walking direction moves smoothly and in order with the mix:
   left mix 0 / 0.25 / 0.5 / 0.75 / 1 -> -18 / 12 / 71 / 119 / 121 deg; right mix -> -18 / -33 / -85 / -122 / -122 deg.
   No falls. (Starts and policy are deterministic, so each value is effectively one run.)

What follows: Motivo's skill space is continuous and steerable, so a small network that outputs the skill vector
every step (an adapter) can steer it smoothly. Turning on the spot exists but is slow; turning while walking is
available through mixing. Blending alone makes the head steadier but does not fix the lean.
