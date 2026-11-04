# A V-JEPA 2 world model for the Motivo body's own eyes

Run 2026-11-02 on Modal. This follows the V-JEPA 2-AC recipe (arXiv 2506.09985): frozen V-JEPA 2 encoder, new
action-conditioned predictor, planning by distance to a goal image in feature space. Everything here is measured on
houses the predictor never saw.

## Setup
- Data: 104,730 head-camera frames, 2,592 episodes in 60 ProcTHOR houses (48 train / 12 test), ~11 s random
  walks with 6 Motivo skills (stand, forward, side-left, side-right, back, walk-turn-right), 3.75 frames/s.
  88 episodes ended in a fall. The body touches a wall or furniture in 50% (train) / 56% (test) of frames.
- Encoder: V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), frozen; each frame as a 2-frame clip, 16x16
  patches average-pooled to 4x4.
- Predictor: 11.4M-parameter block-causal transformer, 8-frame context, teacher-forcing + one-step rollout L1
  losses, 12k steps (29 min on one L40S). Control twin: identical model with the action input blanked.

## Results on 12 unseen houses (7,775 frames)

| Test | Result |
|---|---|
| Next-view prediction error (L1, horizons 1/2/4) | action-aware 0.296 / 0.334 / 0.372; action-blind 0.321 / 0.367 / 0.395; copy last frame 0.338 / 0.394 / 0.436 |
| Which of 6 actions was taken (chance 17%) | 53% one step ahead, 46% two steps ahead. "Stand" over-chosen (2,111 picks vs ~1,300 true); forward most often confused with stand |
| Surprise vs events, AUC (model surprise / plain visual change) | bump onset 0.58 / 0.64; room change 0.49 / 0.54; head drop >5 cm 0.70 / 0.74 |
| Walk to a goal photo 2-5 m away (36 goals in 3 houses, 40 steps, success = within 0.75 m and 45 deg) | world-model planning (2-step search) 3/36, one-step greedy 1/36, random 0/36; mean final distance 3.3 / 3.5 / 3.6 m |

## What this shows
1. The body learns what its own actions do to its view, and that carries over to unseen houses. Meta's frozen
   V-JEPA 2-AC, run on a simulated arm in an earlier probe here, collapsed every action to "no movement" (1/27).
2. That knowledge does not yet turn into reaching goal photos. In a recorded run the body ends 0.33 m from the
   goal spot but facing the wrong way: Motivo cannot turn on the spot (reward-inferred slow turns collapse to
   standing), so it cannot match the goal view's heading.
3. Prediction-error "surprise" does not flag bumps, room changes or stumbles better than raw visual change. We do
   not claim surprise-based event detection.

## Limits
Random-walk data dominated by wall contact; 4x4 pooled features may lose small obstacles; 36 planning goals is a
small sample; goal photos come from the same house's memory bank; planning searches all 2-step sequences only.

Recordings: `runs/vj/world_tour.mp4`, `runs/vj/plan/*.mp4` (follow camera, head camera, goal photo, and the
imagined next view for each action, shown as the nearest real view in memory).
