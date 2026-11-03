# Third-party models and data

The MIT license covers this repository's own code. The models and data it uses keep their own licenses. Nothing
below is redistributed except the noted derived files.

| Component | Source | License | Notes |
| --- | --- | --- | --- |
| Meta Motivo M-1 (FB-CPR) and HumEnv | github.com/facebookresearch/metamotivo, HF `facebook/metamotivo-M-1` | CC BY-NC 4.0 | Downloaded at run time. `assets/skills/z_motion.pt` holds skill vectors derived from it here, by reward inference, under the same non-commercial terms. |
| SAM 3.1 | github.com/facebookresearch/sam3, HF `facebook/sam3.1` (gated) | SAM License | Downloaded at run time with your own access. |
| ProcTHOR-10K houses and THOR assets, via MolmoSpaces | HF `allenai/molmospaces` | CC BY 4.0 | Downloaded by `scripts/modal_fetch_houses.py`. `examples/sample_episode/` contains rendered frames and a layout derived from house `procthor-10k-val_val_100`. |
| Muse Spark 1.3, Llama 4 Maverick | Meta Model API | provider terms | Called as plug-in brains. Muse Spark 1.3 runs on Meta Model API. |
