# Coverage contract

The v11 course maps the original LSS paper, the BEVDepth paper, both official implementations and one fixed nuScenes sample to seven advanced tensor-first scenes. The opening consolidates motivation, depth distribution and context into one real-pixel inspection; the second scene consolidates Lift and the entire calibrated coordinate chain.

| # | Narrative scene | Paper / implementation coverage | Evidence |
|---:|---|---|---|
| 1 | One ray, two predictions | task topology; real `(u,v)` → network anchor; 41-bin checkpoint depth distribution; 64D context payload; vehicle/sky comparison | LSS PAPER + CHECKPOINT |
| 2 | Give every hypothesis a place | broadcast Lift; inverse post-transform; `K⁻¹`; metric depth; real `R,t`; camera point → ego point | LSS CODE + REAL SAMPLE |
| 3 | Resolve collisions | bounds, floor/rank/sort/QuickCumsum; interactive sum/mean/max/bilinear semantics; efficient voxel pooling contrast | LSS + BEVDEPTH CODE |
| 4 | Reason on the ground plane | `[B,C,Z,X,Y]`, Z collapse, `[B,64,200,200]`, BevEncode and vehicle logits | LSS CODE + CHECKPOINT |
| 5 | Where does depth learn? | indirect LSS task gradient versus LiDAR-projected sparse depth target, min pooling, one-hot bins, valid-pixel BCE and `Ldet+3Ldepth` | LSS + BEVDEPTH CODE |
| 6 | Turn logits into a decision | sigmoid, threshold, task-head and post-processing boundaries | LSS CODE + BEVDEPTH PAPER |
| 7 | Audit the map | orientation contract, GT/LiDAR/checkpoint separation, TP/FP/FN and linked evidence | CHECKPOINT + REAL SAMPLE |

## Practice-pause mapping

| Pause | Scenes | Browser contract |
|---|---|---|
| Calibrated geometry | 1–2 | real image, selected feature anchor, checkpoint depth/context tensors, image plane, K⁻¹, camera XYZ, real outward frustums and exact ego transform |
| Pooling comparison | 3 | live sum/mean/max/bilinear collision rule and a candidate crossing a cell boundary |
| Linked BEV evidence | 7 | probability, threshold, GT, errors, LiDAR occupancy, contributors and raw-grid inspection |

## Deliberate boundaries

- BEVDepth coverage is limited to the paper's direct conceptual extensions of the LSS view transformer; no BEVDet/BEVStereo/temporal expansion, browser inference, training service, upload or backend.
- LiDAR is a single-sweep reference visualization, never an LSS input.
- LiDAR is described as BEVDepth training supervision only where explicitly labeled; BEVDepth inference remains camera-only.
- The checkpoint is vehicle semantic segmentation; box decoding, NMS and tracking are not invented.
- Per-frame TP/FP/FN/IoU diagnoses this sample only; dataset metrics are labeled PAPER.
- The public seven-scene lesson ends at view transformation, task supervision, inference and truth auditing. LSS planning/Shoot is outside this edition's teaching path rather than compressed into an unrelated final control.
