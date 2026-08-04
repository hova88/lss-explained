# Evidence manifest

All browser-visible model results are reproducible derivatives of pinned inputs. The official NVIDIA source and checkpoint are used only in ignored local cache storage.

## Pinned sources

| Input | Revision / SHA-256 |
|---|---|
| Official LSS repository | `2903467c91ee9c12f0917a12c22ab1f04e607ae0` |
| OpenMMLab MMDetection3D demo | `fe25f7a51d36e3702f961e198894580d83c4387b` |
| `model525000.pt` | `4543030a339face9facb5651eb8f29add3407f8c7108f9eb21b0f8bceec921a0` |
| Demo pickle | `5ced892a1c939de17b9df1c2719684f84e0103096c2f689761631d1c34fb0ac8` |
| CAM_FRONT_LEFT | `2c8c49149411fa88cd36f9076d19e064be963deaf4e7c0dde4a7c691e83e1e5a` |
| CAM_FRONT | `b7b7d466207462cf46742297a36afdd65315c05ae33126d5d36412aae70a0b62` |
| CAM_FRONT_RIGHT | `005ce4aff67c8df56233457218363687132fe7f40de758cf030aace578f55497` |
| CAM_BACK_LEFT | `3ffae8abed7fc7e703cad8f10f92535c5f55314433660f11b1ed6ef02ebd41bf` |
| CAM_BACK | `51fe14c13dc4f2ab0f7f59b6634b98925b767939b3417a9db617bd3c178c9bb4` |
| CAM_BACK_RIGHT | `ab220a717b467d4fbed66b1a8a923dc522cc3aaaafa041613bcbfca3dd34ad9b` |

nuScenes sample token: `ca9a282c9e77460f8360f564131a8af5`.

## Export assertions

- Strict checkpoint state-dict load: no missing or unexpected keys.
- Input `[1,6,3,128,352]`.
- Lifted features `[1,6,41,8,22,64]`.
- Pooled BEV `[1,64,200,200]`.
- Vehicle logits `[1,1,200,200]`.
- All exported outputs finite.
- Maximum depth-probability normalization error: `4.76837158203125e-7`.
- Ten reproducible vehicle outputs: all cameras, six leave-one-camera-out cases, and CAM_FRONT yaw `−3° / 0° / +3°`.

The model contract hash is `cb36c8e3c2890c659a682691b06e6be9f4fc4bb768cd4b9312b1a4f2060773e6`; the compact depth/context contract hash is `aef9b97140d241de0aefe5ff98a6d600482ef29bbcc89cfb3fe63e359b851edb`.

## Evidence labels

- `official-checkpoint-derived`: computed by the pinned model and checkpoint.
- `nuscenes-annotation-derived`: rasterized from pinned sample annotations and calibration.
- `paper`: transcribed result or claim from the ECCV 2020 paper.
- `teaching`: deterministic explanatory construction, never presented as trained-model output.
