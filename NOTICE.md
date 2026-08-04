# Notices and third-party material

## Site code

Original code in this repository is released under the MIT License. The visual series is authored independently and does not grant rights to third-party data, papers, source code, or checkpoints.

## nuScenes and OpenMMLab demo assets

The six camera images, one LIDAR_TOP frame, and metadata originate from the nuScenes dataset through the pinned MMDetection3D demo. nuScenes data is subject to the [nuScenes Dataset Terms of Use](https://www.nuscenes.org/terms-of-use), including non-commercial restrictions. These fixed-sample assets are included solely to explain and audit the research example; redistribution or downstream use must independently comply with those terms.

MMDetection3D source is Apache-2.0 licensed. This repository does not copy MMDetection3D source; it consumes pinned demo assets and records their hashes.

## NVIDIA Lift-Splat-Shoot

The official Lift-Splat-Shoot source is distributed under the NVIDIA Source Code License and the pretrained checkpoint is a separate NVIDIA artifact. Neither source nor checkpoint is committed here. The reproduction script downloads them into ignored `.cache/` storage and commits only derived numerical/visual evidence. Use of those inputs remains governed by NVIDIA's license and is intended here for non-commercial research and evaluation.

## Paper

Paper text, figures, and tables remain copyright their authors/publisher. This site paraphrases the method and reproduces small numerical facts needed for technical commentary; it does not redistribute the paper.

## Fonts and dependencies

The web application uses open-source packages listed in `package.json` and web fonts served by Google Fonts. Their respective licenses apply.
