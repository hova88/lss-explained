#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
mm3d_base="https://raw.githubusercontent.com/open-mmlab/mmdetection3d/fe25f7a51d36e3702f961e198894580d83c4387b/demo/data/nuscenes"
mkdir -p "$repo_root/.cache/openmmlab" "$repo_root/.cache/lss"

files=(
  "n015-2018-07-24-11-22-45+0800.pkl"
  "n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"
  "n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg"
  "n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402927604844.jpg"
  "n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg"
  "n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402927647423.jpg"
  "n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402927627893.jpg"
)
for file in "${files[@]}"; do
  curl -L --fail --retry 3 -o "$repo_root/.cache/openmmlab/$file" "$mm3d_base/$file"
done

if [ ! -d "$repo_root/.cache/lss/repo/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/nv-tlabs/lift-splat-shoot.git "$repo_root/.cache/lss/repo"
fi
git -C "$repo_root/.cache/lss/repo" checkout 2903467c91ee9c12f0917a12c22ab1f04e607ae0
curl -L --fail --retry 3 -o "$repo_root/.cache/lss/model525000.pt" "https://drive.usercontent.google.com/download?id=1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ&export=download&confirm=t"
