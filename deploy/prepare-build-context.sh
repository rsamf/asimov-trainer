#!/usr/bin/env bash
# Materialise the docker build context: copy sibling repos and resolved
# symlink targets into the current tree so `docker build .` works without
# referring to parent directories.
#
# Idempotent. Local devs: run from repo root before `docker build .`.
# CI: invoked by .github/workflows/train_model.yml after checkouts.
#
# Inputs (env vars, all optional, override defaults):
#   PYROKI_SRC              — path to pyroki source (default: ../pyroki)
#   LOCAL_ROBOT_FILES_SRC   — path to local-robot-files (default: ../local-robot-files)

set -euo pipefail

PYROKI_SRC="${PYROKI_SRC:-../pyroki}"
LOCAL_ROBOT_FILES_SRC="${LOCAL_ROBOT_FILES_SRC:-../local-robot-files}"

ROBOT_DST="data/robot/asimov-v1"
ROBOT_SRC="${LOCAL_ROBOT_FILES_SRC}/asimov-v1/sim-model"

# 1. pyroki -> ./pyroki (copy, not symlink — docker won't follow out-of-context symlinks).
if [[ ! -d pyroki || -L pyroki ]]; then
    if [[ ! -d "$PYROKI_SRC" ]]; then
        echo "ERROR: pyroki source not found at $PYROKI_SRC" >&2
        exit 1
    fi
    rm -rf pyroki
    cp -R "$PYROKI_SRC" pyroki
    # Strip the editable's own .venv / __pycache__ to keep build context small.
    rm -rf pyroki/.venv pyroki/.git
    find pyroki -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
    echo "materialised pyroki/  (from $PYROKI_SRC)"
fi

# 2. data/robot/asimov-v1 — replace symlink with real directory.
if [[ -L "$ROBOT_DST" ]]; then
    if [[ ! -d "$ROBOT_SRC" ]]; then
        echo "ERROR: robot source not found at $ROBOT_SRC" >&2
        exit 1
    fi
    rm "$ROBOT_DST"
    cp -R "$ROBOT_SRC" "$ROBOT_DST"
    echo "materialised $ROBOT_DST  (from $ROBOT_SRC)"
fi

# 3. Sanity: motion .npz exists.
MOTION_NPZ="data/motions/asimov-v1-pyroki-full/dance1_subject3_keypoints_retargeted.npz"
if [[ ! -f "$MOTION_NPZ" ]]; then
    echo "ERROR: $MOTION_NPZ missing. Run the pyroki retarget pipeline first." >&2
    exit 1
fi

echo "build context ready."
