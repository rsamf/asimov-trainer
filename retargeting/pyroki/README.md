# LAFAN1 → Asimov retargeting (PyRoKi)

Two-stage CLI that converts a [LAFAN1 retargeting dataset](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset)
G1 motion (29 joint angles + root pose, 30 fps CSV) into an Asimov-v1
motion (root pose + 27 joint angles, viewer-compatible `.npz`).

## Pipeline

```
LAFAN1 G1 CSV   ─►  keypoint .npy   ─►  Asimov .npz
   (input)          g1_to_keypoints    retarget_to_asimov
```

* **Stage 1 — `g1_to_keypoints.py`**
  Loads the G1 URDF, applies forward kinematics frame-by-frame using the
  joint angles from the CSV, then samples 15 SMPL-style keypoints
  (pelvis, hips, knees, ankles, feet, shoulders, elbows, wrists) + 3
  auxiliary points (hand × 2, torso × 1) at each step. Foot contacts are
  inferred from foot z-height + velocity. Output schema matches
  ProtoMotions' `extract_keypoints_from_*.npy`.

* **Stage 2 — `retarget_to_asimov.py`**
  PyRoKi (JAX-LS based) whole-body optimisation. Simultaneously fits the
  Asimov URDF to the keypoints across the whole clip, with smoothness,
  joint-limit, velocity-limit, foot-contact, and foot-tilt costs.
  Produces `base_frame_pos (T,3)`, `base_frame_wxyz (T,4)`,
  `joint_angles (T, 27)` — viewer-compatible.

## Usage

```bash
# Stage 1: a single CSV → a keypoint .npy. Use --max-frames to cap at
# the first N frames (default = all 3945 of dance1_subject3, etc).
.venv/bin/python -m retargeting.pyroki.g1_to_keypoints \
    data/motions/g1/dance1_subject3.csv \
    --out data/motions/g1-keypoints/dance1_subject3_keypoints.npy

# Stage 2: a directory of keypoint .npys → directory of retargeted .npzs.
.venv/bin/python -m retargeting.pyroki.retarget_to_asimov \
    --keypoints-folder-path data/motions/g1-keypoints \
    --output-dir data/motions/asimov-v1-pyroki \
    --source-type smpl \
    --target-raw-frames 450 \
    --no-visualize \
    --skip-existing
```

### `retarget_to_asimov` flags

| flag | default | what it does |
| --- | --- | --- |
| `--source-type {smpl,rigv1}` | `smpl` | Keypoint-scaling heuristic (use `smpl` for G1/LAFAN1 input). |
| `--subsample-factor N` | `1` | Take every Nth raw frame before solving. Cuts memory + solve time. |
| `--target-raw-frames N` | `450` | Fixed solver buffer length (raw frames). 450 ≈ 15 s @ 30 fps. The full 3945-frame `dance1_subject3` takes ~1 h on CPU but is reachable; smaller windows are minutes. |
| `--input-fps F` | `30` | Source FPS, used by the joint-velocity-limit cost. |
| `--skip-existing` | off | Skip motions whose `*_retargeted.npz` is already present. |
| `--no-visualize` | viser on | Run headless and write `.npz` files; without it, opens a [viser](https://github.com/nerfstudio-project/viser) GUI with a live weight-tuner. |

### Interactive tuning

Drop `--no-visualize` and you get a viser server on `http://localhost:8080`
with a `WeightTuner` panel for the cost terms and a "Retarget Next" button
to walk the keypoint directory:

```bash
.venv/bin/python -m retargeting.pyroki.retarget_to_asimov \
    --keypoints-folder-path data/motions/g1-keypoints \
    --source-type smpl
```

### Replaying a result

```bash
.venv/bin/python -m retargeting.pyroki.viewer \
    data/motions/asimov-v1-pyroki/dance1_subject3_keypoints_retargeted.npz
```

This is also what `dancer.viewer` wraps for evaluating trained policies —
the same `.npz` schema works for both.

## Output schema

`.npz` files written by `retarget_to_asimov.py`:

| key | shape | dtype | notes |
| --- | --- | --- | --- |
| `base_frame_pos` | `(T, 3)` | float32 | world pelvis position |
| `base_frame_wxyz` | `(T, 4)` | float32 | world pelvis orientation, **wxyz** |
| `joint_angles` | `(T, 27)` | float32 | full Asimov URDF joint order |

Note: `base_frame_wxyz` is **wxyz** for compatibility with PyRoKi's `jaxlie.SE3` output.
Downstream code (`dancer/env/motion.py`) converts to xyzw on load.
