# Claude session guide for asimov-trainer

Project-level conventions and gotchas for future Claude sessions. Read
this first; it overrides defaults where they conflict.

## What this project is

Tooling for the **Asimov-v1** humanoid robot:

1. `dancer/` — motion-imitation RL training (PPO + DeepMimic-style reward,
   Newton physics, hydra configs, nebo logging).
2. `retargeting/pyroki/` — CLI that converts LAFAN1 G1 motions to Asimov
   joint trajectories via PyRoKi optimisation.

There is **no** AMASS pipeline, no G1 *training* support, and no
direct-mapping retargeter — those were earlier directions that have been
removed. Don't reintroduce them without an explicit ask.

## Stack (don't substitute without asking)

* **Newton** (`>=1.2.0`) for physics simulation. NOT `mujoco.viewer` /
  `mj_step` at runtime — only `mujoco` Python is used for MJCF *parsing*.
* **Rerun** (and **viser** for playback) for visualisation.
* **Newton's built-in viewers** (`ViewerRerun`, `ViewerViser`, `ViewerGL`,
  `ViewerUSD`) wired into `dancer/train.py` via `cfg.viewer.kind`.
* **hydra-core** + **hydr8** for configs.
* **nebo** for experiment logging.
* **PyTorch** built against CUDA 12.4 (driver on this machine is 12.6).
  `pyproject.toml` pins `torch` to the `pytorch-cu124` index.

## Asimov MJCF / actuator facts

* The Asimov MJCF (`data/robot/asimov-v1/xmls/asimov.xml`) **deliberately
  omits the `<actuator>` block** — actuators are added programmatically
  downstream. So `mujoco.MjModel.from_xml_path(...).nu == 0`.
* The canonical 23 actuated-joint names live in
  `retargeting/joint_map.py:ASIMOV_ACTUATED_JOINT_NAMES`. Both `dancer/`
  and `retargeting/pyroki/` import from there.
* The 4 passive joints (`left_toe_joint`, `right_toe_joint`,
  `neck_yaw_joint`, `neck_pitch_joint`) are spring-loaded in the MJCF and
  must NEVER appear in:
  * the policy action space (23-d, not 27-d),
  * PD targets in Newton,
  * the tracking reward error terms.
  Observations *may* include them (they're observable from real encoders).

## Quaternion convention

**xyzw** (vector-scalar) end-to-end, because Newton uses xyzw. The pyroki
retarget output `.npz` stores `base_frame_wxyz` (wxyz, for PyRoKi
compatibility) — `dancer/env/motion.py` converts on load.

## Common commands

```bash
# Train (default config, headless)
.venv/bin/python -m dancer.train

# Train with live viewer
.venv/bin/python -m dancer.train viewer.kind=rerun

# Smoke-test physics + motion ref
.venv/bin/python -m dancer.eval --replay-reference --n-steps 300

# Retarget a LAFAN1 motion
.venv/bin/python -m retargeting.pyroki.g1_to_keypoints \
    data/motions/g1/<motion>.csv --out data/motions/g1-keypoints/<motion>.npy
.venv/bin/python -m retargeting.pyroki.retarget_to_asimov \
    --keypoints-folder-path data/motions/g1-keypoints \
    --output-dir data/motions/asimov-v1-pyroki \
    --source-type smpl --no-visualize

# Tests
.venv/bin/python -m pytest tests/        # if pytest installed
# Or, since the test files have a no-pytest fallback shim:
.venv/bin/python -c "from tests import test_motion as t; ..."
```

## Sharp edges to watch for

* **`uv sync` can blow away the cu124 torch wheel.** If you see
  `ImportError: libnccl.so.2: cannot open shared object file` or
  `undefined symbol: ncclCommWindowDeregister`, force-reinstall:
  `uv pip install --python .venv/bin/python nvidia-nccl-cu12==2.21.5 --force-reinstall`
  followed by `uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --reinstall`.
* **mujoco 3.8 + np.float32 pybind bug.** `MjsBody.add_joint(stiffness=np.float32(0))`
  raises `TypeError: stiffness should be a numeric scalar or list.`
  `dancer/env/sim.py` installs a runtime monkey-patch on `MjsBody.add_joint`
  that coerces numpy scalars to Python floats. Don't remove it.
* **Newton spawn lift.** Newton's contact solver is fragile to feet
  starting exactly at floor level. `DanceEnv.reset_idx` uses
  `MotionRef.at()` values directly; those already have the right pelvis
  height for reference frames. If you build a non-RSI reset, add a
  ~5 cm clearance.
* **`wp.set_device('cuda:0')`** must be called early in any sim-touching
  process. `dancer/train.py` and `dancer/eval.py` do this; ad-hoc scripts
  should too.

## Memory pointers

Project memory (persistent across sessions) lives at
`~/.claude/projects/-home-sam-Documents-robotics-asimov-trainer/memory/`.
Key entries:

* `project_tooling.md` — Newton + Rerun + xyzw decision
* `asimov_passive_joints.md` — never control toes / neck
* `feedback_imitation_obs.md` — own state + lookahead K; no phase scalar / gravity proj.

Read these before proposing observation / reward / control changes.
