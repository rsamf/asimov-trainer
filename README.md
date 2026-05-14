# Asimov Trainer

Tooling for the **Asimov-v1** humanoid: train motion-imitation policies in
[Newton](https://github.com/NVlabs/Newton), and retarget LAFAN1 mocap onto
the robot via [PyRoKi](https://github.com/pyroki/pyroki).

## Repo layout

```
asimov-trainer/
├── dancer/                       # RL training stack (motion imitation)
│   ├── env/                      #   Newton-based vectorised env, motion ref, reward
│   ├── policy/                   #   actor / critic networks
│   ├── algos/                    #   PPO + rollout buffer
│   ├── configs/                  #   hydra configs (env, algo, network, reward, viewer)
│   ├── train.py                  #   PPO entry point
│   ├── eval.py                   #   roll a checkpoint, write a playback .npz
│   └── viewer.py                 #   thin wrapper around the viser playback viewer
├── retargeting/
│   ├── joint_map.py              # canonical Asimov + LAFAN1 joint-name tables
│   ├── g1.py                     # G1 URDF loader (patched mesh paths)
│   └── pyroki/                   # LAFAN1 → Asimov retargeting CLI (see its README)
├── data/
│   ├── robot/asimov-v1/          # MJCF + URDF for the robot
│   ├── robot/g1/                 # symlink to G1 URDF (used as kinematic intermediate)
│   └── motions/                  # source LAFAN1 + retargeted outputs (gitignored)
└── tests/                        # unit tests (motion ref, joint map)
```

## Quick start

```bash
# 1. Install (editable, CUDA-12.4 torch wheel pinned in pyproject)
uv sync

# 2. Sanity-check that physics + reference motion load
.venv/bin/python -m dancer.eval --replay-reference --n-steps 300
.venv/bin/python -m dancer.viewer eval.npz       # opens viser

# 3. Train (default: 1024 envs, 2000 PPO iters)
.venv/bin/python -m dancer.train

# 4. Train with live viewer
.venv/bin/python -m dancer.train viewer.kind=rerun
#   → open http://localhost:9090
```

Override any hydra config from the CLI, e.g.:

```bash
.venv/bin/python -m dancer.train \
    experiment_name=dance_v2 \
    algo.lr=1e-4 algo.iterations=500 \
    env.num_envs=512 \
    viewer.kind=viser
```

## Training stack

| | choice | rationale |
|---|---|---|
| Sim | **Newton 1.2+** (`SolverMuJoCo`, GPU) | batched, mujoco_warp under the hood, supports the Asimov MJCF |
| Policy | 2-layer 1024-unit MLP, diagonal Gaussian | DeepMimic-style baseline |
| Algo | PPO with GAE, clipped value loss, KL early-stop | well-understood for motion imitation |
| Config | **hydra-core** + **hydr8** | structured, override-from-CLI friendly |
| Logging | **nebo** | run lifecycle + scalar metrics |
| Viz | **viser** (playback), **Newton viewer / rerun** (live training) | rerun gets a web server at port 9090; viser at 8080 |

Observation (179-d) is **own state + lookahead targets only**, deliberately
without phase scalar or gravity-frame projection — the policy infers phase
from the future-target window. Reward and action exclude the 4 passive
joints (toes + neck); the real Asimov can't drive them.

Configs live under `dancer/configs/`. The defaults set 1024 parallel envs,
60 Hz physics with policy at 30 Hz, kp=100 / kd=5 PD, residual action of
±0.5 rad around the reference. Modify `dancer/configs/{env,algo,network,reward,viewer}.yaml`
or override on the CLI.

## Retargeting LAFAN1 → Asimov

See [`retargeting/pyroki/README.md`](retargeting/pyroki/README.md) for the
two-stage CLI.

In short:
```bash
# Stage 1: LAFAN1 G1 CSV → SMPL-style keypoints via G1 FK
.venv/bin/python -m retargeting.pyroki.g1_to_keypoints \
    data/motions/g1/dance1_subject3.csv \
    --out data/motions/g1-keypoints/dance1_subject3_keypoints.npy

# Stage 2: keypoints → Asimov via PyRoKi optimisation
.venv/bin/python -m retargeting.pyroki.retarget_to_asimov \
    --keypoints-folder-path data/motions/g1-keypoints \
    --output-dir data/motions/asimov-v1-pyroki \
    --source-type smpl --no-visualize
```

The pre-retargeted reference used by training is at
`data/motions/asimov-v1-pyroki-full/dance1_subject3_keypoints_retargeted.npz`.
