# Asimov Trainer

Tooling for the **Asimov-v1** humanoid: train motion-imitation policies in
[Newton](https://github.com/NVlabs/Newton), and retarget LAFAN1 mocap onto
the robot via [PyRoKi](https://github.com/pyroki/pyroki).

## Quick start

```bash
# 1. Install (editable, CUDA-12.4 torch wheel pinned in pyproject)
uv sync

# 2. Sanity-check that physics + reference motion load
uv run python -m dancer.eval --replay-reference --n-steps 300
uv run python -m dancer.viewer eval.npz       # opens viser

# 3. Train (default: 1024 envs, 2000 PPO iters)
uv run python -m dancer.train
#   on an 8 GB local GPU (RTX 3070-class), use the pre-tuned config:
#   uv run python -m dancer.train --config-name=train_rtx3070

# 4. Train with live viewer
uv run python -m dancer.train viewer.kind=rerun
#   → open http://localhost:9090
```

## Training

### Pick a config

| Config | `env.num_envs` | When to use |
|---|---|---|
| `dancer/configs/train.yaml` (default) | 1024 | Cloud GPU (V100/A100), or sanity runs |
| `dancer/configs/train_rtx3070.yaml`   | 8192 | Local 8 GB GPU; ~4.3 GB peak, ~80% util, ~1.6 s/iter |

```bash
uv run python -m dancer.train --config-name=train_rtx3070
```

The header of `train_rtx3070.yaml` documents the throughput / VRAM benchmarks
behind the choice (and why it deliberately leaves a margin below the 16384-env
ceiling). Any field of either config can be overridden on the CLI:

```bash
uv run python -m dancer.train --config-name=train_rtx3070 \
    experiment_name=dance_v2 \
    algo.lr=1e-4 algo.iterations=500 \
    env.num_envs=4096
```

Saved checkpoints go into `runs/<experiment_name>-<unix_timestamp>/`.

### Live viewer during training

```bash
uv run python -m dancer.train --config-name=train_rtx3070 viewer.kind=rerun
# → http://localhost:9090
```

`viewer.kind` accepts `null` (default, headless), `rerun`, `viser`, `gl` (native
window), or `usd` (writes a USD timeline to the run dir). Streaming costs
throughput; the recommended pattern is headless training, then evaluate
checkpoints in a viewer afterward (see below).

### Checkpoints & logs

Every `eval.every` iterations (default **25**) plus once at the end, the actor
+ critic + Adam state is saved to `runs/<run>/ckpt_<iter>.pt`. Each file is
~20 MB; at the default 2000 iters that's ~81 files / ~1.6 GB per run — override
`eval.every=200` if you want fewer. Hydra also writes the resolved config to
`runs/<run>/.hydra/`. [nebo](https://github.com/nebopipeline/nebo) events go to
`./.nebo/<timestamp>_<run_id>.nebo` by default; set `NEBO_URI=<dir>` in the
environment to redirect.

## Evaluating a trained policy

Two-step flow: deterministic rollout → playback viewer.

```bash
# 1. Roll the checkpoint to a trajectory .npz (greedy actions, fall-termination
#    disabled so a stumbling policy still plays the full window).
uv run python -m dancer.eval \
    --checkpoint runs/<your-run>/ckpt_final.pt \
    --out runs/<your-run>/eval_final.npz \
    --n-steps 600                                # ~20 s of motion at 30 Hz

# 2. Play it back in a browser (viser server on :8080).
uv run python -m dancer.viewer runs/<your-run>/eval_final.npz
```

As a sanity baseline, the reference motion itself can be replayed the same way
— `--replay-reference` drives the env with `action=0`; the residual-action
design plays the reference exactly:

```bash
uv run python -m dancer.eval --replay-reference --out runs/_ref.npz
uv run python -m dancer.viewer runs/_ref.npz
```

A trained policy should look qualitatively close to this reference and stay
upright through the window; nebo's `train/avg_return`, `train/fall_fraction`,
and `train/avg_episode_len` are the scalar counterparts.

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
uv run python -m retargeting.pyroki.g1_to_keypoints \
    data/motions/g1/dance1_subject3.csv \
    --out data/motions/g1-keypoints/dance1_subject3_keypoints.npy

# Stage 2: keypoints → Asimov via PyRoKi optimisation
uv run python -m retargeting.pyroki.retarget_to_asimov \
    --keypoints-folder-path data/motions/g1-keypoints \
    --output-dir data/motions/asimov-v1-pyroki \
    --source-type smpl --no-visualize
```

The pre-retargeted reference used by training is at
`data/motions/asimov-v1-pyroki-full/dance1_subject3_keypoints_retargeted.npz`.
