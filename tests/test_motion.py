"""Unit tests for dancer/env/motion.py."""

from pathlib import Path

import numpy as np
import torch

try:
    import pytest                                # noqa: F401
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
    class _Stub:                                 # minimal pytest shim so the file is importable
        @staticmethod
        def fixture(*a, **kw):
            def wrap(fn): return fn
            return wrap
    pytest = _Stub()                             # type: ignore

from dancer.env.motion import MotionRef, quat_log_xyz, quat_mul, slerp


REPO = Path(__file__).resolve().parents[1]
NPZ = REPO / "data/motions/asimov-v1-pyroki-full/dance1_subject3_keypoints_retargeted.npz"
XML = REPO / "data/robot/asimov-v1/xmls/asimov.xml"
URDF = REPO / "data/robot/asimov-v1/xmls/asimov.urdf"


@pytest.fixture(scope="module")
def ref() -> MotionRef:
    return MotionRef.load(NPZ, XML, URDF, physics_fps=60, src_fps=30, device="cpu")


def test_upsampled_shape(ref: MotionRef):
    assert ref.upsample == 2
    assert ref.T_up == (ref.T_src - 1) * ref.upsample + 1
    assert ref.base_pos.shape == (ref.T_up, 3)
    assert ref.base_quat.shape == (ref.T_up, 4)
    assert ref.joint_q.shape == (ref.T_up, 27)
    assert ref.joint_qd.shape == (ref.T_up, 27)
    assert ref.base_lin_vel.shape == (ref.T_up, 3)
    assert ref.base_ang_vel.shape == (ref.T_up, 3)


def test_actuated_passive_partition(ref: MotionRef):
    assert ref.n_actuated == 23
    assert ref.actuated_idx.shape == (23,)
    assert ref.passive_idx.shape == (4,)
    # Disjoint + covers all 27 joints.
    union = set(ref.actuated_idx.tolist()) | set(ref.passive_idx.tolist())
    assert union == set(range(27))
    # Passive set is exactly the toes + neck.
    import pyroki as pk
    import yourdfpy
    urdf = yourdfpy.URDF.load(str(URDF), mesh_dir=str(URDF.parent))
    names = list(pk.Robot.from_urdf(urdf).joints.actuated_names)
    passive_names = {names[int(i)] for i in ref.passive_idx}
    assert passive_names == {
        "left_toe_joint", "right_toe_joint",
        "neck_yaw_joint", "neck_pitch_joint",
    }


def test_at_keyframes_match_source(ref: MotionRef):
    """Upsampled frames at multiples of `upsample` should reproduce the source exactly."""
    d = np.load(NPZ)
    for k in (0, 1, 100, ref.T_src - 1):
        out = ref.at(torch.tensor(k * ref.upsample))
        src_pos = torch.tensor(d["base_frame_pos"][k])
        src_quat_xyzw = torch.tensor(d["base_frame_wxyz"][k])[[1, 2, 3, 0]]
        src_q = torch.tensor(d["joint_angles"][k])

        assert torch.allclose(out["base_pos"], src_pos, atol=0.0)
        assert torch.allclose(out["base_quat"], src_quat_xyzw, atol=1e-5)
        assert torch.allclose(out["joint_q"], src_q, atol=0.0)


def test_lookahead_actuated_shape(ref: MotionRef):
    B, K, stride = 32, 4, ref.upsample
    idx = torch.zeros(B, dtype=torch.long)
    fut = ref.lookahead_actuated(idx, K=K, stride=stride)
    assert fut.shape == (B, K, 23)
    # K policy steps ahead corresponds to K * upsample upsampled frames.
    # fut[..., 0, :] is the actuated slice at idx + stride.
    expected = ref.joint_q[stride][ref.actuated_idx]
    assert torch.allclose(fut[0, 0], expected, atol=0.0)


def test_finite_diff_velocity_on_synthetic():
    """Sinusoidal joint_q → analytic d/dt should match finite diff within ~1%."""
    # Synthesize a 1-joint motion: q(t) = sin(ωt), q̇(t) = ω cos(ωt).
    src_fps = 30
    T_src = 200
    dt_src = 1.0 / src_fps
    t_src = np.arange(T_src) * dt_src
    omega = 2.0  # rad/s
    q_src = np.sin(omega * t_src).astype(np.float32)
    qd_expected = omega * np.cos(omega * t_src).astype(np.float32)

    # Build a minimal MotionRef-like object by directly invoking the helpers.
    # We test the loader's finite-diff logic via a tiny end-to-end exercise:
    # save a temporary npz with these joints and load.
    import tempfile, os

    # Need a base_pos and base_quat with the right shape for the loader.
    base_pos = np.zeros((T_src, 3), dtype=np.float32)
    base_quat_wxyz = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (T_src, 1))
    # Loader expects 27 joint columns — pad zeros and put our signal in column 0
    # via the URDF order. We just verify the first (joint-0) column behaves.
    joints = np.zeros((T_src, 27), dtype=np.float32)
    joints[:, 0] = q_src

    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "synth.npz"
        np.savez(npz, base_frame_pos=base_pos,
                 base_frame_wxyz=base_quat_wxyz, joint_angles=joints)
        ref = MotionRef.load(npz, XML, URDF, physics_fps=60, src_fps=30, device="cpu")

    # joint_qd at frame i (upsampled) should be (q[i] - q[i-1]) / (1/60).
    # For our synthetic sin: at indices that line up with source frames, the
    # finite diff over physics dt = 1/60 sampled from a 30 Hz source should
    # still match ω cos within a few %.
    sample_up_indices = [50, 100, 150, 200]
    for i in sample_up_indices:
        t = i / ref.physics_fps
        analytic = omega * np.cos(omega * t)
        observed = float(ref.joint_qd[i, 0])
        # tolerance ~5% (finite differencing on upsampled-by-lerp signal
        # introduces small error; the test is about order-of-magnitude
        # correctness of the dt rate)
        assert abs(observed - analytic) < 0.1, (
            f"joint_qd[{i}, 0]={observed:.4f} vs analytic {analytic:.4f}"
        )


def test_slerp_halfway():
    """SLERP at α=0.5 between two quats should split the angle in half."""
    # 0 and 90° around z, then halfway should be 45° around z.
    q0 = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity (xyzw)
    q1 = torch.tensor([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90° z
    alpha = torch.tensor([0.5])
    mid = slerp(q0.unsqueeze(0), q1.unsqueeze(0), alpha)[0]
    expected = torch.tensor([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)])
    assert torch.allclose(mid, expected, atol=1e-6)


def test_quat_log_xyz_identity():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # identity
    out = quat_log_xyz(q)
    assert torch.allclose(out, torch.zeros(1, 3), atol=1e-7)


def test_quat_log_xyz_90deg_z():
    # 90° rotation about +z (xyzw)
    s, c = np.sin(np.pi / 4), np.cos(np.pi / 4)
    q = torch.tensor([[0.0, 0.0, s, c]])
    out = quat_log_xyz(q)
    # log(q) should be (0, 0, π/4) (half-angle * axis)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, np.pi / 4]], dtype=out.dtype), atol=1e-6)
