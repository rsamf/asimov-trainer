"""Sanity checks on the canonical Asimov / LAFAN1 joint-name tables."""

from pathlib import Path

import mujoco

from retargeting.joint_map import ASIMOV_ACTUATED_JOINT_NAMES, G1_CSV_JOINT_ORDER


MJCF = Path(__file__).resolve().parents[1] / "data/robot/asimov-v1/xmls/asimov.xml"


def test_actuated_count():
    assert len(ASIMOV_ACTUATED_JOINT_NAMES) == 23


def test_actuated_unique():
    assert len(set(ASIMOV_ACTUATED_JOINT_NAMES)) == len(ASIMOV_ACTUATED_JOINT_NAMES)


def test_actuated_names_exist_in_mjcf():
    """Every name in the actuated list must be a real joint on the Asimov MJCF."""
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    for name in ASIMOV_ACTUATED_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"joint {name!r} not found in MJCF"


def test_passive_set():
    """The 27 hinges minus our 23 actuated should leave exactly the 4 passive."""
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    hinge_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == 3:  # hinge
            hinge_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    passive = set(hinge_names) - set(ASIMOV_ACTUATED_JOINT_NAMES)
    assert passive == {
        "left_toe_joint", "right_toe_joint",
        "neck_yaw_joint", "neck_pitch_joint",
    }, f"unexpected passive set: {passive}"


def test_g1_csv_joint_order_length():
    assert len(G1_CSV_JOINT_ORDER) == 29
    assert len(set(G1_CSV_JOINT_ORDER)) == 29
