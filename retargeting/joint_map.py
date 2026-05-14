"""Canonical joint-name tables for LAFAN1 → Asimov retargeting.

The pyroki pipeline needs two things from this module:

* `G1_CSV_JOINT_ORDER` — the order of the 29 joint columns in a LAFAN1 G1
  CSV row, after the 7-value root prefix `[X, Y, Z, QX, QY, QZ, QW]`. Used
  by `retargeting/pyroki/g1_to_keypoints.py` to index the input CSV.

* `ASIMOV_ACTUATED_JOINT_NAMES` — the 23 names of Asimov's actuated joints,
  in MJCF actuator order. Used by `dancer/env/motion.py` as the source of
  truth for which joints get PD targets and appear in reward/observation
  slicing.

The Asimov MJCF intentionally omits the `<actuator>` block (actuators are
added programmatically by downstream simulators), so this module is the
canonical source for both.
"""

from __future__ import annotations


# 29 joint columns in a LAFAN1 G1 CSV row (after the 7-value root prefix).
G1_CSV_JOINT_ORDER: list[str] = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


# Asimov-v1's 23 actuated joints, in MJCF actuator order. Toes (×2) and neck
# (yaw + pitch) are passive — never PD-controlled, never in the action space,
# never compared in the tracking reward.
ASIMOV_ACTUATED_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
]
