"""G1 URDF loading.

The shared LAFAN1 G1 URDF declares <mujoco><compiler meshdir="meshes"/></mujoco>
AND its <mesh filename> entries are also "meshes/...", causing MuJoCo to look
for "meshes/meshes/foo.STL". We rewrite the meshdir to absolute and strip the
redundant filename prefix.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco


G1_URDF = Path("data/robot/g1/g1_29dof_rev_1_0.urdf")


def load_g1_model() -> mujoco.MjModel:
    urdf_text = G1_URDF.read_text()
    abs_meshdir = (G1_URDF.parent / "meshes").resolve()
    patched = urdf_text.replace('meshdir="meshes"', f'meshdir="{abs_meshdir}"')
    patched = patched.replace('filename="meshes/', 'filename="')
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as tmp:
        tmp.write(patched)
        tmp_path = tmp.name
    return mujoco.MjModel.from_xml_path(tmp_path)
