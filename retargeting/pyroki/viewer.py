"""Playback viewer for retargeted Asimov motions in viser.

Loads a `.npz` produced by `retarget_to_asimov.py` (`base_frame_pos`,
`base_frame_wxyz`, `joint_angles`) plus the Asimov URDF and streams the
trajectory to a viser server. No optimisation, just FK playback at the
recorded fps.

Usage:
    .venv/bin/python -m retargeting.pyroki.viewer \\
        data/motions/asimov-v1-pyroki/dance1_subject3_keypoints_retargeted.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from .retarget_to_asimov import ASIMOV_MESH_DIR, ASIMOV_URDF


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="Retargeted Asimov .npz from retarget_to_asimov.py")
    p.add_argument("--urdf-path", type=Path, default=ASIMOV_URDF)
    p.add_argument("--mesh-dir", type=Path, default=ASIMOV_MESH_DIR)
    p.add_argument("--fps", type=float, default=30.0,
                   help="Playback rate. Source LAFAN1 is 30 fps.")
    args = p.parse_args()

    data = np.load(args.npz)
    base_pos  = data["base_frame_pos"]    # (T, 3)
    base_wxyz = data["base_frame_wxyz"]   # (T, 4)
    joints    = data["joint_angles"]      # (T, n_actuated)
    T = base_pos.shape[0]
    print(f"loaded {args.npz.name}: {T} frames, {joints.shape[1]} joints @ {args.fps} fps")

    urdf = yourdfpy.URDF.load(str(args.urdf_path), mesh_dir=str(args.mesh_dir))

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    server.scene.add_grid("/ground", width=4.0, height=4.0)

    playing = server.gui.add_checkbox("playing", True)
    timestep = server.gui.add_slider("timestep", 0, T - 1, 1, 0)

    def reset(_: viser.GuiEvent) -> None:
        timestep.value = 0

    server.gui.add_button("reset").on_click(reset)

    print(f"viser running on http://localhost:8080  (open in browser)")
    while True:
        with server.atomic():
            if playing.value:
                timestep.value = (timestep.value + 1) % T
            t = timestep.value
            base_frame.position = np.asarray(base_pos[t])
            base_frame.wxyz = np.asarray(base_wxyz[t])
            urdf_vis.update_cfg(np.asarray(joints[t]))
        time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
