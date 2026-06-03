"""Render an eval/playback .npz to an MP4 video (headless, FK only).

Uses MuJoCo's offscreen renderer purely for VISUALISATION (not physics): sets
the free-joint + hinge qpos from the trajectory each frame, renders with a
camera that tracks the pelvis, and pipes frames to ffmpeg.

The .npz schema matches `dancer.eval` / `retarget_to_asimov`:
    base_frame_pos  (T, 3)
    base_frame_wxyz (T, 4)   wxyz  (== MuJoCo free-joint quat order)
    joint_angles    (T, 27)  hinge order == MJCF joint order

Usage:
    MUJOCO_GL=glfw .venv/bin/python -m dancer.render_video \
        runs/<run>/champion_full_eval.npz --out /tmp/dance.mp4
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np

ASIMOV_XML = "data/robot/asimov-v1/xmls/asimov.xml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path)
    p.add_argument("--xml", type=str, default=ASIMOV_XML)
    p.add_argument("--out", type=Path, default=Path("/tmp/dance.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--distance", type=float, default=3.2, help="camera distance (m)")
    p.add_argument("--azimuth", type=float, default=120.0)
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    args = p.parse_args()

    data = np.load(args.npz)
    pos = data["base_frame_pos"]            # (T,3)
    wxyz = data["base_frame_wxyz"]          # (T,4) wxyz
    jq = data["joint_angles"]               # (T,27)
    T = pos.shape[0]

    model = mujoco.MjModel.from_xml_path(args.xml)
    dat = mujoco.MjData(model)
    nhinge = model.nq - 7                    # free joint is 7 qpos
    assert jq.shape[1] >= nhinge, f"npz has {jq.shape[1]} joints, model wants {nhinge}"

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation
    opt = mujoco.MjvOption()

    # ffmpeg: raw rgb24 frames on stdin -> H.264 mp4.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
         "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", str(args.out)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    frames = range(0, T, args.stride)
    for n, t in enumerate(frames):
        dat.qpos[:3] = pos[t]
        dat.qpos[3:7] = wxyz[t]
        dat.qpos[7:7 + nhinge] = jq[t, :nhinge]
        mujoco.mj_forward(model, dat)
        cam.lookat[:] = pos[t]               # track the pelvis
        renderer.update_scene(dat, camera=cam, scene_option=opt)
        ff.stdin.write(renderer.render().tobytes())
        if n % 300 == 0:
            print(f"  frame {n}/{len(frames)}")
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out}  ({len(frames)} frames @ {args.fps} fps, "
          f"{len(frames)/args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
