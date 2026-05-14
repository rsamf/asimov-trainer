"""Thin wrapper around the existing viser-based motion viewer.

The .npz schema this expects is the same one `eval.py` writes (which is
the same one `retargeting/pyroki/viewer.py` already consumes).

    .venv/bin/python -m dancer.viewer runs/<run>/eval_final.npz
"""

from retargeting.pyroki.viewer import main


if __name__ == "__main__":
    raise SystemExit(main())
