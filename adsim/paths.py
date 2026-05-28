"""Path helpers for the adsim research code.

`DATA_DIR` and `RESULTS_DIR` resolve to `<repo_root>/data` and
`<repo_root>/results` by default, and can be overridden by the
`ADSIM_DATA_DIR` / `ADSIM_RESULTS_DIR` env vars (useful when the inputs
live on an external drive or a shared filesystem).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = Path(
    os.environ.get("ADSIM_DATA_DIR", REPO_ROOT / "data")
).expanduser().resolve()

RESULTS_DIR: Path = Path(
    os.environ.get("ADSIM_RESULTS_DIR", REPO_ROOT / "results")
).expanduser().resolve()
