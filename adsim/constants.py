"""Re-exports `DATA_DIR` / `RESULTS_DIR` from `adsim.paths`.

Kept as a back-compat shim because `adsim/config.py` and a few notebooks
import `PATH_ROOT` from here. New code should import from `adsim.paths`
directly.
"""

from adsim.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR

# Legacy name. New code should use `REPO_ROOT` from `adsim.paths`.
PATH_ROOT = str(REPO_ROOT)

__all__ = ["DATA_DIR", "PATH_ROOT", "REPO_ROOT", "RESULTS_DIR"]
