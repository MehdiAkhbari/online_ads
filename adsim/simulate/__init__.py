"""Simulation entry-points.

Each submodule is runnable via `python -m adsim.simulate.<name>`:

    python -m adsim.simulate.base_ad_helpers
    python -m adsim.simulate.monopoly
    python -m adsim.simulate.duopoly
    python -m adsim.simulate.duopoly_root_n

All modules are import-safe (no I/O at import time); call their `main()`
function or run them as `__main__`.
"""
