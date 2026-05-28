"""adsim — causal-forest-based simulation of online ad serving.

Submodules:
    adsim.paths             REPO_ROOT, DATA_DIR, RESULTS_DIR
    adsim.config            static knobs, ranks_list, forest loaders + registries
    adsim.propensity_model  T-model used by CausalForestDML
    adsim.simulation_steps  estimation helpers + per-step simulation primitives
    adsim.estimate          `python -m adsim.estimate` — fit per-rank causal forests
    adsim.simulate          `python -m adsim.simulate.{base_ad_helpers,monopoly,duopoly,duopoly_root_n}`
"""

__version__ = "0.1.0"
