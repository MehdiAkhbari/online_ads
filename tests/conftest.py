from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

FEATURES = [
    'impression_repeat', 'impression_repeat_base_ad', 'previous_clicks', 'previous_clicks_base_ad',
    'previous_clicks_all_ads', 'total_visits',
    *[f'visit_s{i}' for i in range(1, 14)],
    *[f'sub_{i}' for i in range(1, 14)],
    'publisher_rank_sub', 'day', 'hour', 'mobile', 'ads_on_page',
]


@pytest.fixture
def make_pair():
    """Synthetic estimation rows for the base ad (rank 0) and one focal rank.

    The focal ad's click rate doubles when `hour > 0.5`, so a forest with
    enough clicks per leaf has real heterogeneity to find.
    """
    def _make(n_base, n_focal, rate_base=0.05, rate_focal=0.05, rank=7, n_users=None, seed=0):
        rng = np.random.default_rng(seed)
        n = n_base + n_focal
        df = pd.DataFrame(rng.random((n, len(FEATURES))), columns=FEATURES)
        df["advertiser_rank"] = np.r_[np.zeros(n_base, dtype=int), np.full(n_focal, rank)]
        lift = np.where(df["hour"] > 0.5, 1.5, 0.5)
        p = np.where(df["advertiser_rank"] == 0, rate_base, rate_focal * lift)
        df["is_clicked"] = (rng.random(n) < p).astype(int)
        df["global_token_new"] = rng.integers(0, n_users or max(2, n // 10), n)
        return df

    return _make
