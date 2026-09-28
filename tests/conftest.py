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


@pytest.fixture(scope="session")
def make_estimation_data():
    """Synthetic estimation file in the raw layout (base ad coded 50, a fringe rank 150).

    Users visit 2-12 pages in order (`user_visit_no`). The ad shown depends on
    page context (hour, subject); with `selection=True` ad 1 is also shown more
    to users with many past subject-1 visits, who also click more, which is
    the confounding Assumption 1 rules out. Ad 1's lift is larger in the
    evening, so forests have heterogeneity to find.
    """
    def _make(n_users=1_500, selection=False, seed=0):
        rng = np.random.default_rng(seed)
        visits = rng.integers(2, 13, n_users)
        user = np.repeat(np.arange(n_users), visits)
        n = len(user)
        affinity = rng.normal(size=n_users)[user]
        df = pd.DataFrame({"global_token_new": user,
                           "user_visit_no": np.concatenate([np.arange(1, v + 1) for v in visits])})
        subject = rng.integers(1, 14, n)
        for i in range(1, 14):
            df[f"sub_{i}"] = (subject == i).astype(int)
        df["publisher_rank_sub"] = rng.integers(1, 4, n)
        df["day"] = rng.integers(1, 6, n)
        df["hour"] = rng.integers(0, 24, n)
        df["mobile"] = (rng.random(n_users) < 0.6).astype(int)[user]
        df["ads_on_page"] = rng.integers(1, 6, n)
        df["total_visits"] = df["user_visit_no"]
        for i in range(1, 14):
            df[f"visit_s{i}"] = rng.poisson(np.exp(0.6 * affinity) if i == 1 else np.ones(n))
        df["impression_repeat"] = rng.poisson(1.0, n) + 1
        df["impression_repeat_base_ad"] = rng.poisson(1.0, n) + 1
        df["previous_clicks"] = rng.binomial(1, 0.05, n)
        df["previous_clicks_base_ad"] = rng.binomial(1, 0.05, n)
        df["previous_clicks_all_ads"] = rng.poisson(0.3 * np.exp(0.5 * affinity))

        arms = np.array([50, 1, 2, 3])
        logits = np.zeros((n, 4))
        logits[:, 1] = 0.8 * (df["hour"] >= 12) - 0.3
        logits[:, 2] = 0.6 * df["sub_1"] - 0.2
        if selection:
            logits[:, 1] += 1.2 * (df["visit_s1"] - df["visit_s1"].mean())
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        shown = arms[(rng.random((n, 1)) > probs.cumsum(1)).sum(1)]
        shown[rng.random(n) < 0.01] = 150
        df["advertiser_rank"] = shown

        lift = np.select([shown == 1, shown == 2, shown == 3], [1.3 + 1.2 * (df["hour"] >= 18), 0.8, 1.1], 1.0)
        p = np.clip(0.03 * np.exp(0.5 * affinity) * lift, 0, 1)
        df["is_clicked"] = (rng.random(n) < p).astype(int)
        return df

    return _make
