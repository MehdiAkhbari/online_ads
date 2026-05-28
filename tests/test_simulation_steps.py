"""Tests for the pure-function helpers in adsim.simulation_steps.

These tests deliberately avoid anything that needs a fitted forest or a
real .dta file — they pin down only the data-shaping logic. Per-step
simulation primitives (calc_tes etc.) are not tested here because they
read forests out of `adsim.config.forests`, which is by design empty
until a loader runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adsim.simulation_steps import (
    extract_ranks,
    find_optimal_ads,
    prepare_data,
)


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

def test_prepare_data_remaps_base_ad_to_zero():
    data = pd.DataFrame({"advertiser_rank": [1, 50, 50, 7]})
    prepare_data(data, base_ad=50, max_ad=100)
    # Rank 50 (the base ad) should now be 0; everything else untouched.
    assert data["advertiser_rank"].tolist() == [1, 0, 0, 7]


def test_prepare_data_collapses_fringe_ranks():
    # Mapping: rank<=max_adv_rank (100) untouched; in (100, 200] -> 101;
    # >200 -> 201. Boundaries: 100 stays, 200 collapses to 101.
    data = pd.DataFrame({"advertiser_rank": [50, 99, 100, 101, 150, 200, 201, 999]})
    prepare_data(data, base_ad=50, max_ad=100)
    assert data["advertiser_rank"].tolist() == [0, 99, 100, 101, 101, 101, 201, 201]


def test_prepare_data_does_not_remap_when_no_base_ad_present():
    data = pd.DataFrame({"advertiser_rank": [1, 2, 3]})
    prepare_data(data, base_ad=50, max_ad=100)
    assert data["advertiser_rank"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# extract_ranks
# ---------------------------------------------------------------------------

def test_extract_ranks_returns_sorted_unique_ranks():
    data = pd.DataFrame({"advertiser_rank": [3, 1, 2, 1, 3, 2, 0]})
    assert extract_ranks(data) == [0, 1, 2, 3]


def test_extract_ranks_handles_single_rank():
    data = pd.DataFrame({"advertiser_rank": [5, 5, 5]})
    assert extract_ranks(data) == [5]


def test_extract_ranks_empty_dataframe():
    data = pd.DataFrame({"advertiser_rank": []})
    assert extract_ranks(data) == []


# ---------------------------------------------------------------------------
# find_optimal_ads
# ---------------------------------------------------------------------------

def _make_row(*, ads_on_page: int, y_values: dict[int, float], rev_values: dict[int, float]) -> pd.Series:
    """Build a Series with the y_<rank>, rev_<rank>, ads_on_page indices that
    find_optimal_ads() expects.

    Built via a heterogeneous DataFrame so the extracted row has dtype
    `object`, matching what production code sees — read_stata yields a
    DataFrame with at least one non-numeric column (e.g. publisher_subject),
    which forces row-wise iteration to preserve `int` for `ads_on_page`.
    """
    cols = {f"y_{r}": [v] for r, v in y_values.items()}
    cols.update({f"rev_{r}": [v] for r, v in rev_values.items()})
    cols["ads_on_page"] = [ads_on_page]
    cols["_dtype_anchor"] = ["x"]  # forces row dtype to object
    df = pd.DataFrame(cols)
    return df.iloc[0]


def test_find_optimal_ads_ctr_picks_highest_y():
    row = _make_row(
        ads_on_page=2,
        y_values={1: 0.10, 2: 0.30, 3: 0.20},
        rev_values={1: 0.40, 2: 0.10, 3: 0.50},
    )
    chosen, ys, revs = find_optimal_ads(row, criteria="CTR")
    # Top-2 by y: rank 2 (0.30), then rank 3 (0.20).
    assert chosen == [2, 3]
    # ys returned are sorted-descending y_values for the top-l ads.
    assert list(ys) == pytest.approx([0.30, 0.20])
    # revs are sorted-descending rev_values for the top-l ads (NOT aligned with `chosen`).
    # The function returns top-l of the rev sort, regardless of CTR pick. Doc-as-is.
    assert list(revs) == pytest.approx([0.50, 0.40])


def test_find_optimal_ads_revenue_picks_highest_rev():
    row = _make_row(
        ads_on_page=2,
        y_values={1: 0.10, 2: 0.30, 3: 0.20},
        rev_values={1: 0.40, 2: 0.10, 3: 0.50},
    )
    chosen, _, _ = find_optimal_ads(row, criteria="revenue")
    # Top-2 by rev: rank 3 (0.50), then rank 1 (0.40).
    assert chosen == [3, 1]


def test_find_optimal_ads_caps_at_max_ads_per_page():
    # ads_on_page=20 but max_ads_per_page is 15 in adsim.config.
    from adsim import config
    row = _make_row(
        ads_on_page=20,
        y_values={r: 1.0 - r * 0.01 for r in range(1, 21)},
        rev_values={r: 0.5 for r in range(1, 21)},
    )
    chosen, ys, _ = find_optimal_ads(row, criteria="CTR")
    assert len(chosen) == config.max_ads_per_page == 15


def test_find_optimal_ads_respects_smaller_ads_on_page():
    row = _make_row(
        ads_on_page=1,
        y_values={1: 0.10, 2: 0.30, 3: 0.20},
        rev_values={1: 0.40, 2: 0.10, 3: 0.50},
    )
    chosen, ys, _ = find_optimal_ads(row, criteria="CTR")
    assert chosen == [2]
    assert list(ys) == pytest.approx([0.30])
