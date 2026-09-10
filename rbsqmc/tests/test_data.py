"""Tests for rbsqmc.src.data.data — get_results and get_training_data.

All tests use historical data only (2020-01-01 to 2022-12-31) where scores
are known and stable.  The parquet cache is used (download=False) so no
network access is required.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from rbsqmc.src.data.data import get_results, get_training_data

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HIST_START = "2020-01-01"
HIST_END = "2022-12-31"
TRAIN_START = "2020-01-01"
TEST_START = "2021-06-01"
PRED_START = "2022-01-01"
PRED_END = "2022-12-31"
MAX_GOALS = 8


@pytest.fixture(scope="module")
def results_output():
    """Run get_results once for the module."""
    return get_results(
        start_date=HIST_START, end_date=HIST_END, max_goals=MAX_GOALS, download=False
    )


@pytest.fixture(scope="module")
def training_output():
    """Run get_training_data once for the module."""
    return get_training_data(
        train_start_date=TRAIN_START,
        test_start_date=TEST_START,
        prediction_start_date=PRED_START,
        prediction_end_date=PRED_END,
        max_goals=MAX_GOALS,
        download=False,
    )


# ---------------------------------------------------------------------------
# get_results tests
# ---------------------------------------------------------------------------


class TestGetResults:
    """Validate the three outputs of get_results on historical data."""

    def test_returns_three_outputs(self, results_output):
        df, results, team_id_to_name = results_output
        assert isinstance(df, pd.DataFrame)
        assert hasattr(results, "date")
        assert hasattr(results, "timestamp")
        assert hasattr(results, "timestamp_prev")
        assert hasattr(results, "matches")
        assert hasattr(results, "match_mask")
        assert isinstance(team_id_to_name, dict)

    def test_df_columns(self, results_output):
        df, _, _ = results_output
        assert list(df.columns) == ["date", "timestamp", "timestamp_prev", "matches"]

    def test_df_not_empty(self, results_output):
        df, _, _ = results_output
        assert len(df) > 0

    def test_df_dates_within_range(self, results_output):
        df, _, _ = results_output
        assert df["date"].min() >= pd.Timestamp(HIST_START)
        assert df["date"].max() <= pd.Timestamp(HIST_END)

    def test_df_dates_sorted(self, results_output):
        df, _, _ = results_output
        assert df["date"].is_monotonic_increasing

    def test_no_negative_scores_in_historical_data(self, results_output):
        """Historical matches must have real scores (no -1 sentinels)."""
        df, _, _ = results_output
        for match_list in df["matches"]:
            for m in match_list:
                assert m["home_score"] >= 0, f"Negative home_score: {m}"
                assert m["away_score"] >= 0, f"Negative away_score: {m}"

    def test_scores_respect_max_goals(self, results_output):
        df, _, _ = results_output
        for match_list in df["matches"]:
            for m in match_list:
                assert m["home_score"] <= MAX_GOALS
                assert m["away_score"] <= MAX_GOALS

    def test_jax_array_shapes_consistent(self, results_output):
        """All JAX arrays should have matching T dimension."""
        df, results, _ = results_output
        T = len(df)
        assert results.date.shape == (T,)
        assert results.timestamp.shape == (T,)
        assert results.timestamp_prev.shape == (T,)
        assert results.matches.home_score.shape[0] == T
        assert results.matches.away_score.shape[0] == T
        assert results.matches.home_id.shape[0] == T
        assert results.matches.away_id.shape[0] == T
        assert results.match_mask.shape[0] == T

    def test_match_mask_is_boolean(self, results_output):
        _, results, _ = results_output
        assert results.match_mask.dtype == jnp.bool_

    def test_match_mask_matches_scores(self, results_output):
        """Where match_mask is True, scores should be valid (>= 0)."""
        _, results, _ = results_output
        mask = np.asarray(results.match_mask)
        home = np.asarray(results.matches.home_score)
        away = np.asarray(results.matches.away_score)
        assert (home[mask] >= 0).all()
        assert (away[mask] >= 0).all()

    def test_timestamp_is_non_negative(self, results_output):
        """Timestamps are days since start_date, so must be >= 0."""
        _, results, _ = results_output
        assert (np.asarray(results.timestamp) >= 0).all()

    def test_timestamp_prev_is_previous_timestamp(self, results_output):
        """timestamp_prev[t] should equal timestamp[t-1] for t >= 1."""
        _, results, _ = results_output
        ts = np.asarray(results.timestamp)
        ts_prev = np.asarray(results.timestamp_prev)
        assert ts_prev[0] == 0  # first element has no predecessor
        np.testing.assert_array_equal(ts_prev[1:], ts[:-1])

    def test_team_id_to_name_is_bijective(self, results_output):
        _, _, team_id_to_name = results_output
        ids = list(team_id_to_name.keys())
        names = list(team_id_to_name.values())
        assert len(ids) == len(set(ids)), "Duplicate team IDs"
        assert len(names) == len(set(names)), "Duplicate team names"

    def test_team_ids_are_zero_indexed(self, results_output):
        _, _, team_id_to_name = results_output
        ids = sorted(team_id_to_name.keys())
        assert ids == list(range(len(ids)))

    def test_team_names_in_matches_exist_in_mapping(self, results_output):
        """Every team name in the DataFrame matches must appear in the mapping."""
        df, _, team_id_to_name = results_output
        valid_names = set(team_id_to_name.values())
        for match_list in df["matches"]:
            for m in match_list:
                assert m["home_team"] in valid_names
                assert m["away_team"] in valid_names

    def test_no_friendly_matches_by_default(self, results_output):
        """include_friendly defaults to False, so no Friendly tournament."""
        df, _, _ = results_output
        for match_list in df["matches"]:
            for m in match_list:
                assert "friendly" not in m["tournament"].lower()

    def test_no_duplicate_teams_per_day(self, results_output):
        """Each team should appear at most once per date."""
        df, _, _ = results_output
        for _, row in df.iterrows():
            teams = []
            for m in row["matches"]:
                teams.append(m["home_team"])
                teams.append(m["away_team"])
            assert len(teams) == len(set(teams)), (
                f"Duplicate team on {row['date']}: {teams}"
            )


# ---------------------------------------------------------------------------
# get_training_data tests
# ---------------------------------------------------------------------------


class TestGetTrainingData:
    """Validate the three-way split returned by get_training_data."""

    def test_returns_three_splits(self, training_output):
        (train_df, test_df, pred_df), (train_res, test_res, pred_res), mapping = (
            training_output
        )
        for split_df in (train_df, test_df, pred_df):
            assert isinstance(split_df, pd.DataFrame)
            assert len(split_df) > 0

    def test_split_date_ranges_do_not_overlap(self, training_output):
        (train_df, test_df, pred_df), _, _ = training_output
        assert train_df["date"].max() < pd.Timestamp(TEST_START)
        assert test_df["date"].min() >= pd.Timestamp(TEST_START)
        assert test_df["date"].max() < pd.Timestamp(PRED_START)
        assert pred_df["date"].min() >= pd.Timestamp(PRED_START)
        assert pred_df["date"].max() <= pd.Timestamp(PRED_END)

    def test_split_dates_sorted(self, training_output):
        for split_df in training_output[0]:
            assert split_df["date"].is_monotonic_increasing

    def test_no_negative_scores_in_all_splits(self, training_output):
        """Historical data must have real scores in every split."""
        for split_df in training_output[0]:
            for match_list in split_df["matches"]:
                for m in match_list:
                    assert m["home_score"] >= 0
                    assert m["away_score"] >= 0

    def test_scores_respect_max_goals_in_all_splits(self, training_output):
        for split_df in training_output[0]:
            for match_list in split_df["matches"]:
                for m in match_list:
                    assert m["home_score"] <= MAX_GOALS
                    assert m["away_score"] <= MAX_GOALS

    def test_jax_shapes_consistent_per_split(self, training_output):
        _, (train_res, test_res, pred_res), _ = training_output
        for res in (train_res, test_res, pred_res):
            T = res.date.shape[0]
            assert res.timestamp.shape == (T,)
            assert res.timestamp_prev.shape == (T,)
            assert res.matches.home_score.shape[0] == T
            assert res.match_mask.shape[0] == T

    def test_shared_team_mapping(self, training_output):
        """All splits share the same team_id_to_name mapping."""
        _, _, team_id_to_name = training_output
        assert isinstance(team_id_to_name, dict)
        assert len(team_id_to_name) > 0

    def test_team_ids_in_results_are_valid(self, training_output):
        """Team IDs in JAX arrays must be valid indices into the mapping."""
        _, (train_res, test_res, pred_res), team_id_to_name = training_output
        n_teams = len(team_id_to_name)
        for res in (train_res, test_res, pred_res):
            mask = np.asarray(res.match_mask)
            home_ids = np.asarray(res.matches.home_id)[mask]
            away_ids = np.asarray(res.matches.away_id)[mask]
            assert (home_ids >= 0).all() and (home_ids < n_teams).all()
            assert (away_ids >= 0).all() and (away_ids < n_teams).all()

    def test_timestamp_prev_chains_within_split(self, training_output):
        """timestamp_prev[t] == timestamp[t-1] within each split."""
        _, (train_res, test_res, pred_res), _ = training_output
        for res in (train_res, test_res, pred_res):
            ts = np.asarray(res.timestamp)
            ts_prev = np.asarray(res.timestamp_prev)
            assert ts_prev[0] == 0
            np.testing.assert_array_equal(ts_prev[1:], ts[:-1])

    def test_total_dates_cover_full_range(self, training_output):
        """The union of split dates should cover the full historical range."""
        (train_df, test_df, pred_df), _, _ = training_output
        all_dates = pd.concat(
            [train_df["date"], test_df["date"], pred_df["date"]]
        ).sort_values()
        assert all_dates.min() >= pd.Timestamp(TRAIN_START)
        assert all_dates.max() <= pd.Timestamp(PRED_END)

    def test_no_duplicate_teams_per_day_in_splits(self, training_output):
        for split_df in training_output[0]:
            for _, row in split_df.iterrows():
                teams = []
                for m in row["matches"]:
                    teams.append(m["home_team"])
                    teams.append(m["away_team"])
                assert len(teams) == len(set(teams)), (
                    f"Duplicate team on {row['date']}"
                )
