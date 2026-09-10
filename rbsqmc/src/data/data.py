import os
import json
import pandas as pd
from rbsqmc.src.utils.type import Matches, FootballResults
import numpy as np
import jax.numpy as jnp
import jax
from datetime import datetime

RAW_URL="https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")  # rbpf/data/
PARQUET_PATH = os.path.join(_DATA_DIR, "results.parquet")
WORLDCUP_2026_PATH = os.path.join(_DATA_DIR, "worldcup2026.json")
ACITVE_TEAMS_PATH = os.path.join(_DATA_DIR, "active_teams.json")
TEAMS_SMALL_PATH = os.path.join(_DATA_DIR, "teams_small.json")

with open(WORLDCUP_2026_PATH) as f:
    WORLDCUP_2026_TEAMS: set[str] = set(json.load(f))

with open(ACITVE_TEAMS_PATH) as f:
    ACTIVE_TEAMS: set[str] = set(json.load(f)['teams'])

with open(TEAMS_SMALL_PATH) as f:
    TEAMS_SMALL: set[str] = set(json.load(f)['teams'])

def _drop_duplicate_teams_per_day(data: pd.DataFrame) -> pd.DataFrame:
    """Assume that teams do not play more than once per day, otherwise it causes an issue with the propagation."""
    home_appearances = data[["date", "home_team"]].rename(columns={"home_team": "team"})
    away_appearances = data[["date", "away_team"]].rename(columns={"away_team": "team"})
    team_appearances = pd.concat([home_appearances, away_appearances], ignore_index=True)
    duplicated = team_appearances[team_appearances.duplicated(subset=["date", "team"], keep=False)]
    if not duplicated.empty:
        print("Warning: Dropping duplicate team appearances on the same day:")
        print(duplicated.sort_values(["date", "team"]).to_string(index=False))
    return data[~data.index.isin(duplicated.index)].reset_index(drop=True)

def filter_teams(
    data: pd.DataFrame,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp | None,
    max_goals: int,
    include_friendly: bool,
    teams_only: set[str] | None,
) -> pd.DataFrame:
    # filter data by date range
    data = data[
        (data["date"] >= start_datetime)
        & (data["date"] <= end_datetime if end_datetime is not None else True)
    ]
    # filter data by max_goals
    data = data[
        (data["home_score"].isna() | (data["home_score"] <= max_goals))
        & (data["away_score"].isna() | (data["away_score"] <= max_goals))
    ].copy()
    # filter out friendly matches if include_friendly is False
    data["friendly"] = data["tournament"].str.contains(
        "Friendly", case=False, na=False
    )
    # filter by teams_only if provided
    if teams_only is not None:
        data = data[
            data["home_team"].isin(teams_only) & data["away_team"].isin(teams_only)
        ]
    if not include_friendly:
        data = data[~data["friendly"]]
    data = data.reset_index(drop=True)
    # Fix dates
    data.loc[
        (data["home_team"] == "Morocco")
        & (data["away_team"] == "Senegal")
        & (data["date"] == pd.Timestamp("2026-01-18")),
        ["home_score", "away_score"],
    ] = [0, 1]

    # future games have nan scores - fill with -1
    data[["home_score", "away_score"]] = (
        data[["home_score", "away_score"]].fillna(-1).astype(int)
    )
    # Match-level filters process repeated appearances sequentially at dt=0.
    # Removing these rows changes the dataset, rather than fixing propagation.
    return data

def generate_team_id_mapping(data: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    # Create mapping of team names to integer IDs
    all_teams = pd.unique(data[["home_team", "away_team"]].values.ravel())
    # Create a stable mapping: team_name -> team_id
    team_name_to_id = {name: i for i, name in enumerate(sorted(all_teams))}
    team_id_to_name = {i: name for i, name in enumerate(sorted(all_teams))}

    # Assign integer IDs back to the dataframe
    data["home_id"] = data["home_team"].map(team_name_to_id)
    data["away_id"] = data["away_team"].map(team_name_to_id)
    return team_name_to_id, team_id_to_name

def generate_results_jax(matches: pd.DataFrame) -> tuple[pd.DataFrame, FootballResults]:
    # group by date
    date_grouped = matches.groupby("date")
    # get number of unique dates and max number of matches per day
    T = date_grouped.ngroups
    M = date_grouped.size().max()

    print(f"Generating JAX arrays for {T} unique dates and up to {M} matches per day.")

    # Use NumPy arrays for accumulation (mutable); converted to JAX at the end.
    # Pad with -1 as a sentinel: never a valid score or team id (team ids are >= 0).
    home_score = np.full((T, M), 0, dtype=np.int32)
    away_score = np.full((T, M), 0, dtype=np.int32)
    home_id = np.full((T, M), 0, dtype=np.int32)
    away_id = np.full((T, M), 0, dtype=np.int32)
    match_mask = np.zeros((T, M), dtype=bool)

    dates = []
    timestamps = []
    timestamps_prev = []

    prev_timestamp = 0
    # loop over each date group and fill in the arrays
    for t, (date, group) in enumerate(date_grouped):
        n = len(group)

        home_score[t, :n] = group["home_score"].to_numpy()
        away_score[t, :n] = group["away_score"].to_numpy()
        home_id[t, :n] = group["home_id"].to_numpy()
        away_id[t, :n] = group["away_id"].to_numpy()

        match_mask[t, :n] = True

        timestamp = group["timestamp"].iloc[0]

        dates.append(date)
        timestamps.append(timestamp)
        timestamps_prev.append(prev_timestamp)
        # update prev_timestamp for the next iteration
        prev_timestamp = timestamp

    matches_jax = Matches(
        home_score=jnp.asarray(home_score),
        away_score=jnp.asarray(away_score),
        home_id=jnp.asarray(home_id),
        away_id=jnp.asarray(away_id),
    )

    # dates are pandas Timestamps -> convert to numeric (days since start_date)
    # via np.datetime64 for JAX compatibility. DataFrame keeps the readable form.
    date_values = jnp.asarray([np.datetime64(d).astype("datetime64[D]").astype("int64")
                               for d in dates])

    results = FootballResults(
        date=date_values, # (T, ). used for reference in the future.
        timestamp=jnp.asarray(timestamps), # (T, )
        timestamp_prev=jnp.asarray(timestamps_prev), # (T, )
        matches=matches_jax, # (T, M)
        match_mask=jnp.asarray(match_mask), # (T, M) boolean mask
    )
    # per-date matches as readable dictionaries (one entry per date).
    # Distinct from `results.matches` (the JAX Matches tensor).
    matches_per_date = [
        [
            {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "tournament": row["tournament"],
            }
            for _, row in group.iterrows()
        ]
        for _, group in date_grouped
    ]

    df = pd.DataFrame(
        {
            "date": dates,
            "timestamp": timestamps,
            "timestamp_prev": timestamps_prev,
            "matches": matches_per_date,   # list of T lists of match dicts
        }
    )
    return df, results

def concat_football_results(*splits: FootballResults) -> FootballResults:
    """Concatenate football result splits along the time axis.

    Each split is a ``FootballResults`` with time-varying arrays shaped ``(T_i, ...)``
    (matches / match_mask are ``(T_i, M_i)`` where ``M_i`` can differ between
    splits). This helper:

    - Pads ``matches`` / ``match_mask`` to a shared ``M`` (max over splits) so the
      tensors can be stacked.
    - Fixes the ``timestamp_prev`` boundary: each split was built with its first
      ``timestamp_prev`` reset to 0, but a continuous forward filter must carry the
      previous split's final timestamp across the boundary. Here we recompute it so
      every row's ``timestamp_prev`` equals the timestamp of the prior row.

    Args:
        *splits: two or more ``FootballResults`` to concatenate in order.

    Returns:
        A single ``FootballResults`` spanning the full time range, with team
        coordinate padding to the maximum per-day match count.
    """
    if len(splits) == 1:
        return splits[0]

    # Common number of matches per day across all splits.
    max_m = max(int(s.matches.home_score.shape[1]) for s in splits)

    def _pad_matches(
        matches: Matches, m: int, mask: jax.Array
    ) -> tuple[Matches, jax.Array]:
        n = matches.home_score.shape[0]
        if n == 0:
            return matches, mask
        pad_width = ((0, 0), (0, max_m - m))
        return (
            Matches(
                home_id=jnp.pad(matches.home_id, pad_width),
                away_id=jnp.pad(matches.away_id, pad_width),
                home_score=jnp.pad(matches.home_score, pad_width),
                away_score=jnp.pad(matches.away_score, pad_width),
            ),
            jnp.pad(mask, pad_width),
        )

    dates, timestamps, masks = [], [], []
    matches_fields = {"home_id": [], "away_id": [], "home_score": [], "away_score": []}

    for s in splits:
        m = s.matches.home_score.shape[1]
        padded_matches, padded_mask = _pad_matches(s.matches, m, s.match_mask)
        dates.append(s.date)
        timestamps.append(s.timestamp)
        masks.append(padded_mask)
        for key in matches_fields:
            matches_fields[key].append(getattr(padded_matches, key))

    date = jnp.concatenate(dates)
    timestamp = jnp.concatenate(timestamps)
    # Fix the timestamp_prev boundary: previous row's timestamp (0 for the
    # very first row of the combined sequence).
    timestamp_prev = jnp.concatenate([jnp.array([0]), timestamp[:-1]])

    return FootballResults(
        date=date,
        timestamp=timestamp,
        timestamp_prev=timestamp_prev,
        matches=Matches(**{k: jnp.concatenate(v) for k, v in matches_fields.items()}),
        match_mask=jnp.concatenate(masks),
    )


def unpack_football_results(
    model_inputs: FootballResults,
) -> FootballResults:
    valid_t, valid_m = np.nonzero(np.asarray(model_inputs.match_mask))
    if valid_t.size == 0:
        raise ValueError("FootballResults contains no valid matches.")

    timestamp = model_inputs.timestamp[valid_t]
    timestamp_prev = jnp.concatenate(
        [
            jnp.asarray([model_inputs.timestamp_prev[0]]),
            timestamp[:-1],
        ]
    )

    return FootballResults(
        date=model_inputs.date[valid_t],
        timestamp=timestamp,
        timestamp_prev=timestamp_prev,
        matches=Matches(
            home_id=model_inputs.matches.home_id[valid_t, valid_m, None],
            away_id=model_inputs.matches.away_id[valid_t, valid_m, None],
            home_score=model_inputs.matches.home_score[valid_t, valid_m, None],
            away_score=model_inputs.matches.away_score[valid_t, valid_m, None],
        ),
        match_mask=jnp.ones((len(valid_t), 1), dtype=bool),
    )

def rbpf_inputs_to_dataframe(model_inputs, team_id_to_name=None):
    to_numpy = lambda x: np.asarray(jax.device_get(x))

    # date = pd.to_datetime(
    #     to_numpy(model_inputs.date),
    #     unit="D",
    #     origin="unix",
    # )
    
    timestamp = to_numpy(model_inputs.timestamp)
    timestamp_prev = to_numpy(model_inputs.timestamp_prev)

    home_id = to_numpy(model_inputs.matches.home_id)[:, 0]
    away_id = to_numpy(model_inputs.matches.away_id)[:, 0]
    home_score = to_numpy(model_inputs.matches.home_score)[:, 0]
    away_score = to_numpy(model_inputs.matches.away_score)[:, 0]

    gamma = to_numpy(model_inputs.gamma)
    gamma_pred = to_numpy(model_inputs.gamma_pred)
    gamma_observed = to_numpy(model_inputs.gamma_observed)[:, 0]
    kalman_gain = to_numpy(model_inputs.kalman_gain)[:, 0]

    df = pd.DataFrame({
        # "date": date,
        "timestamp": timestamp,
        "timestamp_prev": timestamp_prev,
        "dt": timestamp - timestamp_prev,
        "home_id": home_id,
        "away_id": away_id,
        "home_score": home_score,
        "away_score": away_score,

        # The 2x2 covariance immediately before this match.
        "home_variance_before": gamma_observed[:, 0, 0],
        "away_variance_before": gamma_observed[:, 1, 1],
        "home_away_covariance_before": gamma_observed[:, 0, 1],

        # Compact full-matrix diagnostics.
        "gamma_pred_trace": np.trace(gamma_pred, axis1=1, axis2=2),
        "gamma_after_trace": np.trace(gamma, axis1=1, axis2=2),
        "kalman_gain_norm": np.linalg.norm(
            kalman_gain, axis=(1, 2)
        ),
    })

    if team_id_to_name is not None:
        df["home_team"] = df["home_id"].map(team_id_to_name)
        df["away_team"] = df["away_id"].map(team_id_to_name)

    return df

def get_results(
    start_date: str = "1872-11-30",  # date of first game
    end_date: str | None = None,
    max_goals: int = 8,
    include_friendly: bool = False,
    teams_only: set[str] | None = None,
    download: bool = False,
):
    """
    Fetch and process international football results.

    If `download` is True, pulls the latest data from the CSV source over the
    network; otherwise reads from the local parquet cache (no internet needed).
    Both paths apply identical filtering/processing.

    Outputs:
        - results_df: pd.DataFrame with columns ['match_index_id', 'timestamp',
          'home_team_id', 'away_team_id', 'home_score', 'away_score']
        - results: FootballResults named tuple with the same data as jax arrays
        - team_id_to_name: dict mapping team_id to team_name
    """
    if download:
        data = pd.read_csv(RAW_URL, date_format="%Y-%m-%d")
    else:
        data = pd.read_parquet(PARQUET_PATH)

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) if end_date else None
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(by="date", inplace=True)

    data = filter_teams(
        data=data,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        max_goals=max_goals,
        include_friendly=include_friendly,
        teams_only=teams_only,
    )
    team_name_to_id, team_id_to_name = generate_team_id_mapping(data)

    # timestamp is days since start_date (start_date is the reference point)
    data["timestamp"] = (data["date"] - start_datetime).dt.days

    # group by date and convert each matches to jax array
    # then add in timestamp and timestamp_prev columns
    df, football_results = generate_results_jax(data)

    return df, football_results, team_id_to_name

def get_training_data(
    train_start_date: str,  # date of first game
    test_start_date: str,
    prediction_start_date: str,
    prediction_end_date: str | None = None,
    max_goals: int = 8,
    include_friendly: bool = False,
    teams_only: set[str] | None = None,
    download: bool = False,
):
    """
    Train, Test, Prediction data split.

    Outputs:
        - (train_df, test_df, prediction_df): tuple of pd.DataFrames for each split
        - (train_results, test_results, prediction_results): tuple of FootballResults named tuples for each split
        - team_id_to_name: dict mapping team_id to team_name
    """
    if download:
        data = pd.read_csv(RAW_URL, date_format="%Y-%m-%d")
    else:
        data = pd.read_parquet(PARQUET_PATH)
    train_start_datetime, test_start_datetime, prediction_start_datetime = map(
        pd.to_datetime, [train_start_date, test_start_date, prediction_start_date]
    )
    prediction_end_datetime = pd.to_datetime(prediction_end_date) if prediction_end_date else None

    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(by="date", inplace=True)

    data = filter_teams(
        data=data,
        start_datetime=train_start_datetime,
        end_datetime=prediction_end_datetime,
        max_goals=max_goals,
        include_friendly=include_friendly,
        teams_only=teams_only,
    )
    _, team_id_to_name = generate_team_id_mapping(data)
    data["timestamp"] = (data["date"] - train_start_datetime).dt.days

    # Split data into train, test, and prediction sets
    train_data = data[(data["date"] >= train_start_datetime) & (data["date"] < test_start_datetime)]
    test_data = data[(data["date"] >= test_start_datetime) & (data["date"] < prediction_start_datetime)]
    prediction_data = data[(data["date"] >= prediction_start_datetime) & (data["date"] <= prediction_end_datetime if prediction_end_datetime is not None else True)]

    # Generate JAX arrays for each split
    train_df, train_results = generate_results_jax(train_data)
    test_df, test_results = generate_results_jax(test_data)
    prediction_df, prediction_results = generate_results_jax(prediction_data)

    return (train_df, test_df, prediction_df), (train_results, test_results, prediction_results), team_id_to_name

def main():
    df, results_jax, team_id_to_name = get_results(
        start_date="2020-01-01", end_date="2026-12-31", max_goals=8, download=True
    )
    print("DataFrame head:")
    print(df[["date", "timestamp", "timestamp_prev", "matches"]].head(5))
    last = df.iloc[-1]["matches"][-1]
    print("Last match details:")
    print("   Date:       ", df.iloc[-1]["date"])
    print("   Timestamp:  ", df.iloc[-1]["timestamp"])
    print("   Timestamp Prev: ", df.iloc[-1]["timestamp_prev"])
    print("   Home Team:  ", last["home_team"])
    print("   Away Team:  ", last["away_team"])
    print("   Home Score: ", last["home_score"])
    print("   Away Score: ", last["away_score"])
    print("   Tournament: ", last["tournament"])
    # print(df[['date', 'timestamp', 'timestamp_prev', 'matches']].tail(5))
    # # print("\nFootballResults named tuple:")
    # # print(results_jax)
    # print("\nTeam ID to Name mapping:")
    # print(f"ID 1 : {team_id_to_name[1]}")
    # print(f"ID 2 : {team_id_to_name[2]}")

if __name__ == "__main__":
    main()
