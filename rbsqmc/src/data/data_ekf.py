"""One frozen, chronologically ordered dataset for both comparison methods."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from rbsqmc.src.data.data import filter_teams, generate_team_id_mapping
from rbsqmc.src.model.ekf.model import MatchInputs
from rbsqmc.src.utils.helpers import resolve_teams
from rbsqmc.src.utils.type import FootballResults, Matches


@dataclass
class Dataset:
    frame: pd.DataFrame
    teams: dict
    train_count: int
    test_count: int
    inputs: MatchInputs
    sqmc: FootballResults
    metadata: dict


def load_dataset(path, cfg, smoke=False):
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["source_row"] = np.arange(len(raw))
    # Stable ordering preserves source order within a date; both methods see
    # precisely the same sequence, even if a team appears twice that day.
    raw = raw.sort_values("date", kind="stable")
    options = dict(start_datetime=pd.Timestamp(cfg["training_start_date"]),
                   end_datetime=None, include_friendly=cfg["include_friendly"],
                   teams_only=resolve_teams(cfg))
    eligible = filter_teams(raw.copy(), max_goals=float("inf"), **options)
    frame = filter_teams(raw.copy(), max_goals=cfg["max_goals"], **options)
    _, teams = generate_team_id_mapping(frame)
    split = pd.Timestamp(cfg["test_start_date"])
    prediction = pd.Timestamp(cfg["prediction_start_date"])
    train = frame.loc[frame.date < split]
    test = frame.loc[(frame.date >= split) & (frame.date < prediction)]
    pred = frame.loc[frame.date >= prediction]
    if smoke:
        train, test, pred = train.tail(32), test.tail(8), pred.head(4)
    if any(part.empty for part in (train, test, pred)):
        raise ValueError("Comparison requires nonempty train, test, and prediction splits")
    frame = pd.concat([train, test, pred], ignore_index=True)
    if (frame.home_id == frame.away_id).any():
        raise ValueError("A fixture must contain two distinct teams")
    scores = frame[["home_score", "away_score"]].to_numpy()
    if (scores[:len(train) + len(test)] < 0).any():
        raise ValueError("Training and test splits must contain known results")
    times = (frame.date - pd.Timestamp(cfg["training_start_date"])).dt.days.to_numpy()
    previous = np.zeros((len(frame), 2))
    last = np.zeros(len(teams))
    for i, row in frame.iterrows():
        pair = [row.home_id, row.away_id]
        previous[i] = last[pair]
        last[pair] = times[i]
    home, away = jnp.asarray(frame.home_id), jnp.asarray(frame.away_id)
    inputs = MatchInputs(home, away, jnp.asarray(scores, dtype=float),
                         jnp.asarray(frame.friendly), jnp.asarray(times, dtype=float),
                         jnp.asarray(previous))
    sqmc = FootballResults(
        date=jnp.asarray(frame.date.to_numpy(dtype="datetime64[D]").astype(int)),
        timestamp=jnp.asarray(times), timestamp_prev=jnp.asarray(np.r_[0, times[:-1]]),
        matches=Matches(home[:, None], away[:, None],
                        jnp.asarray(scores[:, 0, None]), jnp.asarray(scores[:, 1, None])),
        match_mask=jnp.ones((len(frame), 1), dtype=bool))
    excluded = eligible.loc[~eligible.source_row.isin(filter_teams(raw.copy(), max_goals=cfg["max_goals"], **options).source_row)]
    metadata = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_rows=len(raw), max_goals_exclusions=excluded.source_row.tolist(),
                    train_count=len(train), test_count=len(test), prediction_count=len(pred),
                    worldcup_count=int(((frame.tournament == "FIFA World Cup") &
                                       (frame.date.dt.year == 2026)).sum()),
                    chronological_order="date, then original source row",
                    score_grid="0..max_goals, normalized within grid; raw mass reported",
                    smoke=smoke)
    return Dataset(frame, teams, len(train), len(test), inputs, sqmc, metadata)
