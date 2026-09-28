"""Контракт сабмита и воспроизводимая проверка point-in-time источников."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.configuration import load_config
from src.data_loader import discover_file, load_macro
from src.nlp_features import JSON_TO_NEWS_COLUMNS


@pytest.fixture(scope="module")
def submission() -> pd.DataFrame:
    return pd.read_csv(Path("submission.csv"))


def test_submission_shape(submission: pd.DataFrame) -> None:
    assert submission.shape == (12_564, 3)
    assert submission.columns.tolist() == ["mo", "period", "pred"]


def test_no_missing_or_inf(submission: pd.DataFrame) -> None:
    assert submission.notna().all().all()
    assert np.isfinite(pd.to_numeric(submission["pred"], errors="coerce")).all()


def test_positive_bounds(submission: pd.DataFrame) -> None:
    assert submission["pred"].gt(0).all()


def test_geography_coverage(submission: pd.DataFrame) -> None:
    assert submission["mo"].nunique() == 2_094
    assert submission.groupby("mo")["period"].nunique().eq(6).all()
    assert not submission.duplicated(["mo", "period"]).any()


def test_point_in_time_leakage() -> None:
    config = load_config(Path("configs/config.yaml"))
    assert config.data.macro_lag_months >= 1
    assert config.data.nlp.lag_months >= 1
    assert config.features.news_lag_months >= 1
    panel = pd.read_parquet(config.paths.supervised)
    macros = [str(column) for column in panel if str(column).startswith("macro_")]
    assert macros and all(column.endswith("_lag_1") for column in macros)
    dates = pd.to_datetime(panel["period"])
    assert dates.min() >= pd.Timestamp("2023-02-01")
    source = config.data.macro_sources[0]
    path = discover_file(config.data.directory, source.pattern)
    assert path is not None
    current_macro = load_macro(config.data.directory, config.data).set_index("period")
    lagged_column = macros[0]
    expected = current_macro[lagged_column].reindex(
        dates.drop_duplicates().sort_values()
    ).to_numpy(dtype=float)
    actual = panel.drop_duplicates("period").sort_values("period")[lagged_column].to_numpy(dtype=float)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    news = pd.read_parquet(Path("data/processed/news_features.parquet"))
    news = news.set_index("period")
    assert set(JSON_TO_NEWS_COLUMNS.values()).issubset(panel.columns)
    for original, renamed in JSON_TO_NEWS_COLUMNS.items():
        expected_news = news[original].reindex(
            dates.drop_duplicates().sort_values()
        ).to_numpy(dtype=float)
        actual_news = panel.drop_duplicates("period").sort_values("period")[renamed].to_numpy(dtype=float)
        np.testing.assert_allclose(actual_news, expected_news, equal_nan=True)
    assert config.data.nlp.publication_date_is_availability


def test_changed_candidate_synchronizes_submission(tmp_path: Path) -> None:
    import hashlib

    from src.submission import SubmissionValidator

    root = tmp_path / "submission.csv"
    candidate = tmp_path / "reports" / "submission.csv"
    periods = pd.date_range("2024-07-01", periods=2, freq="MS")
    initial = pd.DataFrame({"mo": ["a", "a"], "period": periods, "pred_catboost": [10.0, 20.0]})
    validator = SubmissionValidator(candidate, root)
    validator.build(initial, expected_rows=2, expected_entities=1, periods=periods)
    digest = hashlib.md5(root.read_bytes()).hexdigest()

    changed = initial.copy()
    changed["pred_catboost"] = [11.0, 21.0]
    output = validator.build(changed, expected_rows=2, expected_entities=1, periods=periods)
    assert output["pred"].tolist() == [11.0, 21.0]
    assert pd.read_csv(candidate)["pred"].tolist() == [11.0, 21.0]
    assert hashlib.md5(root.read_bytes()).hexdigest() != digest
    assert root.read_bytes() == candidate.read_bytes()


def test_changed_candidate_updates_same_path(tmp_path: Path) -> None:
    from src.submission import SubmissionValidator

    root = tmp_path / "submission.csv"
    periods = pd.date_range("2024-07-01", periods=1, freq="MS")
    validator = SubmissionValidator(root, root)
    frame = pd.DataFrame({"mo": ["a"], "period": periods, "pred_catboost": [10.0]})
    validator.build(frame, expected_rows=1, expected_entities=1, periods=periods)
    frame["pred_catboost"] = [11.0]
    validator.build(frame, expected_rows=1, expected_entities=1, periods=periods)
    assert pd.read_csv(root)["pred"].tolist() == [11.0]


def test_explicit_prophet_selection_writes_identical_outputs(tmp_path: Path) -> None:
    from src.submission import SubmissionValidator

    periods = pd.date_range("2024-07-01", periods=2, freq="MS")
    frame = pd.DataFrame({
        "mo": ["a", "a"], "period": periods,
        "pred_catboost": [10.0, 11.0], "pred_prophet": [20.0, 21.0],
    })
    root = tmp_path / "submission.csv"
    candidate = tmp_path / "reports" / "submission.csv"
    result = SubmissionValidator(candidate, root).build(
        frame, expected_rows=2, expected_entities=1, periods=periods,
        prediction_column="pred_prophet",
    )
    assert result.pred.tolist() == [20.0, 21.0]
    assert root.read_bytes() == candidate.read_bytes()


def test_partial_oof_uses_fixed_roster_and_only_past_history(tmp_path: Path) -> None:
    from src.submission import SubmissionValidator

    root = tmp_path / "submission.csv"
    periods = pd.date_range("2024-07-01", periods=2, freq="MS", tz="UTC")
    reference = pd.DataFrame({"mo": ["a", "a", "b", "b"], "period": list(periods) * 2, "pred": [999.0] * 4})
    reference.to_csv(root, index=False)
    frozen = root.read_bytes()
    predictions = pd.DataFrame({"mo": ["a"], "period": periods[1:], "pred_catboost": [30.0]})
    history = pd.DataFrame({
        "mo": ["a"] * 3 + ["b"] * 3,
        "period": list(pd.date_range("2024-06-01", periods=3, freq="MS", tz="UTC")) * 2,
        "y": [10.0, 20.0, 9000.0, 40.0, 50.0, 9000.0],
    })

    result = SubmissionValidator(tmp_path / "candidate.csv", root).build(
        predictions, expected_rows=4, expected_entities=2,
        periods=periods.tz_localize(None), history=history,
    )

    assert result["mo"].tolist() == ["a", "a", "b", "b"]
    assert result["pred"].tolist() == [10.0, 30.0, 40.0, 50.0]
    assert root.read_bytes() != frozen
    assert root.read_bytes() == (tmp_path / "candidate.csv").read_bytes()


def test_explicit_roster_supports_cold_start_without_future_history(tmp_path: Path) -> None:
    from src.submission import SubmissionValidator

    periods = pd.date_range("2024-07-01", periods=1, freq="MS")
    predictions = pd.DataFrame({"mo": ["a"], "period": periods, "pred_catboost": [30.0]})
    history = pd.DataFrame({
        "mo": ["a", "c", "b"],
        "period": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-07-01"]),
        "y": [10.0, 20.0, 9000.0],
    })

    result = SubmissionValidator(tmp_path / "candidate.csv", tmp_path / "root.csv").build(
        predictions, expected_rows=2, expected_entities=2, periods=periods,
        entity_universe=["a", "b"], history=history,
    )

    assert result["pred"].tolist() == [30.0, 15.0]


def test_submission_rejects_entities_outside_reference_roster(tmp_path: Path) -> None:
    from src.submission import SubmissionValidator

    periods = pd.date_range("2024-07-01", periods=1, freq="MS")
    predictions = pd.DataFrame({"mo": ["unknown"], "period": periods, "pred_catboost": [30.0]})

    with pytest.raises(ValueError, match="вне списка submission"):
        SubmissionValidator(tmp_path / "candidate.csv", tmp_path / "root.csv").build(
            predictions, expected_rows=1, expected_entities=1, periods=periods,
            entity_universe=["a"],
        )


def test_submission_keeps_count_check_without_reference_roster(tmp_path: Path) -> None:
    from src.submission import SubmissionValidator

    predictions = pd.DataFrame({"mo": ["a"], "period": ["2024-07-01"], "pred_catboost": [30.0]})

    with pytest.raises(ValueError, match="Ожидалось 2 уникальных МО"):
        SubmissionValidator(tmp_path / "candidate.csv", tmp_path / "root.csv").build(
            predictions, expected_rows=12, expected_entities=2,
        )


def test_submission_fallback_rejects_history_available_only_in_future(tmp_path: Path) -> None:
    from src.submission import SubmissionValidator

    periods = pd.date_range("2024-07-01", periods=1, freq="MS")
    predictions = pd.DataFrame({"mo": ["a"], "period": periods, "pred_catboost": [30.0]})
    history = pd.DataFrame({"mo": ["b"], "period": periods, "y": [9000.0]})

    with pytest.raises(ValueError, match="Нет истории до"):
        SubmissionValidator(tmp_path / "candidate.csv", tmp_path / "root.csv").build(
            predictions, expected_rows=2, expected_entities=2, periods=periods,
            entity_universe=["a", "b"], history=history,
        )
