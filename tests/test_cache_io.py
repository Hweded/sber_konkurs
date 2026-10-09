from pathlib import Path

import pandas as pd
import pytest

from src.cache_io import CACHE_ERROR, load_cached_predictions


def _model_frame(model: str, *, with_target: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "mo": ["a", "b"],
            "period": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "horizon": [1, 1],
            "fold": [1, 1],
            f"pred_{model}": [10.0, 20.0],
        }
    )
    if with_target:
        frame["target"] = [11.0, 19.0]
    return frame


def test_loads_canonical_csv_and_normalises_horizon(tmp_path: Path) -> None:
    oof = _model_frame("catboost", with_target=True)
    for model in ("prophet", "chronos"):
        oof[f"pred_{model}"] = [10.0, 20.0]
    test = oof.drop(columns="target").copy()
    oof.to_csv(tmp_path / "oof_predictions.csv", index=False)
    test.to_csv(tmp_path / "test_predictions.csv", index=False)

    loaded_oof, loaded_test = load_cached_predictions(tmp_path)

    assert loaded_oof["lead_months"].tolist() == [1, 1]
    assert loaded_test["lead_months"].tolist() == [1, 1]
    assert {"pred_catboost", "pred_prophet", "pred_chronos"}.issubset(loaded_oof)


def test_assembles_oof_and_test_from_per_model_files(tmp_path: Path) -> None:
    for model in ("prophet", "catboost", "chronos"):
        oof = _model_frame(model, with_target=True)
        test = _model_frame(model, with_target=False)
        test["period"] = pd.to_datetime(["2024-07-01", "2024-07-01"])
        pd.concat([oof, test], ignore_index=True).to_csv(tmp_path / f"{model}_preds.csv", index=False)

    loaded_oof, loaded_test = load_cached_predictions(tmp_path)
    assert len(loaded_oof) == 2
    assert len(loaded_test) == 2
    assert loaded_oof["target"].notna().all()
    assert loaded_test["target"].isna().all()


def test_missing_cache_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Кэш базовых прогнозов не найден") as error:
        load_cached_predictions(tmp_path)
    assert str(error.value).startswith(CACHE_ERROR)
