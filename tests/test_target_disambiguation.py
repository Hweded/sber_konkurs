from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_config import TargetConfig
from src.data_loader import load_target


def test_ambiguous_target_rows_are_retained_with_slots(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "period": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "value": [10, 20, 11, 21],
            "obs_status": ["A"] * 4,
            "source": ["same"] * 4,
            "category_15": ["Все категории"] * 4,
            "mo": ["Советский муниципальный район"] * 4,
            "freq": ["Месяц"] * 4,
            "decimals": [0] * 4,
            "unit_measure": ["руб."] * 4,
            "unit_mult": [0] * 4,
        }
    )
    frame.to_csv(tmp_path / "target.csv", sep=";", index=False)
    result = load_target(
        tmp_path,
        TargetConfig(pattern="target.csv", duplicate_policy="exclude_ambiguous_mo"),
    )
    assert len(result) == 4
    assert result.mo.nunique() == 2
    assert any("#2" in str(value) for value in result.mo.unique())
    assert result.y.sum() == 62


def test_unique_target_keeps_plain_mo(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "period": ["2023-01-01"],
            "value": [10],
            "obs_status": ["A"],
            "source": ["same"],
            "category_15": ["Все категории"],
            "mo": ["001"],
            "freq": ["Месяц"],
            "decimals": [0],
            "unit_measure": ["руб."],
            "unit_mult": [0],
        }
    )
    frame.to_csv(tmp_path / "target.csv", sep=";", index=False)
    result = load_target(tmp_path, TargetConfig(pattern="target.csv"))
    assert result.mo.tolist() == ["001"]
