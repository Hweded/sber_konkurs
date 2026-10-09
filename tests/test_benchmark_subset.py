"""Контракт генератора бенчмарк-выборки.

Критичны два свойства: якоря разрешаются однозначно (свободный поиск по подстроке
подхватил бы «Казанский район» вместо Казани) и стратификация воспроизводима при
фиксированном сиде.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.create_benchmark_subset import (ANCHORS, build, even_sizes, load_levels,
                                             quartile_labels, resolve_anchors, verify)


def _panel(path, n_mo=400, periods=12, seed=0):
    rng = np.random.default_rng(seed)
    mos = [f"МО-{i:03d} [Данные СберИндекса]" for i in range(n_mo)]
    mos[0] = "городской округ город Казань [Данные СберИндекса]"
    mos[1] = "городской округ город Норильск [Данные СберИндекса]"
    mos[2] = "Суздальский муниципальный район [Данные СберИндекса]"
    mos[3] = "Казанский муниципальный район [Данные СберИндекса]"   # ловушка совпадения
    rows = []
    for i, mo in enumerate(mos):
        level = 15000 + i * 40 + rng.normal(0, 500)
        for p in range(periods):
            rows.append({"period": pd.Timestamp("2024-01-01") + pd.DateOffset(months=p),
                         "mo": mo, "y": level + rng.normal(0, 100)})
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def test_anchor_resolution_is_strict_not_substring_loose():
    mos = ["городской округ город Казань [Данные СберИндекса]",
           "Казанский муниципальный район [Данные СберИндекса]",
           "городской округ город Норильск [Данные СберИндекса]",
           "Суздальский муниципальный район [Данные СберИндекса]"]
    got = resolve_anchors(mos)
    assert got["Казань"] == "городской округ город Казань [Данные СберИндекса]"
    assert "Казанский" not in got["Казань"]


def test_ambiguous_anchor_stops_the_run():
    """Два совпадения должны останавливать генерацию, а не выбирать наугад."""
    mos = ["Суздальский муниципальный район [Данные СберИндекса]",
           "Суздальский сельсовет [Данные СберИндекса]",
           "городской округ город Казань [Данные СберИндекса]",
           "городской округ город Норильск [Данные СберИндекса]"]
    with pytest.raises(SystemExit):
        resolve_anchors(mos)


def test_missing_anchor_stops_the_run():
    with pytest.raises(SystemExit):
        resolve_anchors(["городской округ город Казань [Данные СберИндекса]"])


def test_even_sizes_never_differ_by_more_than_one():
    assert even_sizes(147, 4) == [37, 37, 37, 36]
    assert sum(even_sizes(147, 4)) == 147
    assert even_sizes(148, 4) == [37, 37, 37, 37]
    assert even_sizes(3, 4) == [1, 1, 1, 0]


def test_quartile_labels_fall_back_to_rank_when_levels_are_indistinguishable():
    """Полностью одинаковые уровни не дают 4 бина по значению — обязателен ранговый fallback."""
    levels = pd.Series([250.0] * 160)
    q, method = quartile_labels(levels)
    assert q.nunique() == 4
    assert "rank" in method
    assert q.value_counts().tolist() == [40, 40, 40, 40]


def test_build_produces_a_valid_reproducible_subset(tmp_path):
    panel = tmp_path / "panel.parquet"
    _panel(panel)
    out = tmp_path / "sub.json"
    build(panel, out, n_mo=150, seed=42, target_col=None)
    verify(out, 150)

    data = json.loads(out.read_text(encoding="utf-8"))
    for label, needle in ANCHORS.items():
        hits = [m for m in data["mo_list"] if needle.lower() in m.lower()]
        assert len(hits) == 1, f"якорь {label} должен встречаться ровно один раз, найдено {len(hits)}"
    assert data["n_mo"] == 150

    # тот же сид -> тот же список; другой сид -> другой список
    out2 = tmp_path / "sub2.json"
    build(panel, out2, n_mo=150, seed=42, target_col=None)
    assert json.loads(out2.read_text())["mo_list"] == data["mo_list"]
    out3 = tmp_path / "sub3.json"
    build(panel, out3, n_mo=150, seed=7, target_col=None)
    assert json.loads(out3.read_text())["mo_list"] != data["mo_list"]


def test_build_is_balanced_across_quartiles(tmp_path):
    panel = tmp_path / "panel.parquet"
    _panel(panel, n_mo=600)
    out = tmp_path / "sub.json"
    build(panel, out, n_mo=150, seed=42, target_col=None)
    strata = json.loads(out.read_text(encoding="utf-8"))["stratification"]["quartiles"]
    picked = [s["picked"] for s in strata]
    assert len(picked) == 4
    assert max(picked) - min(picked) <= 1, f"квартили не выровнены: {picked}"
    assert sum(picked) == 147


def test_target_column_is_auto_detected_and_recorded(tmp_path):
    panel = tmp_path / "panel.parquet"
    _panel(panel)
    out = tmp_path / "sub.json"
    build(panel, out, n_mo=150, seed=42, target_col=None)
    assert json.loads(out.read_text(encoding="utf-8"))["target_column"] == "y"


def test_missing_target_column_is_rejected(tmp_path):
    df = pd.DataFrame({"period": [pd.Timestamp("2024-01-01")], "mo": ["X"], "zzz": [1.0]})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(SystemExit):
        load_levels(p, None)


def test_build_refuses_an_impossible_request(tmp_path):
    panel = tmp_path / "panel.parquet"
    _panel(panel, n_mo=50)
    with pytest.raises(SystemExit):
        build(panel, tmp_path / "sub.json", n_mo=150, seed=42, target_col=None)


def test_verify_catches_a_tampered_file(tmp_path):
    panel = tmp_path / "panel.parquet"
    _panel(panel)
    out = tmp_path / "sub.json"
    build(panel, out, n_mo=150, seed=42, target_col=None)
    data = json.loads(out.read_text(encoding="utf-8"))
    data["mo_list"] = data["mo_list"][:-1]            # подмена длины
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        verify(out, 150)