"""Потоковая агрегация, кэш и point-in-time Росстат без скачивания весов."""
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import pytest

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import warnings

from src.data_config import NLPConfig, RosstatSourceConfig
from src.nlp_features import build_news_features, news_coverage
from src.rosstat_loader import (
    discover_rosstat_files,
    infer_rosstat_schema,
    merge_rosstat,
    normalize_rosstat_table,
    read_rosstat_tables,
)


def test_json_source_discovery_ignores_missing_legacy_result(tmp_path: Path) -> None:
    """Отсутствующий result.json не мешает найти актуальные альтернативы."""
    import json

    from src.nlp_features import resolve_json_news_paths

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    for name in ("smartlab.json", "bank offshore.json", "RBK.json"):
        (datasets / name).write_text(json.dumps({"messages": []}), encoding="utf-8")
    config = NLPConfig(telegram_input=Path("datasets/result.json"))

    existing, checked = resolve_json_news_paths(
        config,
        project_root=tmp_path,
        data_directory=Path("datasets"),
    )

    assert {path.name for path in existing} == {"smartlab.json", "bank offshore.json", "RBK.json"}
    assert any(path.name == "result.json" for path in checked)


def test_json_source_discovery_can_be_disabled(tmp_path: Path) -> None:
    """Явный isolated-конфиг не смешивается с файлами каталога проекта."""
    import json

    from src.nlp_features import resolve_json_news_paths

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "RBK.json").write_text(json.dumps({"messages": []}), encoding="utf-8")
    config = NLPConfig(
        telegram_input=Path("datasets/result.json"),
        news_auto_discover=False,
    )

    existing, _ = resolve_json_news_paths(
        config,
        project_root=tmp_path,
        data_directory=Path("datasets"),
    )

    assert existing == ()


def test_news_coverage_returns_first_and_last(tmp_path: Path) -> None:
    path = tmp_path / "news.csv"
    pd.DataFrame({
        "date": ["2010/01/05", "2019/12/14", "1914/09/16", "not-a-date", "2020/06/01"],
        "title": ["a", "b", "c", "d", "e"],
        "text": ["x"] * 5,
        "topic": ["Экономика"] * 5,
        "tags": [""] * 5,
    }).to_csv(path, index=False)
    config = NLPConfig(path=path, cache=tmp_path / "cache.parquet", min_date="1999-01-01")
    coverage = news_coverage(config)
    assert coverage is not None
    first, last = coverage
    assert first == pd.Timestamp("2010-01-05")
    assert last == pd.Timestamp("2020-06-01")


def test_news_coverage_empty_without_valid_dates(tmp_path: Path) -> None:
    path = tmp_path / "news.csv"
    pd.DataFrame({"date": ["garbage", "also-bad"], "title": ["a", "b"]}).to_csv(path, index=False)
    config = NLPConfig(path=path, cache=tmp_path / "cache.parquet", min_date="1999-01-01")
    assert news_coverage(config) is None


def test_streaming_cache_and_shift(tmp_path: Path) -> None:
    path = tmp_path / "news.csv"
    records = []
    for month, count in ((1, 1), (2, 2), (3, 3), (4, 4)):
        records.extend({"date": f"2023/{month:02d}/10", "title": "economic", "text": "body", "topic": "Экономика", "tags": ""} for _ in range(count))
    records.append({"date": "2023/05/10", "title": "sport", "text": "body", "topic": "Спорт", "tags": ""})
    pd.DataFrame(records).to_csv(path, index=False)
    calls: list[int] = []

    def scorer(texts: Sequence[str]) -> NDArray[np.float64]:
        calls.append(len(texts))
        return np.full(len(texts), 0.5)

    config = NLPConfig(path=path, cache=tmp_path / "cache.parquet", chunksize=3, batch_size=2)
    result = build_news_features(config, scorer=scorer).set_index("period")
    assert max(calls) <= 2
    assert result.loc["2023-02-01", "news_volume"] == 1
    assert result.loc["2023-05-01", "news_volume"] == 4
    assert result.loc["2023-05-01", "news_shock_score"] == 2
    assert result.loc["2023-06-01", "news_volume"] == 0
    assert np.isnan(result.loc["2023-06-01", "sentiment_index"])
    assert result.loc["2023-02-01", "sentiment_index"] == 0.5
    count = len(calls)
    cached = build_news_features(config, scorer=scorer).set_index("period")
    assert len(calls) == count
    pd.testing.assert_frame_equal(result, cached)


def test_rosstat_content_schema_and_merge(tmp_path: Path) -> None:
    path = tmp_path / "rosstat.csv"
    pd.DataFrame(
        {
            "period": ["2023-01-01", "2023-01-01"],
            "mo": ["a", "a"],
            "released_at": ["2023-01-31", "2023-03-01"],
            "wage": ["1 000,5", "1 100,5"],
        }
    ).to_csv(path, sep=";", index=False, encoding="cp1251")
    config = RosstatSourceConfig(
        path=path,
        name="wages",
        separator=";",
        encoding="cp1251",
        value_columns=("wage",),
    )
    tables = read_rosstat_tables(path, config)
    schema = infer_rosstat_schema(tables["data"])
    assert schema["period_column"] == "period"
    assert schema["entity_column"] == "mo"
    normalized = normalize_rosstat_table(tables["data"], config)
    assert normalized["rosstat_wages_wage"].tolist() == [1000.5, 1100.5]

    frame = pd.DataFrame({"period": pd.to_datetime(["2023-02-01", "2023-04-01"]), "mo": ["a", "a"], "y": [1.0, 2.0]})
    result = merge_rosstat(frame, config)
    # Для апреля требуется показатель марта; его нет, поэтому future fill
    # из январского показателя запрещён.
    assert result["rosstat_wages_wage"].iloc[0] == 1000.5
    assert pd.isna(result["rosstat_wages_wage"].iloc[1])


def test_rosstat_file_discovery_finds_formats(tmp_path: Path) -> None:
    (tmp_path / "balans_trud_2025.xlsx").write_bytes(b"placeholder")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    found = discover_rosstat_files(tmp_path)
    assert [path.name for path in found] == ["balans_trud_2025.xlsx"]


def test_rosstat_proxy_growth_does_not_fill_missing_observations(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "mo": ["a"] * 5, "y": [10.0, np.nan, 20.0, 30.0, 40.0],
    })
    config = RosstatSourceConfig(path=tmp_path / "missing.csv", name="wages")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = merge_rosstat(frame, config)

    np.testing.assert_allclose(
        result["rosstat_wages_proxy_spending_growth"],
        [np.nan, np.nan, np.nan, np.nan, 0.5], equal_nan=True,
    )


def test_rosstat_required_invalid_source_is_informative(tmp_path: Path) -> None:
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"not an excel workbook")
    config = RosstatSourceConfig(path=path, name="broken", required=True, value_columns=("value",))
    with pytest.raises(ValueError, match="Росстат"):
        merge_rosstat(pd.DataFrame({"period": pd.to_datetime(["2023-01-01"]), "mo": ["a"], "y": [1.0]}), config)


def test_rosstat_future_revision_not_used(tmp_path: Path) -> None:
    path = tmp_path / "wages.csv"
    pd.DataFrame({"period": ["2023-01-01"] * 2, "mo": ["a", "a"], "released_at": ["2023-01-31", "2023-03-01"], "wage": [100, 999]}).to_csv(path, index=False)
    frame = pd.DataFrame({"period": pd.to_datetime(["2023-02-01"]), "mo": ["a"], "y": [1.0]})
    config = RosstatSourceConfig(path=path, name="wages", value_columns=("wage",))
    result = merge_rosstat(frame, config)
    assert result.rosstat_wages_wage.iloc[0] == 100
    assert not any(column.startswith("_") for column in result)


def test_rosstat_late_release_stays_missing(tmp_path: Path) -> None:
    path = tmp_path / "wages.csv"
    pd.DataFrame({"period": ["2023-01-01"], "mo": ["a"], "released_at": ["2023-02-15"], "wage": [100]}).to_csv(path, index=False)
    frame = pd.DataFrame({"period": pd.to_datetime(["2023-02-01"]), "mo": ["a"], "y": [1.0]})
    result = merge_rosstat(frame, RosstatSourceConfig(path=path, name="wages", value_columns=("wage",)))
    assert result.rosstat_wages_wage.isna().all()


def test_telegram_news_extractor_smoke(tmp_path: Path) -> None:
    """NewsFeatureExtractor: парсинг result.json и агрегация."""
    import json

    from src.data_config import NLPConfig
    from src.nlp_features import NewsFeatureExtractor

    # Создаём тестовый result.json
    messages = [
        {
            "id": 1,
            "type": "message",
            "date": "2024-01-15T10:00:00",
            "text": "Инфляция в январе ускорилась до 7.5% по оценкам экспертов банка России. "
                    "Данные указывают на устойчивый рост цен в потребительском секторе.",
        },
        {
            "id": 2,
            "type": "message",
            "date": "2024-01-20T14:30:00",
            "text": [{"text": "Цены на импорт"}, " выросли на 12% из-за санкций."],
        },
        {
            "id": 3,
            "type": "message",
            "date": "2024-02-05T09:00:00",
            "text": "Короткий текст",
        },
        {
            "id": 4,
            "type": "message",
            "date": "2024-02-10T12:00:00",
            "text": "Дефицит бюджета составил 1.7 трлн руб. Налоги не покрывают расходы на ВПК.",
        },
        {
            "id": 5,
            "type": "message",
            "date": "invalid-date",
            "text": "ВВП вырос на 3.6% в 2023 году согласно данным Росстата и прогнозам аналитиков",
        },
    ]
    source = tmp_path / "result.json"
    source.write_text(json.dumps({"messages": messages}), encoding="utf-8")

    cache = tmp_path / "telegram_news.parquet"
    config = NLPConfig(
        telegram_input=source,
        telegram_keywords=("инфляция", "ставка", "цены", "санкции", "дефицит", "налоги", "бюджет", "расходы", "ввп"),
        telegram_min_length=40,
        cache=cache,
        path=tmp_path / "news.csv",
        model_id="cointegrated/rubert-tiny-sentiment-balanced",
        device="cpu",
    )
    extractor = NewsFeatureExtractor(config)
    # Без загрузки модели проверяем парсинг (через _load_data)
    df = extractor._load_data()
    assert len(df) >= 2, f"Ожидалось >=2 сообщений, получено {len(df)}"
    assert "text" in df.columns
    assert "period" in df.columns
    # Сообщение #3 (< 40 символов) должно быть отфильтровано
    assert not any("Короткий" in t for t in df["text"])
    # Сообщение #5 (invalid date) должно быть отфильтровано
    assert len(df) <= 4


def test_json_news_normalization_deduplicates_and_cleans(tmp_path: Path) -> None:
    """Разные схемы, timezone, HTML, URL, Unicode и дубликаты нормализуются."""
    import json

    from src.data_config import NLPConfig
    from src.nlp_features import NewsFeatureExtractor

    first = tmp_path / "first.json"
    first.write_text(
        json.dumps(
            {
                "articles": [
                    {
                        "article_id": "1",
                        "published_at": "2024-01-31T23:30:00Z",
                        "headline": "Инфляция 😀",
                        "description": "<b>Цены выросли</b> подробнее https://secret.example/token",
                        "source": "Agency",
                        "link": "https://EXAMPLE.com/a?token=secret",
                        "category": "Экономика",
                    },
                    {
                        "article_id": "1",
                        "published_at": "2024-02-01T02:30:00+03:00",
                        "headline": "Инфляция 😀",
                        "description": "Цены выросли подробнее",
                        "source": "Agency",
                    },
                    {"published_at": "bad-date", "text": "Инфляция и цены в повреждённой строке"},
                    {"published_at": "2024-02-02", "text": [123, {"text": "ставка банка выросла"}]},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = NLPConfig(
        news_json_paths=(first,),
        telegram_keywords=("инфляция", "ставка", "цены"),
        telegram_min_length=10,
        cache=tmp_path / "cache.parquet",
        path=tmp_path / "unused.csv",
    )

    result = NewsFeatureExtractor(config)._load_data()

    assert len(result) == 2
    assert result["date"].dt.tz is None
    assert result.iloc[0]["period"] == pd.Timestamp("2024-02-01")
    assert "<b>" not in result.iloc[0]["text"]
    assert "https://" not in result.iloc[0]["text"]
    assert result.iloc[0]["url"] == "https://example.com/a"
    assert "😀" in result.iloc[0]["text"]


def test_json_news_features_are_lagged_without_future_leakage(tmp_path: Path, monkeypatch: Any) -> None:
    """Публикации месяца M появляются только в признаках M+1."""
    import json

    from src.data_config import NLPConfig
    from src.nlp_features import NewsFeatureExtractor

    source = tmp_path / "news.json"
    source.write_text(
        json.dumps(
            [
                {"date": "2024-01-10", "text": "Инфляция и цены выросли достаточно сильно"},
                {"date": "2024-02-10", "text": "Ставка банка изменилась достаточно сильно"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = NLPConfig(
        news_json_paths=(source,),
        telegram_keywords=("инфляция", "цены", "ставка", "банк"),
        telegram_min_length=10,
        lag_months=1,
        news_sentiment_enabled=True,
        cache=tmp_path / "cache.parquet",
        path=tmp_path / "unused.csv",
    )
    extractor = NewsFeatureExtractor(config)
    monkeypatch.setattr(extractor, "_get_scorer", lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        extractor,
        "_score_batch",
        lambda texts: np.arange(1, len(texts) + 1, dtype=np.float64),  # type: ignore[attr-defined]
    )

    result = extractor.compute_nlp_features(force=True).set_index("period")

    assert pd.isna(result.loc["2024-01-01", "telegram_sentiment"])
    assert result.loc["2024-02-01", "telegram_sentiment"] == 1.0
    assert result.loc["2024-03-01", "telegram_sentiment"] == 2.0
    assert result.loc["2024-02-01", "telegram_volume"] == np.log1p(1)


def test_enabled_json_sentiment_fails_instead_of_silent_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    import json

    from src.data_config import NLPConfig
    from src.nlp_features import NewsFeatureExtractor

    source = tmp_path / "news.json"
    source.write_text(
        json.dumps([{"date": "2024-01-10", "text": "Инфляция и цены выросли достаточно сильно"}], ensure_ascii=False),
        encoding="utf-8",
    )
    config = NLPConfig(
        news_json_paths=(source,),
        telegram_keywords=("инфляция",),
        telegram_min_length=10,
        news_sentiment_enabled=True,
        cache=tmp_path / "cache.parquet",
        path=tmp_path / "unused.csv",
    )
    extractor = NewsFeatureExtractor(config)

    def fail_model() -> None:
        raise OSError("model unavailable")

    monkeypatch.setattr(extractor, "_get_scorer", fail_model)
    with pytest.raises(RuntimeError, match="sentiment не удалось измерить"):
        extractor.compute_nlp_features(force=True)
    assert not (tmp_path / "news_features.parquet").exists()


def test_json_news_cache_shared_by_relative_and_absolute_paths(tmp_path: Path, monkeypatch: Any) -> None:
    import json

    from src.data_config import NLPConfig
    from src.nlp_features import NewsFeatureExtractor

    source = tmp_path / "news.json"
    source.write_text(
        json.dumps([{"date": "2024-01-10", "text": "Инфляция и цены выросли достаточно сильно"}], ensure_ascii=False),
        encoding="utf-8",
    )
    config = NLPConfig(
        news_json_paths=(source,),
        telegram_keywords=("инфляция",),
        telegram_min_length=10,
        cache=Path("cache.parquet"),
        path=Path("unused.csv"),
    )
    first = NewsFeatureExtractor(config, project_root=tmp_path)
    monkeypatch.setattr(first, "_get_scorer", lambda: True)
    monkeypatch.setattr(first, "_score_batch", lambda texts: np.ones(len(texts)))
    first.compute_nlp_features()

    absolute = config.model_copy(update={"cache": tmp_path / config.cache, "path": tmp_path / config.path})
    second = NewsFeatureExtractor(absolute, project_root=tmp_path)
    monkeypatch.setattr(second, "_get_scorer", lambda: pytest.fail("cache miss"))
    cached = second.compute_nlp_features()
    assert cached["telegram_sentiment"].notna().sum() == 1


def test_json_news_corrupt_and_empty_inputs_degrade_gracefully(tmp_path: Path) -> None:
    """Повреждённый, пустой и отсутствующий источники не останавливают ETL."""
    from src.data_config import NLPConfig
    from src.nlp_features import NewsFeatureExtractor

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text('{"messages": [', encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text('{"messages": []}', encoding="utf-8")
    config = NLPConfig(
        news_json_paths=(corrupt, empty, tmp_path / "absent.json"),
        cache=tmp_path / "cache.parquet",
        path=tmp_path / "unused.csv",
    )

    result = NewsFeatureExtractor(config)._load_data()

    assert result.empty
    assert result.columns.tolist() == list(NewsFeatureExtractor.NORMALIZED_COLUMNS)


def test_telegram_news_extractor_empty(tmp_path: Path) -> None:
    """NewsFeatureExtractor: отсутствующий файл — пустой DataFrame."""
    from src.data_config import NLPConfig
    from src.nlp_features import NewsFeatureExtractor

    config = NLPConfig(
        telegram_input=tmp_path / "nonexistent.json",
        cache=tmp_path / "cache.parquet",
        path=tmp_path / "news.csv",
    )
    extractor = NewsFeatureExtractor(config)
    df = extractor._load_data()
    assert df.empty


def test_tda_analyzer_smoke() -> None:
    """TopologicalAnalyzer: структурный сигнал, проверка выходной схемы и значений."""
    import pandas as pd
    from src.tda_engine import TDAConfig, TopologicalAnalyzer

    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    t = np.arange(36, dtype=np.float64)
    values = (
        10.0 * np.sin(2 * np.pi * t / 12.0)
        + 0.5 * t
        + np.where(t >= 18, 15.0, 0.0)
        + 100.0
    )
    series = pd.Series(values, index=dates, name="test_mo")

    config = TDAConfig(window_size=6, embedding_dimension=2, time_delay=1)
    analyzer = TopologicalAnalyzer(config)
    result = analyzer.fit_transform(series)

    assert "period" in result.columns, f"Колонки: {result.columns.tolist()}"
    assert "tda_entropy" in result.columns
    assert "tda_wasserstein_dist" in result.columns

    # После shift(1): ws=6, первая строка — NaN (окно [0:6] → период 7)
    assert pd.isna(result["tda_entropy"].iloc[0])
    assert pd.isna(result["tda_wasserstein_dist"].iloc[0])

    # H₁ может быть пустой в отдельных коротких окнах, но на структурном
    # периодическом сигнале должны существовать конечные TDA-значения.
    assert result["tda_entropy"].notna().any()
    assert result["tda_wasserstein_dist"].notna().any()

    # Число результатов равно числу полных скользящих окон.
    assert len(result) == len(series) - config.window_size + 1
    assert result["tda_entropy"].dtype == np.float64
    assert result["tda_wasserstein_dist"].dtype == np.float64


def test_tda_ripser_square_cloud_warning_is_suppressed() -> None:
    """Квадратное облако точек остаётся point cloud без warning от ripser."""
    from src.tda_engine import TDAConfig, TopologicalAnalyzer

    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    series = pd.Series(np.linspace(1.0, 12.0, 12), index=dates)
    analyzer = TopologicalAnalyzer(
        TDAConfig(window_size=5, embedding_dimension=3, time_delay=1)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyzer.fit_transform(series)

    ripser_warnings = [
        item for item in caught if "distance matrix" in str(item.message).lower()
    ]
    assert ripser_warnings == []
    assert len(result) == 8


def test_tda_shock_detection() -> None:
    """TopologicalAnalyzer.detect_shocks: адаптивный порог на синтетике."""
    import pandas as pd
    from src.tda_engine import TDAConfig, TopologicalAnalyzer

    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    # Стабильный сигнал + всплеск
    wass = pd.Series(
        [0.1]*10 + [0.15]*17 + [5.0]*4 + [0.2]*5,
        index=dates,
    )
    config = TDAConfig(shock_threshold_std=2.0)
    analyzer = TopologicalAnalyzer(config)
    shocks = analyzer.detect_shocks(wass)

    assert isinstance(shocks, pd.Series)
    assert shocks.name == "tda_shock"
    # Всплеск после стабильного периода должен детектироваться
    assert shocks.iloc[27:31].any(), "Адаптивный порог должен детектировать всплеск"


def test_tda_config_validation() -> None:
    """TDAConfig: валидация границ."""
    from pydantic import ValidationError

    from src.tda_engine import TDAConfig

    TDAConfig(window_size=3, embedding_dimension=2)  # OK
    try:
        TDAConfig(window_size=2)  # < 3
        raise AssertionError("Должен быть ValidationError")
    except ValidationError:
        pass
    try:
        TDAConfig(embedding_dimension=11)  # > 10
        raise AssertionError("Должен быть ValidationError")
    except ValidationError:
        pass
