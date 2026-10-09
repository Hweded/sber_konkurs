"""Потоковая тональность экономических новостей и версионированный кэш."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from src.data_config import NLPConfig
from src.data_loader import month_start

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWS_SCHEMA_VERSION = 4
NEWS_COLUMNS = ("sentiment_index", "news_volume", "news_shock_score")
TELEGRAM_NEWS_COLUMNS = ("telegram_sentiment", "telegram_volume", "telegram_shock_score")
JSON_TO_NEWS_COLUMNS = dict(zip(TELEGRAM_NEWS_COLUMNS, NEWS_COLUMNS, strict=True))
Scorer = Callable[[Sequence[str]], NDArray[np.float64]]
_PROGRESS_LOG_INTERVAL_SECONDS = 30.0


class SentimentScorer:
    """Открытая модель с ОБУЧЕННОЙ sentiment-головой, не голый rubert-tiny2."""

    def __init__(self, config: NLPConfig) -> None:
        # На Windows эти переменные нужны до импорта torch, иначе ловили deadlock OpenMP.
        cpu_threads = str(max(1, os.cpu_count() or 4))
        os.environ.setdefault("OMP_NUM_THREADS", cpu_threads)
        os.environ.setdefault("MKL_NUM_THREADS", cpu_threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", cpu_threads)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.device import resolve_device

        device_info = resolve_device(config.device)
        self.device_info = device_info
        self.device = torch.device(device_info.device)
        if device_info.device == "cpu":
            torch.set_num_threads(int(cpu_threads))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        self.torch = torch
        self.config = config
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            config.model_id, revision=config.revision, trust_remote_code=False
        )
        self.model: Any = (
            AutoModelForSequenceClassification.from_pretrained(
                config.model_id, revision=config.revision, trust_remote_code=False
            )
            .to(self.device)
            .eval()
        )
        labels = {str(value).lower(): int(key) for key, value in self.model.config.id2label.items()}
        if not {"positive", "negative"}.issubset(labels):
            raise ValueError(f"Модель не имеет именованных positive/negative меток: {labels}")
        self.positive, self.negative = labels["positive"], labels["negative"]

    def __call__(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if not texts:
            return np.empty(0, dtype=np.float64)
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)
        try:
            with (
                self.torch.inference_mode(),
                self.torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.mixed_precision and self.device.type == "cuda",
                ),
            ):
                probabilities = self.model(**encoded).logits.softmax(dim=-1)
        except RuntimeError as error:
            from src.device import cuda_memory_error

            if not cuda_memory_error(error) or not self.config.fallback_to_cpu_on_oom:
                raise
            LOGGER.error("NLP CUDA OOM; CPU fallback выполняется для текущего батча")
            self.model.to("cpu")
            self.device = self.torch.device("cpu")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                probabilities = self.model(**encoded).logits.softmax(dim=-1)
        scores = probabilities[:, self.positive] - probabilities[:, self.negative]
        return np.asarray(scores.cpu().numpy(), dtype=np.float64)


def _fingerprint(config: NLPConfig) -> str:
    stat = config.path.stat()
    payload = {
        "version": 1,
        "config": config.model_dump(mode="json"),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def news_coverage(config: NLPConfig) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Потоково находит первую и последнюю дату архива без запуска модели."""
    import csv

    minimum = pd.Timestamp(config.min_date)
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    with config.path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if "date" not in header:
            raise ValueError(f"Нет колонки date в новостном корпусе: {header}")
        date_index = header.index("date")
        for row in reader:
            if date_index >= len(row):
                continue
            parsed = pd.to_datetime(
                row[date_index].replace("/", "-"), format="%Y-%m-%d", errors="coerce"
            )
            if pd.isna(parsed) or parsed < minimum:
                continue
            stamp = pd.Timestamp(parsed)
            first = stamp if first is None else min(first, stamp)
            last = stamp if last is None else max(last, stamp)
    if first is None or last is None:
        return None
    return first, last


def build_news_features(config: NLPConfig, *, scorer: Scorer | None = None) -> pd.DataFrame:
    """Считает лагированные новостные признаки по экономическим статьям."""
    started = time.perf_counter()
    manifest = config.cache.with_suffix(".manifest.json")
    # Валидный кэш должен отработать до импорта torch и загрузки весов.
    fingerprint: str | None = None
    if config.path.exists():
        fingerprint = _fingerprint(config)
    if config.cache.exists():
        cached = pd.read_parquet(config.cache)
        valid_schema = (
            list(cached.columns) == ["period", *NEWS_COLUMNS]
            and not cached["period"].duplicated().any()
        )
        if not valid_schema:
            raise ValueError("Некорректная схема NLP-кэша")
        metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        cache_is_current = not manifest.exists() or (
            fingerprint is not None and metadata.get("fingerprint") == fingerprint
        )
        if cache_is_current:
            LOGGER.info(
                "NLP: кэш загружен за %.2fs: %s, %d месяцев",
                time.perf_counter() - started,
                config.cache,
                len(cached),
            )
            return cached
        LOGGER.info("NLP: fingerprint кэша устарел, запускается пересчёт")
    if not config.publication_date_is_availability:
        raise ValueError("Архив без first_seen_at требует явного допущения о дате доступности")
    totals: dict[pd.Timestamp, tuple[float, int]] = {}
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    score_batch = scorer
    score_cache: dict[str, float] = {}
    tag_set = {value.casefold() for value in config.tags}
    minimum = pd.Timestamp(config.min_date)
    LOGGER.info(
        "NLP: чтение CSV чанками=%d, batch=%d, max_length=%d",
        config.chunksize,
        config.batch_size,
        config.max_length,
    )
    rows_read = 0
    batches_done = 0
    chunk_started = time.perf_counter()
    chunk_progress = tqdm(
        pd.read_csv(
            config.path,
            encoding="utf-8",
            usecols=["date", "title", "text", "topic", "tags"],
            dtype="string",
            chunksize=config.chunksize,
        ),
        desc="NLP CSV chunks",
        unit="chunk",
        dynamic_ncols=True,
    )
    for number, chunk in enumerate(chunk_progress, start=1):
        rows_read += len(chunk)
        dates = pd.to_datetime(
            chunk["date"].str.replace("/", "-", regex=False), format="%Y-%m-%d", errors="coerce"
        )
        valid = dates.notna() & dates.ge(minimum)
        if valid.any():
            low, high = pd.Timestamp(dates[valid].min()), pd.Timestamp(dates[valid].max())
            first = low if first is None else min(first, low)
            last = high if last is None else max(last, high)
        selected = chunk["topic"].str.strip().isin(config.topics)
        if tag_set:
            selected |= (
                chunk["tags"]
                .fillna("")
                .map(
                    lambda value: bool(
                        tag_set.intersection(
                            token.strip().casefold() for token in re.split(r"[,;|]", str(value))
                        )
                    )
                )
            )
        selected &= valid
        subset = chunk.loc[selected]
        months = dates.loc[selected].dt.to_period("M").dt.to_timestamp()
        texts = (subset["title"].fillna("") + "\n" + subset["text"].fillna("")).str.strip()
        if texts.eq("").any():
            raise ValueError("Экономическая статья не содержит ни заголовка, ни текста")
        unique_texts = [
            text for text in texts.drop_duplicates().tolist() if text not in score_cache
        ]
        if unique_texts and score_batch is None:
            score_batch = SentimentScorer(config)
        with tqdm(
            total=len(unique_texts), desc="RuBERT Inference", unit="texts", dynamic_ncols=True
        ) as progress:
            for start in range(0, len(unique_texts), config.batch_size):
                batch = unique_texts[start : start + config.batch_size]
                assert score_batch is not None
                scores = np.asarray(score_batch(batch), dtype=np.float64)
                if (
                    scores.shape != (len(batch),)
                    or not np.isfinite(scores).all()
                    or (np.abs(scores) > 1).any()
                ):
                    raise ValueError("Невалидные оценки тональности")
                score_cache.update(zip(batch, scores.tolist(), strict=True))
                batches_done += 1
                progress.update(len(batch))
                if time.perf_counter() - chunk_started >= _PROGRESS_LOG_INTERVAL_SECONDS:
                    LOGGER.info(
                        "NLP chunk %d: inference progress=%d/%d texts, batches=%d, elapsed=%.1fs",
                        number,
                        min(start + len(batch), len(unique_texts)),
                        len(unique_texts),
                        batches_done,
                        time.perf_counter() - started,
                    )
        mapped = texts.map(score_cache)
        if mapped.isna().any():
            raise ValueError("Не удалось сопоставить тональность всем статьям")
        for month, score in zip(months, mapped, strict=True):
            key = pd.Timestamp(month)
            total, count = totals.get(key, (0.0, 0))
            totals[key] = total + float(score), count + 1
        elapsed = time.perf_counter() - started
        chunk_elapsed = time.perf_counter() - chunk_started
        LOGGER.info(
            "NLP chunk %d завершён: строк=%d, экономических=%d, уникальных=%d, батчей=%d, chunk=%.1fs, elapsed=%.1fs",
            number,
            rows_read,
            len(subset),
            len(unique_texts),
            batches_done,
            chunk_elapsed,
            elapsed,
        )
        chunk_started = time.perf_counter()
    chunk_progress.close()
    if first is None or last is None:
        raise ValueError("Нет валидных дат новостного корпуса")
    calendar = pd.date_range(
        first.to_period("M").start_time, last.to_period("M").start_time, freq="MS"
    )
    result = pd.DataFrame(index=calendar)
    result["news_volume"] = [totals.get(date, (0.0, 0))[1] for date in calendar]
    result["sentiment_index"] = [
        total / count if count else np.nan
        for total, count in (totals.get(date, (0.0, 0)) for date in calendar)
    ]
    history = (
        result["news_volume"]
        .shift(1)
        .rolling(config.shock_window, min_periods=config.shock_window)
        .mean()
    )
    result["news_shock_score"] = result["news_volume"] - history
    calendar = pd.date_range(
        calendar[0], calendar[-1] + pd.offsets.MonthBegin(config.lag_months), freq="MS"
    )
    result = result.reindex(calendar).shift(config.lag_months)
    result.index.name = "period"
    result = result.reset_index()[["period", *NEWS_COLUMNS]]
    config.cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.cache.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False)
    temporary.replace(config.cache)
    if fingerprint is None:
        fingerprint = _fingerprint(config)
    manifest.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "coverage_start": first.isoformat(),
                "coverage_end": last.isoformat(),
                "lagged": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "NLP: кэш сохранён %s, покрытие %s — %s, строк=%d, батчей=%d, elapsed=%.1fs",
        config.cache,
        first,
        last,
        rows_read,
        batches_done,
        time.perf_counter() - started,
    )
    return result


def resolve_json_news_paths(
    config: NLPConfig,
    *,
    project_root: Path = PROJECT_ROOT,
    data_directory: Path | None = None,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Находит настроенные и известные JSON-выгрузки новостей."""
    roots = [project_root]
    if data_directory is not None:
        directory = (
            data_directory if data_directory.is_absolute() else project_root / data_directory
        )
        roots.append(directory)
    else:
        roots.append(project_root / "datasets")

    configured = [*config.news_json_paths]
    if config.telegram_input is not None:
        configured.insert(0, config.telegram_input)
    checked: list[Path] = []
    for candidate in configured:
        variants = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
        if (
            not candidate.is_absolute()
            and candidate.parts
            and candidate.parts[0].casefold() == "datasets"
        ):
            variants.append(roots[-1] / Path(*candidate.parts[1:]))
        checked.extend(path.resolve() for path in variants)

    known_names = {"smartlab.json", "bank offshore.json", "rbk.json"}
    if config.news_auto_discover:
        for root in roots:
            if root.exists():
                checked.extend(
                    path.resolve()
                    for path in root.glob("*.json")
                    if path.name.casefold() in known_names
                )
    unique_checked = tuple(dict.fromkeys(checked))
    existing = tuple(path for path in unique_checked if path.is_file())
    return existing, unique_checked


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


class NewsFeatureExtractor:
    """Извлекает лагированные месячные признаки из JSON-новостей."""

    DEFAULT_KEYWORDS: tuple[str, ...] = (
        "инфляция",
        "ставка",
        "банк",
        "цены",
        "санкции",
        "дефицит",
        "налоги",
        "бюджет",
        "расходы",
        "спрос",
        "ввп",
        "импорт",
        "экспорт",
        "рецессия",
        "стагфляция",
    )
    CACHE_NAME: str = "news_features.parquet"

    NORMALIZED_COLUMNS: tuple[str, ...] = (
        "date",
        "period",
        "text",
        "title",
        "source",
        "url",
        "category",
        "article_id",
    )
    DATE_FIELDS: tuple[str, ...] = (
        "date",
        "published_at",
        "published",
        "publication_date",
        "created_at",
        "datetime",
        "timestamp",
    )
    TITLE_FIELDS: tuple[str, ...] = ("title", "headline", "name")
    TEXT_FIELDS: tuple[str, ...] = ("text", "description", "body", "content", "summary")
    SOURCE_FIELDS: tuple[str, ...] = ("source", "from", "author", "channel")
    URL_FIELDS: tuple[str, ...] = ("url", "link", "href")
    CATEGORY_FIELDS: tuple[str, ...] = ("category", "topic", "section", "tags")
    ID_FIELDS: tuple[str, ...] = ("id", "article_id", "message_id", "guid", "uuid")

    def __init__(
        self,
        config: NLPConfig,
        *,
        project_root: Path = PROJECT_ROOT,
        data_directory: Path | None = None,
    ) -> None:
        self.config = config
        self._project_root = project_root.resolve()
        self._scorer: Any = None
        self._keywords = config.telegram_keywords or self.DEFAULT_KEYWORDS
        self._min_length = config.telegram_min_length
        self._source_paths, self._checked_paths = resolve_json_news_paths(
            config,
            project_root=project_root,
            data_directory=data_directory,
        )
        explicit_paths = [*config.news_json_paths]
        if config.telegram_input is not None:
            explicit_paths.append(config.telegram_input)
        if explicit_paths:
            data_root = data_directory or project_root / "datasets"
            if not data_root.is_absolute():
                data_root = project_root / data_root
            explicit_resolved = {
                path.resolve()
                for candidate in explicit_paths
                for path in (
                    [candidate]
                    if candidate.is_absolute()
                    else [project_root / candidate, data_root / candidate.name]
                )
            }
            self._source_paths = tuple(
                path for path in self._source_paths if path in explicit_resolved
            )
        cache_root = config.cache if config.cache.is_absolute() else project_root / config.cache
        self._cache_path = cache_root.parent / self.CACHE_NAME
        try:
            self._timezone = ZoneInfo(config.news_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(
                f"Неизвестный часовой пояс новостей: {config.news_timezone}"
            ) from error

    @classmethod
    def _extract_text(cls, value: Any) -> str:
        """Безопасно извлекает текст из строк, Telegram entities и вложенных значений."""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return " ".join(filter(None, (cls._extract_text(item) for item in value)))
        if isinstance(value, dict):
            for key in ("text", "value", "content", "description"):
                if key in value:
                    return cls._extract_text(value[key])
        return ""

    @staticmethod
    def _first(record: dict[str, Any], fields: Sequence[str]) -> Any:
        for field in fields:
            value = record.get(field)
            if value is not None and value != "":
                return value
        return None

    def _clean_text(self, value: Any) -> str:
        text = html.unescape(self._extract_text(value)).replace("\x00", " ")
        if self.config.news_strip_html:
            text = _HTML_TAG_RE.sub(" ", text)
        if self.config.news_remove_urls:
            text = _URL_RE.sub(" ", text)
        return _WHITESPACE_RE.sub(" ", text).strip()[: self.config.news_max_text_length]

    @staticmethod
    def _normalize_url(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        candidate = value.strip()
        if candidate.startswith("www."):
            candidate = "https://" + candidate
        try:
            parsed = urlsplit(candidate)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return ""
            return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", ""))
        except ValueError:
            return ""

    def _parse_date(self, value: Any) -> pd.Timestamp | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
            numeric = float(value)
            unit = "ms" if abs(numeric) >= 10**11 else "s"
            parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(value, utc=False, errors="coerce")
        if pd.isna(parsed):
            return None
        stamp = pd.Timestamp(parsed)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(self._timezone, ambiguous="NaT", nonexistent="shift_forward")
        else:
            stamp = stamp.tz_convert(self._timezone)
        if pd.isna(stamp):
            return None
        return stamp.tz_localize(None)

    @staticmethod
    def _iter_records(raw: Any) -> Iterator[dict[str, Any]]:
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    yield item
            return
        if not isinstance(raw, dict):
            return
        for key in ("messages", "items", "articles", "news", "results", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))
                return
            if isinstance(value, dict):
                yield from NewsFeatureExtractor._iter_records(value)
                return
        if any(
            field in raw
            for field in (*NewsFeatureExtractor.DATE_FIELDS, *NewsFeatureExtractor.TEXT_FIELDS)
        ):
            yield raw

    @staticmethod
    def _record_prefix(path: Path) -> str | None:
        with path.open("rb") as stream:
            head = stream.read(64 * 1024).decode("utf-8-sig", errors="replace")
        for key in ("messages", "items", "articles", "news", "results"):
            if re.search(rf'"{key}"\s*:\s*\[', head):
                return f"{key}.item"
        stripped = head.lstrip()
        if stripped.startswith("["):
            return "item"
        return None

    def _iter_source_records(self, path: Path) -> Iterator[dict[str, Any]]:
        """Потоково читает крупные JSON-массивы без загрузки файла в память."""
        try:
            import ijson

            ijson_error = getattr(ijson, "JSONError", None)
            if ijson_error is None:
                ijson_error = getattr(getattr(ijson, "common", None), "JSONError", ValueError)
        except ImportError:
            # ijson — необязательное ускорение; fallback использует тот же путь
            # нормализации и подходит для небольших JSON.
            try:
                with path.open("r", encoding="utf-8-sig") as stream:
                    raw = json.load(stream)
                yield from self._iter_records(raw)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                LOGGER.warning("JSON-новости %s повреждены или недоступны: %s", path.name, error)
            return
        try:
            prefix = self._record_prefix(path)
            if prefix is None:
                LOGGER.warning(
                    "JSON-новости %s: не найден поддерживаемый массив записей", path.name
                )
                return
            # ijson не держит вторую строковую копию многогигабайтного файла.
            with path.open("rb") as stream:
                for item in ijson.items(stream, prefix):
                    if isinstance(item, dict):
                        yield item
        except (OSError, UnicodeError, ValueError, ijson_error) as error:
            LOGGER.warning("JSON-новости %s повреждены или недоступны: %s", path.name, error)
        except MemoryError:
            LOGGER.error("JSON-новости %s не обработаны: недостаточно памяти", path.name)

    def _normalize_record(
        self, record: dict[str, Any], *, default_source: str
    ) -> tuple[dict[str, Any] | None, str]:
        date = self._parse_date(self._first(record, self.DATE_FIELDS))
        if date is None:
            return None, "invalid_date"
        title = self._clean_text(self._first(record, self.TITLE_FIELDS))
        body_parts = [
            self._clean_text(record.get(field)) for field in self.TEXT_FIELDS if field in record
        ]
        body = " ".join(dict.fromkeys(part for part in body_parts if part))
        text = _WHITESPACE_RE.sub(" ", f"{title} {body}").strip()
        if not text:
            return None, "empty_text"
        if len(text) < self._min_length:
            return None, "short_text"
        if self._keywords and not any(
            keyword.casefold() in text.casefold() for keyword in self._keywords
        ):
            return None, "keyword_filter"
        source = self._clean_text(self._first(record, self.SOURCE_FIELDS)) or default_source
        article_id = self._clean_text(self._first(record, self.ID_FIELDS))
        return {
            "date": date,
            "period": date.to_period("M").to_timestamp(),
            "text": text[: self.config.news_max_text_length],
            "title": title,
            "source": source,
            "url": self._normalize_url(self._first(record, self.URL_FIELDS)),
            "category": self._clean_text(self._first(record, self.CATEGORY_FIELDS)),
            "article_id": article_id,
        }, "ok"

    def _load_data(self) -> pd.DataFrame:
        """Нормализует JSON-источники и пропускает только повреждённые записи."""
        if not self._source_paths:
            LOGGER.warning(
                "JSON-источники новостей не найдены; проверены пути: %s",
                [str(path) for path in self._checked_paths],
            )
            return pd.DataFrame(columns=self.NORMALIZED_COLUMNS)
        records: list[dict[str, Any]] = []
        rejection_counts: dict[str, int] = {}
        existing_sources = 0
        for path in self._source_paths:
            existing_sources += 1
            source_count = 0
            for record in self._iter_source_records(path):
                normalized, reason = self._normalize_record(record, default_source=path.stem)
                if normalized is None:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue
                records.append(normalized)
                source_count += 1
            LOGGER.info("JSON-новости %s: принято %d записей", path.name, source_count)
        if not records:
            if existing_sources:
                LOGGER.warning("JSON-новости: нет пригодных записей; причины=%s", rejection_counts)
            return pd.DataFrame(columns=self.NORMALIZED_COLUMNS)

        result = pd.DataFrame.from_records(records, columns=self.NORMALIZED_COLUMNS)
        result = result.sort_values("date", kind="stable").reset_index(drop=True)
        if self.config.news_deduplicate:
            normalized_text = (
                result["text"].str.casefold().str.replace(r"\W+", " ", regex=True).str.strip()
            )
            text_hash = normalized_text.map(
                lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
            )
            id_key = result["source"].astype(str) + "\x1f" + result["article_id"].astype(str)
            duplicate_id = result["article_id"].ne("") & id_key.duplicated(keep="first")
            duplicate_url = result["url"].ne("") & result["url"].duplicated(keep="first")
            duplicate_text = text_hash.duplicated(keep="first")
            duplicate = duplicate_id | duplicate_url | duplicate_text
            if duplicate.any():
                LOGGER.info("JSON-новости: удалено дубликатов %d", int(duplicate.sum()))
                result = result.loc[~duplicate].reset_index(drop=True)
        LOGGER.info(
            "JSON-новости: принято %d, отклонено %d; причины=%s; период %s — %s",
            len(result),
            sum(rejection_counts.values()),
            rejection_counts,
            result["date"].min().date(),
            result["date"].max().date(),
        )
        return result

    def _get_scorer(self) -> Any:
        if self._scorer is None:
            cpu_threads = str(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
            os.environ.setdefault("OMP_NUM_THREADS", cpu_threads)
            os.environ.setdefault("MKL_NUM_THREADS", cpu_threads)
            os.environ.setdefault("OPENBLAS_NUM_THREADS", cpu_threads)
            import torch
            from src.device import resolve_device

            device_info = resolve_device(self.config.device)
            self._device = torch.device(device_info.device)
            if device_info.device == "cpu":
                torch.set_num_threads(int(cpu_threads))
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_id = self.config.model_id
            LOGGER.info("JSON NLP: загрузка токенизатора %s (device=%s)", model_id, self._device)
            tokenizer_started = time.perf_counter()
            self._tokenizer: Any = AutoTokenizer.from_pretrained(
                model_id,
                revision=self.config.revision,
                trust_remote_code=False,
            )
            LOGGER.info(
                "JSON NLP: токенизатор загружен за %.1fs", time.perf_counter() - tokenizer_started
            )
            LOGGER.info(
                "JSON NLP: загрузка модели %s (device=%s); операция может занять минуты",
                model_id,
                self._device,
            )
            model_started = time.perf_counter()
            self._model: Any = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    revision=self.config.revision,
                    trust_remote_code=False,
                )
                .to(self._device)
                .eval()
            )
            LOGGER.info("JSON NLP: модель загружена за %.1fs", time.perf_counter() - model_started)
            labels = {str(v).lower(): int(k) for k, v in self._model.config.id2label.items()}
            if not {"positive", "negative"}.issubset(labels):
                raise ValueError(f"Модель не имеет positive/negative меток: {labels}")
            self._pos_id = labels["positive"]
            self._neg_id = labels["negative"]
            self._torch = torch
            self._scorer = True
        return self._scorer

    def _score_batch(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Вычисляет sentiment для батча текстов."""
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self._device)
        from src.device import cuda_memory_error

        try:
            with (
                self._torch.inference_mode(),
                self._torch.autocast(
                    device_type=self._device.type,
                    enabled=self.config.mixed_precision and self._device.type == "cuda",
                ),
            ):
                probs = self._model(**encoded).logits.softmax(dim=-1)
        except RuntimeError as error:
            if not cuda_memory_error(error) or not self.config.fallback_to_cpu_on_oom:
                raise
            LOGGER.error("JSON NLP CUDA OOM; CPU fallback включён")
            self._model.to("cpu")
            self._device = self._torch.device("cpu")
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                probs = self._model(**encoded).logits.softmax(dim=-1)
        scores = probs[:, self._pos_id] - probs[:, self._neg_id]
        return np.asarray(scores.cpu().numpy(), dtype=np.float64)

    def compute_nlp_features(self, force: bool = False) -> pd.DataFrame:
        """Агрегирует JSON-новости по месяцам и обновляет кэш."""
        manifest = self._cache_path.with_suffix(".manifest.json")

        source_signature = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in self._source_paths
        ]
        config_signature = self.config.model_dump(mode="json")
        for key in ("path", "cache"):
            configured_path = Path(config_signature[key])
            if configured_path.is_absolute():
                try:
                    config_signature[key] = str(configured_path.relative_to(self._project_root))
                except ValueError:
                    pass
        fingerprint_payload = {
            "schema_version": NEWS_SCHEMA_VERSION,
            "config": config_signature,
            "sources": source_signature,
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if not force and self._cache_path.exists() and manifest.exists():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                cached = pd.read_parquet(self._cache_path)
                if (
                    list(cached.columns) == ["period", *TELEGRAM_NEWS_COLUMNS]
                    and not cached["period"].duplicated().any()
                ):
                    LOGGER.info(
                        "JSON NLP: загружен кэш %s, %d месяцев", self._cache_path, len(cached)
                    )
                    return cached

        df = self._load_data()
        if df.empty:
            return pd.DataFrame(columns=["period", *TELEGRAM_NEWS_COLUMNS])

        if self.config.news_sentiment_enabled:
            try:
                unique_texts = df["text"].drop_duplicates().astype(str)
                text_values = unique_texts.tolist()
                self._get_scorer()
                batches = range(0, len(text_values), self.config.batch_size)
                scores: list[float] = []
                for start in tqdm(
                    batches,
                    total=(len(text_values) + self.config.batch_size - 1) // self.config.batch_size,
                    desc="RuBERT Sentiment",
                    unit="batch",
                    dynamic_ncols=True,
                ):
                    batch = text_values[start : start + self.config.batch_size]
                    batch_scores = np.asarray(self._score_batch(batch), dtype=np.float64)
                    if batch_scores.shape != (len(batch),) or not np.isfinite(batch_scores).all():
                        raise ValueError("Некорректная форма или значения sentiment")
                    scores.extend(batch_scores.tolist())
                scores_by_text = pd.Series(scores, index=text_values, dtype="float64")
                df["sentiment"] = df["text"].map(scores_by_text)
                if df["sentiment"].isna().any():
                    raise ValueError("Не всем текстам сопоставлена оценка sentiment")
            except Exception as error:
                raise RuntimeError("JSON NLP: включённый sentiment не удалось измерить") from error
        else:
            LOGGER.info("JSON NLP: sentiment отключён, рассчитываются volume/shock")
            df["sentiment"] = np.nan
        df["period"] = month_start(df["period"])

        # Пустой месяц внутри покрытия отличается от отсутствующего архива.
        grouped = df.groupby("period").agg(
            telegram_sentiment=("sentiment", "mean"),
            article_count=("text", "size"),
        )
        calendar = pd.date_range(df["period"].min(), df["period"].max(), freq="MS")
        monthly = grouped.reindex(calendar)
        monthly.index.name = "period"
        monthly["article_count"] = monthly["article_count"].fillna(0.0)
        monthly["telegram_volume"] = np.log1p(monthly["article_count"].to_numpy(dtype=float))

        # Текущий месяц не участвует в собственном пороге шока.
        ws = self.config.shock_window
        history = monthly["telegram_volume"].shift(1).rolling(window=ws, min_periods=ws)
        mean = history.mean()
        std = history.std()
        monthly["telegram_shock_score"] = (monthly["telegram_volume"] - mean) / std.where(
            std.gt(1e-10)
        )
        monthly["telegram_shock_score"] = monthly["telegram_shock_score"].fillna(0.0)
        monthly = monthly.drop(columns="article_count")

        lag = self.config.lag_months
        extended = pd.date_range(
            monthly.index[0], monthly.index[-1] + pd.offsets.MonthBegin(lag), freq="MS"
        )
        monthly = monthly.reindex(extended).shift(lag)
        monthly.index.name = "period"
        monthly = monthly.reset_index()[["period", *TELEGRAM_NEWS_COLUMNS]]

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._cache_path.with_suffix(".tmp.parquet")
        monthly.to_parquet(temporary, index=False)
        temporary.replace(self._cache_path)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": NEWS_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "sources": source_signature,
                    "periods": len(monthly),
                    "coverage_start": df["date"].min().isoformat(),
                    "coverage_end": df["date"].max().isoformat(),
                    "lag_months": self.config.lag_months,
                    "sentiment_status": "measured"
                    if df["sentiment"].notna().any()
                    else "not_measured",
                    "shock_signal": "standardized_log_article_count_vs_past_months",
                    "articles": len(df),
                    "columns": list(TELEGRAM_NEWS_COLUMNS),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info(
            "Telegram NLP: кэш сохранён %s, %d месяцев",
            self._cache_path,
            len(monthly),
        )
        return monthly


def build_telegram_news_features(
    config: NLPConfig,
    *,
    force: bool = False,
    project_root: Path = PROJECT_ROOT,
    data_directory: Path | None = None,
) -> pd.DataFrame:
    """Строит признаки JSON-новостей или возвращает пустую таблицу."""
    extractor = NewsFeatureExtractor(
        config,
        project_root=project_root,
        data_directory=data_directory,
    )
    return extractor.compute_nlp_features(force=force)
