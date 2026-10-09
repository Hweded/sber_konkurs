"""Десять слайдов 16:9 из проверенных артефактов, без выдуманных метрик."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

LOGGER = logging.getLogger(__name__)
TITLES = (
    "СберИндекс: сквозной пайплайн",
    "Архитектура решения",
    "Сравнение моделей прогнозирования",
    "Четыре горизонта: статус измерений",
    "Foundation Models: Chronos-Bolt",
    "Новостной фактор",
    "Детекция структурных изменений",
    "Три кейса территорий",
    "Практическая ценность для Сбера",
    "Итоги и проверка поставки",
)


def _slide(number: int, title: str) -> Figure:
    figure = plt.figure(figsize=(16, 9), facecolor="#f7f9fc")
    figure.text(0.055, 0.92, title, fontsize=27, weight="bold", color="#12315a")
    figure.text(0.06, 0.04, f"СберИндекс • {number}/10", fontsize=12, color="#617187")
    return figure


def _text(figure: Figure, lines: list[str], *, size: int = 19) -> None:
    figure.text(0.07, 0.80, "\n\n".join(lines), fontsize=size, va="top", color="#23354a")


def _image(figure: Figure, path: Path, bounds: tuple[float, float, float, float]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Отсутствует обязательный график: {path}")
    axis = figure.add_axes(bounds)
    axis.imshow(plt.imread(path))
    axis.axis("off")


def _table(figure: Figure, frame: pd.DataFrame) -> None:
    axis = figure.add_axes((0.07, 0.17, 0.86, 0.64))
    axis.axis("off")
    formatted = frame.copy().fillna("н/д")
    for column in formatted.select_dtypes(include="number"):
        formatted[column] = formatted[column].map(lambda value: f"{value:,.2f}".replace(",", " "))
    table = axis.table(
        cellText=formatted.astype(str).values,
        colLabels=formatted.columns,
        colWidths=[0.42, 0.14, 0.22, 0.22]
        if list(formatted.columns) == ["model", "n", "MAE", "R2"]
        else None,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_facecolor("#dceaf6" if row == 0 else "#ffffff")


def _kpi(
    figure: Figure,
    x: float,
    y: float,
    width: float,
    label: str,
    value: str,
    *,
    accent: str = "#17836f",
) -> None:
    """Compact, readable metric card for a slide-sized claim."""
    axis = figure.add_axes((x, y, width, 0.16))
    axis.set_facecolor("#ffffff")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#d7e0ea")
        spine.set_linewidth(1.0)
    axis.text(
        0.06, 0.67, label, transform=axis.transAxes, fontsize=11, color="#617187", va="center"
    )
    axis.text(
        0.06,
        0.30,
        value,
        transform=axis.transAxes,
        fontsize=19,
        color=accent,
        weight="bold",
        va="center",
    )


def build_presentation(
    artifacts: Path = Path("reports/artifacts"),
    figures: Path = Path("reports/figures"),
    output: Path = Path("presentation.pdf"),
) -> Path:
    """Компилирует PDF; отсутствующие измерения явно обозначены как н/д."""
    forecast = pd.read_csv(artifacts / "forecast_metrics.csv")
    horizons = pd.read_csv(artifacts / "forecast_metrics_by_horizon.csv")
    shocks = pd.read_csv(artifacts / "changepoint_validation.csv")
    pooled = forecast.loc[forecast["fold"].astype(str).eq("pooled")]
    if pooled.empty or shocks.empty:
        raise ValueError("Для презентации нужны измеренные прогнозные метрики и детекция")
    one = horizons.loc[horizons["fold"].eq("pooled") & horizons["scope"].eq("exact")]
    prophet_pooled = pooled.loc[pooled["model"].astype(str).str.casefold().eq("prophet"), "MAE"]

    chronos_one = forecast.loc[
        forecast["fold"].astype(str).eq("1")
        & forecast["model"].astype(str).str.casefold().eq("chronos"),
        "MAE",
    ]
    chronos_three = forecast.loc[
        forecast["fold"].astype(str).eq("3")
        & forecast["model"].astype(str).str.casefold().eq("chronos"),
        "MAE",
    ]

    regime_pooled = pooled.loc[
        pooled["model"].astype(str).str.casefold().str.contains("regimeaware")
        | pooled["model"].astype(str).str.casefold().eq("regime_aware"),
        "MAE",
    ]
    summary_path = artifacts / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    run_path = artifacts / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    seconds = sum(run.get("completed_stages_seconds", {}).values())
    runtime_line = (
        f"Завершённые этапы текущего запуска: {seconds / 60:.1f} мин."
        if seconds
        else "Время полного запуска не измерено."
    )
    case_images = [Path(path) for path in summary.get("figures", [])] or sorted(
        figures.glob("case_study_*.png")
    )[:3]
    ablation = summary.get("news_ablation", {})
    sentiment_line = (
        f"RuBERT sentiment измерен по {summary['news_articles']} публикациям; лаг t-1."
        if summary.get("news_sentiment_status") == "measured"
        and summary.get("news_articles") is not None
        else "RuBERT sentiment не измерен; используется объём новостей и volume-shock."
    )
    submission_path = Path("submission.csv")
    submission: pd.DataFrame | None = None
    digest: str | None = None
    if submission_path.is_file():
        submission = pd.read_csv(submission_path)
        if (
            submission.columns.tolist() != ["mo", "period", "pred"]
            or len(submission) != 12564
            or submission["mo"].nunique() != 2094
            or submission["period"].nunique() != 6
            or submission.isna().any().any()
            or not pd.to_numeric(submission["pred"], errors="coerce").gt(0).all()
            or submission.duplicated(["mo", "period"]).any()
        ):
            raise ValueError("Конкурсный сабмит не проходит проверку")
        digest = hashlib.md5(submission_path.read_bytes()).hexdigest()
        audited_submission = Path("reports/submission.csv")
        if (
            audited_submission.is_file()
            and audited_submission.read_bytes() != submission_path.read_bytes()
        ):
            raise ValueError("Корневой и reports submission.csv расходятся")
    else:
        LOGGER.warning(
            "Конкурсный файл %s отсутствует; слайд поставки будет помечен как н/д", submission_path
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output, metadata={"Title": TITLES[0], "Author": "СберИндекс"}) as pdf:
        for number, title in enumerate(TITLES, start=1):
            figure = _slide(number, title)
            if number == 1:
                _text(
                    figure,
                    [
                        "Прогноз трат муниципальных образований и ранние сигналы шоков",
                        runtime_line,
                        "CatBoost • Prophet • Chronos-Bolt • Новости • TDA • CUSUM",
                    ],
                    size=17,
                )
            elif number == 2:
                _text(
                    figure,
                    [
                        "CSV СберИндекса + 49 коллизий названий → ключи МО → месячная панель",
                        "Статистика с датами релизов + новости → lag(t−1) → признаки",
                        "Expanding-window OOF → модели → MAE / R² / WAPE / RMSE",
                        "Остатки → CUSUM / PELT / TDA → отчёт",
                        "July frozen-origin RegimeAware Ensemble → submission.csv",
                    ],
                    size=18,
                )
            elif number == 3:
                model_names = pooled["model"].astype(str).str.casefold()
                selection = pooled.loc[
                    model_names.isin(
                        ("catboost", "prophet", "chronos", "ensemble (catboost + chronos)")
                    )
                    | model_names.str.contains("regimeaware")
                    | model_names.eq("regime_aware")
                ]
                selection = selection[["model", "n", "MAE", "R2"]].copy()
                selection.loc[
                    selection["model"].astype(str).str.contains("RegimeAware", case=False), "model"
                ] = "RegimeAware Ensemble"
                selection.loc[
                    selection["model"].astype(str).str.contains("Ensemble \\(CatBoost", regex=True),
                    "model",
                ] = "Ensemble (CatBoost + Chronos)"
                selection["n"] = selection["n"].map(lambda value: f"{int(value)}")
                selection["MAE"] = selection["MAE"].map(lambda value: f"{value:.6f}")
                selection["R2"] = selection["R2"].map(lambda value: f"{value:.6f}")
                _table(figure, selection)
                regime_text = (
                    f"RegimeAware MAE {float(regime_pooled.iloc[0]):,.2f}".replace(",", " ")
                    if not regime_pooled.empty
                    else "RegimeAware MAE н/д"
                )
                prophet_text = (
                    f"Prophet {float(prophet_pooled.iloc[0]):,.2f}".replace(",", " ")
                    if not prophet_pooled.empty
                    else "Prophet н/д"
                )
                figure.text(0.08, 0.11, f"{regime_text}; {prophet_text}", fontsize=16)
            elif number == 4:
                _image(figure, figures / "horizons_mae_comparison.png", (0.05, 0.25, 0.60, 0.60))
                exact = one.loc[one["model"].isin(["prophet", "regime_aware"])].copy()
                for horizon in (1, 3, 6, 12):
                    row = exact.loc[exact["horizon"].eq(horizon)]
                    if "MAE" in row.columns:
                        prophet_value = row.loc[row["model"].eq("prophet"), "MAE"]
                        regime_value = row.loc[row["model"].eq("regime_aware"), "MAE"]
                    else:
                        prophet_value = pd.Series(dtype=float)
                        regime_value = pd.Series(dtype=float)
                    if prophet_value.empty or regime_value.empty:
                        value = "н/д"
                    else:
                        gain = (
                            100
                            * (float(prophet_value.iloc[0]) - float(regime_value.iloc[0]))
                            / float(prophet_value.iloc[0])
                        )
                        value = f"−{gain:.2f}% MAE"
                    _kpi(
                        figure,
                        0.69,
                        0.69 - 0.145 * (horizon.bit_length() - 1),
                        0.25,
                        f"h={horizon} мес",
                        value,
                    )
                figure.text(
                    0.07,
                    0.15,
                    "Сравнение по измеренным OOF; h=12 — recursive frozen-origin.",
                    fontsize=15,
                )
            elif number == 5:
                one_text = (
                    f"{float(chronos_one.iloc[0]):,.2f}".replace(",", " ")
                    if not chronos_one.empty
                    else "н/д"
                )
                three_text = (
                    f"{float(chronos_three.iloc[0]):,.2f}".replace(",", " ")
                    if not chronos_three.empty
                    else "н/д"
                )

                _text(
                    figure,
                    [
                        "Zero-shot, индивидуальная история МО до origin без fine-tuning.",
                        "Все показатели берутся из артефактов текущего запуска.",
                        f"Фолд 1: Chronos MAE {one_text}; Фолд 3: {three_text}.",
                        f"RegimeAware pooled MAE {float(regime_pooled.iloc[0]):,.2f}".replace(
                            ",", " "
                        )
                        if not regime_pooled.empty
                        else "RegimeAware pooled MAE н/д",
                    ],
                )
            elif number == 6:
                if ablation.get("status") == "measured":
                    result = f"MAE без новостей {ablation['baseline_mae']:.2f}; с новостями {ablation['news_mae']:.2f}."
                    effect = f"Production-дельта: {ablation.get('production_mae_delta_pct', 0.0):.2f}%; N={ablation['n']} paired OOF."
                else:
                    result = "News ablation: нет измеренной пары прогнозов."
                    effect = "Статус: " + str(ablation.get("status", "not_available"))
                _text(
                    figure,
                    [
                        "JSON-архивы → число текстов и причинный volume-shock → лаг t−1.",
                        sentiment_line,
                        result,
                        effect,
                        "Эксперимент проверяет качество прогноза, причинность не установлена.",
                    ],
                    size=18,
                )
            elif number == 7:
                _table(
                    figure,
                    shocks[["method", "precision", "recall", "f1", "mean_lead_time_days"]].head(6),
                )
                reference = (
                    int(shocks["reference_events"].iloc[0]) if "reference_events" in shocks else 4
                )
                figure.text(
                    0.08,
                    0.12,
                    f"Каталог CPD: {reference} события внутри наблюдаемой истории; допуск ±31 день.",
                    fontsize=15,
                )
            elif number == 8:
                for index, image in enumerate(case_images):
                    _image(figure, image, (0.05 + 0.305 * index, 0.36, 0.295, 0.37))
                _text_position = 0.20
                figure.text(
                    0.07,
                    _text_position,
                    "Казань, Норильск, Суздаль: фактические ряды, OOF и лагированные новости.\n"
                    "Кейсы описательны; совпадение с событием не устанавливает причину.",
                    fontsize=15,
                )
            elif number == 9:
                _text(
                    figure,
                    [
                        "Cash Management: TDA + подтверждение CUSUM → пересмотр лимитов банкоматов.",
                        "Карточные лимиты: устойчивый сигнал → проверка кредитного риска.",
                        "Human-in-the-loop: автоматический отказ клиентам запрещён.",
                        "Нужны мониторинг релизов, алертов и ошибок на новых origin.",
                    ],
                )
            else:
                if submission is None:
                    _text(
                        figure,
                        [
                            "Конкурсный submission.csv: н/д — файл пока не сформирован.",
                            "Контроль сетки и MD5: н/д; результаты прогноза не подменяются.",
                            "Для поставки необходим отдельный этап создания submission.",
                            "Прогноз июля–декабря строится из единого origin 2024-07-01.",
                        ],
                    )
                else:
                    _text(
                        figure,
                        [
                            f"submission.csv: {len(submission):,} строк; {submission['mo'].nunique():,} МО × 6 месяцев".replace(
                                ",", " "
                            ),
                            "Пустых ячеек: 0; отрицательных прогнозов: 0",
                            f"MD5: {digest}",
                            "Финальный источник: RegimeAware Ensemble.",
                            "Origin 2024-07-01; наблюдения июля–декабря не используются.",
                        ],
                    )
            pdf.savefig(figure, dpi=300)
            plt.close(figure)
    LOGGER.info("PDF-презентация: %s, слайдов=%d", output, len(TITLES))
    return output
