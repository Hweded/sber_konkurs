"""Экспорт измеренных метрик и синхронизированного финального отчёта."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.configuration import AppConfig
from src.forecasting import export_metrics_artifacts, metric_table
from src.ablation import export_news_ablation
from src.foundation import foundation_benchmark_table
from src.visualize import plot_case_study


def select_case_entities(shocks: pd.DataFrame, entity_column: str, limit: int) -> list[str]:
    """Prefer three named economic profiles over alphabetically first rows."""
    available = sorted(shocks[entity_column].dropna().astype(str).unique())
    chosen = []
    for name in ("Казань", "Норильск", "Суздаль"):
        entity = next((value for value in available if name.casefold() in value.casefold()), None)
        if entity is not None:
            chosen.append(entity)
    return list(dict.fromkeys([*chosen, *(value for value in available if value != "__national__")]))[:limit]


def _metric_value(metrics: pd.DataFrame, fold: str, model: str) -> float | None:
    names = {model, model.casefold(), "Ensemble (CatBoost + Chronos)" if model == "ensemble" else model}
    normalized_models = metrics["model"].astype("string").fillna("").str.casefold()
    mask = metrics["fold"].astype(str).eq(str(fold)) & (metrics["model"].isin(names) | normalized_models.eq(model.casefold()))
    values = pd.to_numeric(metrics.loc[mask, "MAE"], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def _format_metric(value: float | None) -> str:
    return "н/д" if value is None else f"{value:,.2f}".replace(",", " ")


def _metrics_from_existing_report(path: Path) -> pd.DataFrame:
    """Восстанавливает уже измеренную таблицу без повторного обучения моделей."""
    if not path.exists():
        raise FileNotFoundError("Нет forecast_metrics.csv, OOF и предыдущего final_report.md")
    pattern = re.compile(
        r"^\|\s*(?P<fold>1|2|3|pooled)\s*\|\s*(?P<model>[^|]+?)\s*\|\s*(?P<n>\d+)\s*\|\s*(?P<mae>[\d.]+)\s*\|\s*(?P<r2>[-\d.]+)\s*\|$",
    )
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        values = match.groupdict()
        rows.append({"fold": values["fold"], "model": values["model"], "n": int(values["n"]), "MAE": float(values["mae"]), "R2": float(values["r2"])})
    if not rows:
        raise ValueError(f"В {path} не найдена таблица прогнозных метрик")
    return pd.DataFrame(rows)


def generate_report(config: AppConfig) -> None:
    settings = config.changepoint_detection
    destination = config.paths.artifacts
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "forecast_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    elif Path(settings.predictions).exists():
        oof = pd.read_parquet(settings.predictions)
        metrics = metric_table(oof)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    else:
        metrics = _metrics_from_existing_report(destination / "final_report.md")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    shocks_path = Path(settings.output)
    shocks = pd.read_parquet(shocks_path) if shocks_path.exists() else pd.DataFrame()
    pooled_catboost = _metric_value(metrics, "pooled", "catboost")
    pooled_chronos = _metric_value(metrics, "pooled", "chronos")
    pooled_prophet = _metric_value(metrics, "pooled", "prophet")
    pooled_ensemble = _metric_value(metrics, "pooled", "ensemble")
    base = pooled_prophet
    improvement = (base - pooled_catboost) / base if base and pooled_catboost is not None else None
    pooled_rows = metrics.loc[metrics["fold"].astype(str).eq("pooled")].copy()
    pooled_rows["MAE"] = pd.to_numeric(pooled_rows["MAE"], errors="coerce")
    pooled_rows = pooled_rows.dropna(subset=["MAE"])
    best_model = str(pooled_rows.loc[pooled_rows["MAE"].idxmin(), "model"]) if not pooled_rows.empty else "н/д"
    figures: list[str] = []
    case_results = []
    if not shocks.empty and settings.entity_column in shocks:
        from src.data_loader import load_target
        history = load_target(config.data.directory, config.data.target, config.data.separator)
        history["period"] = pd.to_datetime(history["period"], utc=True).dt.tz_localize(None)
        for entity in select_case_entities(shocks, settings.entity_column, min(3, settings.max_figures)):
            group = shocks.loc[shocks[settings.entity_column].astype(str).eq(entity)].copy()
            group["period"] = pd.to_datetime(group["period"], utc=True).dt.tz_localize(None)
            past = history.loc[history["mo"].eq(entity), ["period", "y"]]
            if not past.empty:
                group = past.merge(group.drop(columns="y"), on="period", how="left", validate="one_to_one")
                news_source = config.paths.supervised
                if news_source.exists():
                    news_data = pd.read_parquet(news_source)
                    news_data["period"] = pd.to_datetime(news_data["period"], utc=True).dt.tz_localize(None)
                    news_data = news_data.drop_duplicates("period")
                    for column in ("news_shock_score", "sentiment_index", "news_volume"):
                        if column in news_data:
                            mapped = group["period"].map(news_data.set_index("period")[column])
                            group[column] = group[column].fillna(mapped) if column in group else mapped
            figures.append(str(plot_case_study(group, str(entity), Path(settings.figures))))
            paired = group[["y", "prediction"]].dropna()
            monthly = group.set_index("period")["y"]
            growth = monthly.pct_change(fill_method=None)
            case_results.append({"municipality": str(entity), "history_months": len(group),
                                 "oof_rows": len(paired), "MAE": float((paired.y - paired.prediction).abs().mean()) if len(paired) else None,
                                 "july_2024_mom_pct": float(growth.get(pd.Timestamp("2024-07-01"), float("nan")) * 100),
                                 "october_2024_mom_pct": float(growth.get(pd.Timestamp("2024-10-01"), float("nan")) * 100),
                                 "interpretation": "Сезонность, календарь и финансовые условия являются гипотезами; причинность и локальная экспозиция не установлены."})
    (destination / "municipal_case_studies.json").write_text(pd.DataFrame(case_results).to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    validation_path = destination / "changepoint_validation.csv"
    validation = pd.read_csv(validation_path) if validation_path.exists() else pd.DataFrame(columns=["method", "precision", "recall", "f1", "mean_lead_time_days", "coverage_pct"])
    horizon_path = destination / "forecast_metrics_by_horizon.csv"
    horizon = pd.read_csv(horizon_path) if horizon_path.exists() else pd.DataFrame()
    if not horizon.empty:
        export_metrics_artifacts(horizon, destination)
        foundation_payload = foundation_benchmark_table(horizon)
        (destination / "foundation_benchmark.json").write_text(
            json.dumps(foundation_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    ablation_frame = pd.read_parquet(settings.predictions) if Path(settings.predictions).exists() else pd.DataFrame()
    if "lead_months" in ablation_frame:
        ablation_frame = ablation_frame.loc[ablation_frame["lead_months"].eq(1)]
    ablation = export_news_ablation(ablation_frame, destination / "news_ablation.json")
    summary: dict[str, Any] = {
        "catboost_relative_mae_improvement": improvement,
        "catboost_beats_prophet": bool(improvement is not None and improvement > 0),
        "submission_model": "prophet" if pooled_prophet is not None else "fallback",
        "best_pooled_model_by_mae": best_model,
        "pooled_mae": {"catboost": pooled_catboost, "chronos": pooled_chronos, "prophet": pooled_prophet, "ensemble": pooled_ensemble},
        "chronos_fold_1_mae": _metric_value(metrics, "1", "chronos"),
        "chronos_fold_3_mae": _metric_value(metrics, "3", "chronos"),
        "figures": figures,
        "warning": "Согласованность детекторов не доказывает причинность; метрики рассчитаны относительно фиксированного каталога событий ДКП.",
        "news_ablation": ablation,
        "cpd_reference_catalog": config.changepoint_detection.event_catalog,
        "foundation_benchmark": str(destination / "foundation_benchmark.json") if (destination / "foundation_benchmark.json").exists() else None,
    }
    (destination / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_rows = ["| Фолд | Модель | N | MAE | R² |", "|---|---|---:|---:|---:|"]
    metric_rows.extend(f"| {row.fold} | {row.model} | {row.n} | {row.MAE:.2f} | {row.R2} |" for row in metrics.itertuples())
    validation_rows = ["| Метод | Precision | Recall | F1 | Mean lead, дней | Coverage, % |", "|---|---:|---:|---:|---:|---:|"]
    for row in validation.itertuples():
        lead_value = pd.to_numeric(pd.Series([row.mean_lead_time_days]), errors="coerce").iloc[0]
        lead = "н/д" if pd.isna(lead_value) else f"{lead_value:.1f}"
        validation_rows.append(
            f"| {row.method} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {lead} | {row.coverage_pct:.1f} |",
        )
    fold_one = _metric_value(metrics, "1", "chronos")
    fold_three = _metric_value(metrics, "3", "chronos")
    improvement_pct = "н/д" if improvement is None else f"{improvement:.2%}"
    comparison_text = (
        f"CatBoost снизил MAE относительно Prophet на {improvement_pct}. "
        if improvement is not None and improvement >= 0 else
        f"CatBoost уступил Prophet по MAE на {-improvement:.2%}. "
        if improvement is not None else "Сравнение CatBoost с Prophet недоступно. "
    )
    report = (
        "# Финальный отчёт\n\n"
        "## Прогноз\n\n"
        + "\n".join(metric_rows)
        + f"\n\nCatBoost закончил с pooled MAE {_format_metric(pooled_catboost)}, Chronos — {_format_metric(pooled_chronos)}, "
        + f"Prophet — {_format_metric(pooled_prophet)}. {comparison_text}"
        + f"Лучшая pooled MAE среди измеренных моделей: {best_model}. Финальный сабмит строится на Prophet.\n\n"
        + f"Chronos: MAE {_format_metric(fold_one)} на первом фолде; третий фолд — {_format_metric(fold_three)}. "
        + "Пустой фолд обозначается как н/д и не заменяется ручным числом.\n\n"
        "## Шоки\n\n"
        + "\n".join(validation_rows)
        + "\n\nКаталог включает шесть событий 2022–2024; два события 2022 года вне доступной истории, поэтому recall использует четыре. Допуск — ±31 день. "
        + f"Макроалерт включается от {settings.macro_alert_share:.1%} доступных МО либо на агрегированном ряду.\n\n"
        + "TDA ловит изменение формы фазовой траектории, а не скачок среднего в лоб. "
        + "PELT чаще подтверждает уже сложившийся режим, CUSUM работает как чувствительный онлайн-алерт. "
        + "Четырёх наблюдаемых событий недостаточно для универсальных выводов, поэтому эти метрики относятся только к текущему каталогу.\n"
        + "\n## News ablation\n\n"
        + ("Выполнено: MAE baseline = " + f"{ablation.get('baseline_mae', float('nan')):.2f}, "
           + "MAE с новостями = " + f"{ablation.get('news_mae', float('nan')):.2f}, "
           + "изменение = " + f"{ablation.get('mae_improvement_pct', float('nan')):.2f}%. "
           + "Проверка доступности новостей: " + str(ablation.get('leakage_check', 'n/a')) + ". "
           + ("Снижение detection delay = " + f"{ablation['detection_delay_reduction_days']:.2f} дней.\n"
              if "detection_delay_reduction_days" in ablation else "Detection delay для обеих ветвей не передан.\n")
           if ablation.get("status") == "measured" else
           "Анализ ожидает две раздельно обученные OOF-колонки pred_catboost_no_news и pred_catboost_news; ручные числа не подставляются.\n")
    )
    (destination / "final_report.md").write_text(report, encoding="utf-8")
