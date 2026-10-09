"""Изолированная проверка ансамбля по кэшам без переобучения моделей."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import pandas as pd

from src.cache_io import cached_paths, load_cached_predictions

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FAST_DIR = ROOT / "reports" / "fast_run"
PROTECTED = (
    "submission.csv",
    "metrics.json",
    "reports/submission.csv",
    "reports/artifacts/metrics.json",
    "reports/artifacts/tables.md",
)
# Эталонный MD5 текущего кандидата весов.
EXPECTED_SUBMISSION_MD5 = "e3b2d8abc292640a5be7e1a955c2a3d6"
SUBMISSION_COLUMN = "pred_regime_aware"
SUBMISSION_LEAD = 6


def _rel(path: Path) -> str:
    """Путь относительно корня репозитория; если он вне корня — как есть."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _abs(path: Path) -> Path:
    """Абсолютный путь: относительные пути конфигурации считаются от корня репозитория."""
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _md5(path: Path) -> str | None:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def snapshot_protected(root: Path = ROOT) -> dict[str, str | None]:
    """MD5 боевых артефактов — до и после прогона."""
    return {name: _md5(root / name) for name in PROTECTED}


def _assert_isolated(target: Path, *, fast: bool, fast_dir: Path) -> None:
    """В быстром режиме любой путь обязан лежать внутри fast_dir."""
    if not fast:
        return
    resolved = target.resolve()
    if resolved != fast_dir.resolve() and fast_dir.resolve() not in resolved.parents:
        raise RuntimeError(f"ГАРДРЕЙЛ: быстрый режим пытался писать вне {fast_dir}: {target}")


def _enforce_protected(
    before: dict[str, str | None], after: dict[str, str | None], *, fast: bool
) -> None:
    """Пост-условие: в быстром режиме боевые артефакты не изменились."""
    if not fast:
        return
    changed = [name for name in before if before[name] != after.get(name)]
    if changed:
        raise RuntimeError(
            "ГАРДРЕЙЛ НАРУШЕН: быстрый режим изменил боевые артефакты: "
            + ", ".join(f"{n} ({before[n]} -> {after.get(n)})" for n in changed)
        )


def _blender(weights: dict[int, Sequence[float]] | None):
    """Тот же блендер, что в боевом пайплайне; при None — заводские веса."""
    from src.ensemble import RegimeAwareBlender
    from src.forecasting import RegimeAwareEnsemble

    if weights is None:
        return RegimeAwareEnsemble()
    return RegimeAwareEnsemble(blender=RegimeAwareBlender(weights=weights))


def load_weights(path: str | Path | None) -> dict[int, tuple[float, float, float]] | None:
    """Прочитать набор весов кандидата: {"1": [p, c, ch], ...}."""
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    weights: dict[int, tuple[float, float, float]] = {}
    for horizon, values in data.items():
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(
                f"Веса для горизонта {horizon} должны быть тройкой [prophet, catboost, chronos]"
            )
        a, b, c = (float(v) for v in values)
        weights[int(horizon)] = (a, b, c)
    if not weights:
        raise ValueError(f"{path}: пустой набор весов")
    return weights


def recompute_ensemble(
    oof: pd.DataFrame, test: pd.DataFrame, weights=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Пересобрать RegimeAware-ансамбль из базовых рук — и на OOF, и на тесте."""
    if weights is None:
        weights = resolve_default_weights()
    blender = _blender(weights)
    oof = oof.copy()
    prediction, alpha = blender.predict(oof)
    oof[SUBMISSION_COLUMN] = prediction.clip(lower=1e-6)
    oof["regime_alpha"] = alpha
    test = test.copy()
    gate = test.copy()
    gate["lead_months"] = SUBMISSION_LEAD
    prediction, alpha = blender.predict(gate)
    test[SUBMISSION_COLUMN] = prediction.clip(lower=1e-6)
    test["regime_alpha"] = alpha
    return oof, test


def pooled_metrics(oof: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """Боевая таблица метрик по горизонтам (та же функция, что в пайплайне)."""
    from src.forecasting import horizon_metric_table

    table = horizon_metric_table(oof, horizons)
    return table.loc[table["fold"].eq("pooled") & table["scope"].eq("exact")].copy()


DEFAULT_WEIGHTS_FILE = ROOT / "configs" / "candidate_oof_optimum.json"


def resolve_default_weights() -> dict[int, tuple[float, float, float]] | None:
    """Return the measured OOF candidate when it is present and valid."""
    if not DEFAULT_WEIGHTS_FILE.exists():
        return None
    return load_weights(DEFAULT_WEIGHTS_FILE)


def compare(
    candidate: pd.DataFrame,
    baseline: dict[int, float],
    *,
    label: str = "",
    universe_match: bool = True,
) -> list[dict[str, Any]]:
    """Построчное сравнение кандидата с боевой линией по горизонтам."""
    rows: list[dict[str, Any]] = []
    ra = candidate.loc[candidate["model"].eq("regime_aware")]
    prophet = candidate.loc[candidate["model"].eq("prophet")]
    for row in ra.sort_values("horizon").itertuples():
        horizon = int(row.horizon)
        base = baseline.get(horizon)
        prophet_mae = (
            float(prophet.loc[prophet["horizon"].eq(horizon), "MAE"].iloc[0])
            if not prophet.loc[prophet["horizon"].eq(horizon)].empty
            else None
        )
        rows.append(
            {
                "horizon": horizon,
                "n": int(row.n),
                "candidate_mae": float(row.MAE),
                "baseline_mae": base,
                "delta_pct": (100.0 * (base - float(row.MAE)) / base) if base else None,
                "prophet_mae": prophet_mae,
                "vs_prophet_pct": (100.0 * (prophet_mae - float(row.MAE)) / prophet_mae)
                if prophet_mae
                else None,
                "verdict": (
                    "лучше"
                    if base and float(row.MAE) < base
                    else "хуже"
                    if base and float(row.MAE) > base
                    else "ровно"
                ),
                "baseline_label": label,
                "universe_match": universe_match,
            }
        )
    return rows


def run_from_cache(
    config: Any,
    *,
    cache_dir: str | Path = "reports/artifacts/",
    fast: bool = False,
    dry_run: bool = False,
    weights: dict[int, Sequence[float]] | None = None,
    tag: str | None = None,
    fast_submission: bool = False,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Пересобрать ансамбль/метрики/сабмит из кэша базовых прогнозов."""
    started = perf_counter()
    before = snapshot_protected()
    horizons = list(horizons or config.validation.horizons)

    out_dir = FAST_DIR if fast else _abs(config.paths.artifacts)
    figures = (FAST_DIR / "figures") if fast else _abs(config.changepoint_detection.figures)
    _assert_isolated(out_dir, fast=fast, fast_dir=FAST_DIR)
    _assert_isolated(figures, fast=fast, fast_dir=FAST_DIR)

    oof, test = load_cached_predictions(cache_dir, fast_mode=fast)
    mo_count = int(oof["mo"].nunique())
    if fast:
        subset_path = ROOT / "configs" / "benchmark_mo_150.json"
        if not subset_path.exists():
            raise FileNotFoundError(f"Нет списка подвыборки: {subset_path}")
        mo_list = json.loads(subset_path.read_text(encoding="utf-8"))["mo_list"]
        missing = sorted(set(mo_list) - set(oof["mo"].unique()))
        if missing:
            raise ValueError(
                f"В кэше нет {len(missing)} МО из бенчмарка, например: {missing[:3]}. "
                "Бенчмарк должен быть стратифицирован по оцениваемому универсуму: "
                "py -3.12 scripts/create_benchmark_subset.py --eval-universe auto"
            )
        oof = oof.loc[oof["mo"].isin(mo_list)].copy()
        test = test.loc[test["mo"].isin(mo_list)].copy()
        mo_count = int(oof["mo"].nunique())
        # изоляция: производные файлы не должны терять привязку к подвыборке
        assert mo_count == 150, f"бенчмарк должен покрывать 150 МО, получено {mo_count}"
        LOGGER.info("FAST: подвыборка %d МО из %s", mo_count, subset_path.name)

    effective_weights = weights if weights is not None else resolve_default_weights()
    raw_oof, raw_test = oof.copy(), test.copy()
    oof, test = recompute_ensemble(oof, test, effective_weights)
    table = pooled_metrics(oof, horizons)
    # Сравниваем с заводскими весами, а не с самим кандидатом.
    from src.ensemble import RegimeAwareBlender

    legacy_oof, _ = recompute_ensemble(
        raw_oof,
        raw_test,
        dict(RegimeAwareBlender.DEFAULT_WEIGHTS),
    )
    legacy_table = pooled_metrics(legacy_oof, horizons)
    legacy_rows = legacy_table.loc[legacy_table["model"].eq("regime_aware")]
    baseline = {int(row.horizon): float(row.MAE) for row in legacy_rows.itertuples()}
    baseline_label = "factory weights (до интеграции candidate_oof_optimum.json)"
    universe_match = True
    written_baseline = None
    comparison = compare(table, baseline, label=baseline_label, universe_match=universe_match)
    elapsed_ensemble = perf_counter() - started

    written: dict[str, str] = {}
    if not dry_run:
        from src.forecasting import metric_table, save_horizon_metrics

        out_dir.mkdir(parents=True, exist_ok=True)
        save_horizon_metrics(oof, out_dir, figures, horizons, mirror_root=not fast)
        written["metrics.json"] = _rel(out_dir / "metrics.json")
        written["tables.md"] = _rel(out_dir / "tables.md")
        derived = metric_table(oof.loc[oof["lead_months"].eq(1)])
        pooled = table.copy()
        pooled["fold"] = pooled["horizon"].map(lambda h: f"h={h}")
        pd.concat([derived, pooled], ignore_index=True).to_csv(
            out_dir / "forecast_metrics.csv", index=False
        )
        written["forecast_metrics.csv"] = _rel(out_dir / "forecast_metrics.csv")
        for name in ("metrics.json", "tables.md", "forecast_metrics.csv"):
            _assert_isolated(out_dir / name, fast=fast, fast_dir=FAST_DIR)

    if tag and not dry_run:
        report = {
            "tag": tag,
            "fast": fast,
            "mo_count": mo_count,
            "weights": None
            if effective_weights is None
            else {str(k): list(v) for k, v in effective_weights.items()},
            "weights_source": "explicit --weights-json"
            if weights is not None
            else str(DEFAULT_WEIGHTS_FILE),
            "cache": cached_paths(cache_dir),
            "comparison": comparison,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        exp_dir = FAST_DIR / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / f"{tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written["experiment"] = _rel(exp_dir / f"{tag}.json")

    submission_info: dict[str, Any] = {"built": False}
    if fast and not fast_submission:
        submission_info["skipped"] = "быстрый режим не собирает сабмит (см. --fast-submission)"
    if dry_run:
        submission_info["skipped"] = "dry-run: запись не выполнялась"
    if (not fast or fast_submission) and not dry_run:
        submission_info = _build_submission(config, test, fast=fast, dry_run=dry_run)
    elif (not fast) and dry_run:
        submission_info = _build_submission(config, test, fast=fast, dry_run=True)

    if not fast and not dry_run:
        _update_protocol(config, mo_count=mo_count, submission=submission_info)

    after = snapshot_protected()
    _enforce_protected(before, after, fast=fast)

    summary = {
        "mode": "from_cache",
        "fast": fast,
        "dry_run": dry_run,
        "mo_count": mo_count,
        "oof_rows": int(len(oof)),
        "comparison": comparison,
        "written": written,
        "submission": submission_info,
        "seconds_total": round(perf_counter() - started, 3),
        "seconds_ensemble": round(elapsed_ensemble, 3),
        "guardrail": {
            "protected_unchanged": (before == after) if fast else True,
            "fast_dir": _rel(FAST_DIR),
        },
        "baseline": {
            "label": baseline_label,
            "same_universe": universe_match,
            "frozen_here": written_baseline,
            "is_baseline_run": False,
        },
    }
    return summary


def _build_submission(
    config: Any, test: pd.DataFrame, *, fast: bool, dry_run: bool
) -> dict[str, Any]:
    """Собрать сабмит боевым валидатором.

    В быстром режиме цель — ``reports/fast_run/submission_150.csv``.
    В dry-run запись идёт в изолированный scratch и удаляется: так проверяется
    байтовая идентичность, не трогая боевые пути.
    """
    from src.submission import SubmissionValidator

    roster = sorted(test["mo"].dropna().astype(str).unique().tolist())
    if fast:
        output, root_output = FAST_DIR / "submission_150.csv", FAST_DIR / "submission_150.csv"
        expected_entities, expected_rows = 150, 150 * 6
    else:
        output = _abs(config.paths.artifacts).parent / "submission.csv"
        root_output = ROOT / "submission.csv"
        expected_entities, expected_rows = len(roster), len(roster) * 6

    if dry_run:
        scratch = FAST_DIR / "_dryrun" / "submission.csv"
        _assert_isolated(scratch, fast=True, fast_dir=FAST_DIR)
        validator = SubmissionValidator(scratch, scratch)
        frame = validator.build(
            test,
            prediction_column=SUBMISSION_COLUMN,
            expected_rows=expected_rows,
            expected_entities=expected_entities,
            entity_universe=roster,
        )
        target = output if output.exists() else None
        digest = _md5(scratch)
        reference = _md5(target) if target is not None else _md5(ROOT / "submission.csv")
        scratch.unlink()
        scratch.parent.rmdir()
        return {
            "built": False,
            "dry_run": True,
            "would_write": _rel(output),
            "rows": int(len(frame)),
            "entities": int(frame["mo"].nunique()),
            "md5": digest,
            "reference_md5": reference,
            "matches_expected": digest == EXPECTED_SUBMISSION_MD5,
            "matches_reference": digest == reference,
        }

    frame = SubmissionValidator(output, root_output).build(
        test,
        prediction_column=SUBMISSION_COLUMN,
        expected_rows=expected_rows,
        expected_entities=expected_entities,
        entity_universe=roster,
    )
    digest = _md5(output)
    return {
        "built": True,
        "rows": int(len(frame)),
        "entities": int(frame["mo"].nunique()),
        "md5": digest,
        "expected_md5": EXPECTED_SUBMISSION_MD5,
        "matches_expected": digest == EXPECTED_SUBMISSION_MD5,
        "path": _rel(output),
    }


def _update_protocol(config: Any, *, mo_count: int, submission: dict[str, Any]) -> None:
    """Дописать в протокол честную фиксацию режима воспроизведения."""
    path = _abs(config.paths.artifacts) / "submission_protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(
        {
            "execution_mode": "from_cache",
            "base_models_source": "frozen_origin_verified_cache",
            "ensemble_model": "RegimeAware Ensemble",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mo_count": mo_count,
            "submission_rows": submission.get("rows"),
            "submission_md5": submission.get("md5"),
            "submission_matches_expected_md5": submission.get("matches_expected"),
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_comparison(rows: Sequence[dict[str, Any]]) -> str:
    """Короткая таблица для консоли."""
    lines = []
    if rows:
        label = rows[0].get("baseline_label") or "—"
        mismatch = not rows[0].get("universe_match", True)
        lines.append(
            f"базовая линия: {label}"
            + ("   ⚠ РАЗНЫЙ УНИВЕРСУМ: сравнение некорректно" if mismatch else "")
        )
    lines.append(
        f"{'h':>3} {'n':>7} {'боевая':>9} {'кандидат':>9} {'Δ':>8} {'vs Prophet':>11}  вывод"
    )
    for row in rows:
        base = f"{row['baseline_mae']:.0f}" if row["baseline_mae"] else "—"
        delta = f"{row['delta_pct']:+.2f}%" if row["delta_pct"] is not None else "—"
        vs = f"{row['vs_prophet_pct']:+.2f}%" if row["vs_prophet_pct"] is not None else "—"
        lines.append(
            f"{row['horizon']:>3} {row['n']:>7} {base:>9} {row['candidate_mae']:>9.0f} "
            f"{delta:>8} {vs:>11}  {row['verdict']}"
        )
    return "\n".join(lines)
