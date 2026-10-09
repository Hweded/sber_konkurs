"""Генератор стратифицированной бенчмарк-выборки из 150 МО.

Зачем: полный OOF-цикл на 2 094 МО занимает ~40 минут и блокирует быструю проверку
гипотез (веса ансамбля, новые макро-признаки, пороги алертов). Фиксированная
репрезентативная подвыборка позволяет получить итерацию за 60-90 секунд на CPU.

Состав выборки (по умолчанию 150):
  * 3 обязательных якоря — Казань, Норильск, Суздаль (разные режимы расходов);
  * остальные 147 — стратифицированно по 4 квартилям средних расходов МО
    (`pd.qcut` по среднему уровню целевой переменной), примерно по 36-37 на квартиль,
    фиксированный ``random_state=42``.

Скрипт только читает панель и пишет JSON-список: боевые артефакты поставки
(`submission.csv`, `metrics.json`, `tables.md`) он не трогает по построению.

Пример:
    py -3.12 scripts/create_benchmark_subset.py
    py -3.12 scripts/create_benchmark_subset.py --n-mo 150 --seed 42 \
        --out configs/benchmark_mo_150.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "processed" / "supervised.parquet"
DEFAULT_OUT = ROOT / "configs" / "benchmark_mo_150.json"
DESCRIPTION = "Stratified 150 MO benchmark subset (Kazan, Norilsk, Suzdal + 4 spending quartiles)"

# Якоря заданы точным фрагментом имени: «Казан» подходит и Казанскому району Татарстана,
# поэтому свободный поиск по подстроке запрещён — иначе в выборку попадёт не тот объект.
ANCHORS: dict[str, str] = {
    "Казань": "город Казань",  # сервисный мегаполис-миллионник
    "Норильск": "город Норильск",  # высоковолатильный моногород, северный завоз
    "Суздаль": "Суздальский",  # туристический кластер, выраженная сезонность
}
QUARTILES = 4


def evaluation_universe(root: Path = ROOT) -> set[str] | None:
    """МО, которые реально оцениваются: пересечение OOF-панели и ростера сабмита.

    Стратификация по более широкой панели (``data/processed``) даёт МО, которых нет
    в OOF, — такой бенчмарк нельзя ни измерить, ни отправить.  Пересечение
    гарантирует, что каждая МО выборки имеет OOF-оценку и место в сабмите.
    """
    oof = root / "reports" / "predictions_oof.parquet"
    submission = root / "reports" / "predictions_submission.parquet"
    if not (oof.exists() and submission.exists()):
        return None
    left = set(pd.read_parquet(oof, columns=["mo"])["mo"].unique())
    right = set(pd.read_parquet(submission, columns=["mo"])["mo"].unique())
    return left & right


def resolve_anchors(mos: pd.Index | list[str]) -> dict[str, str]:
    """Найти каждый якорь строго: ровно одно совпадение, иначе — остановка."""
    out: dict[str, str] = {}
    for label, needle in ANCHORS.items():
        hits = [m for m in mos if needle.lower() in str(m).lower()]
        if len(hits) != 1:
            raise SystemExit(
                f"Якорь {label!r} не разрешён однозначно: найдено {len(hits)} совпадений "
                f"по {needle!r} -> {hits[:5]}. Уточните образец в ANCHORS."
            )
        out[label] = hits[0]
    return out


def load_levels(source: Path, target_col: str | None) -> tuple[pd.Series, str]:
    """Средний уровень расходов по каждому МО (ось стратификации) и имя целевой колонки."""
    if not source.exists():
        raise SystemExit(f"Панель не найдена: {source}")
    df = pd.read_parquet(source)
    if target_col is None:
        for candidate in ("target", "y"):
            if candidate in df.columns:
                target_col = candidate
                break
        else:
            raise SystemExit(
                f"В {source.name} нет ни 'target', ни 'y'. Колонки: {list(df.columns)[:12]}"
            )
    if "mo" not in df.columns:
        raise SystemExit(f"В {source.name} нет колонки 'mo'")
    if target_col not in df.columns:
        raise SystemExit(f"В {source.name} нет колонки {target_col!r}")
    levels = df.groupby("mo")[target_col].mean()
    levels = levels[levels > 0]
    return levels.rename("level"), target_col


def quartile_labels(levels: pd.Series) -> tuple[pd.Series, str]:
    """4 равных квартиля по уровню. При жёстких связях переходим на ранговый qcut."""
    try:
        q = pd.qcut(levels, QUARTILES, labels=False, duplicates="drop")
        if q.nunique() == QUARTILES:
            return q, "pd.qcut(level, 4)"
    except ValueError:
        pass
    q = pd.qcut(levels.rank(method="first"), QUARTILES, labels=False)
    return q, "pd.qcut(rank(level, 'first'), 4) [fallback: ties in level]"


def _rel(path: Path) -> str:
    """Путь относительно корня репозитория, если он внутри; иначе — как есть."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def even_sizes(total: int, parts: int) -> list[int]:
    """Разложить total на parts частей так, чтобы разница не превышала 1."""
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def build(
    source: Path,
    out: Path,
    n_mo: int,
    seed: int,
    target_col: str | None,
    universe: set[str] | None = None,
) -> dict:
    levels, resolved_col = load_levels(source, target_col)
    dropped = 0
    if universe is not None:
        before = len(levels)
        levels = levels.loc[levels.index.isin(universe)]
        dropped = before - len(levels)
        if len(levels) < n_mo:
            raise SystemExit(
                f"В оцениваемом универсуме только {len(levels)} МО — меньше запрошенных {n_mo}"
            )
    anchors = resolve_anchors(levels.index)
    anchor_list = [anchors[k] for k in ANCHORS]

    rest = levels.drop(index=anchor_list)
    want_rest = n_mo - len(anchor_list)
    if want_rest <= 0 or want_rest > len(rest):
        raise SystemExit(f"Невозможно отобрать {want_rest} МО из {len(rest)} доступных")

    q, method = quartile_labels(rest)
    sizes = even_sizes(want_rest, QUARTILES)

    picked: list[str] = []
    per_quartile: list[dict] = []
    for qi, k in enumerate(sizes):
        pool = rest.index[q == qi].tolist()
        if k > len(pool):
            raise SystemExit(f"Квартиль {qi}: запрошено {k}, доступно {len(pool)}")
        take = pd.Series(pool).sample(n=k, random_state=seed).tolist()
        picked.extend(take)
        per_quartile.append(
            {
                "quartile": qi + 1,
                "pool": len(pool),
                "picked": len(take),
                "level_min": round(float(rest.loc[pool].min()), 1),
                "level_max": round(float(rest.loc[pool].max()), 1),
            }
        )

    mo_list = anchor_list + sorted(picked)
    if len(mo_list) != n_mo or len(set(mo_list)) != n_mo:
        raise SystemExit(
            f"Итоговый список невалиден: {len(mo_list)} записей, {len(set(mo_list))} уникальных"
        )

    payload = {
        "n_mo": n_mo,
        "description": DESCRIPTION,
        "mo_list": mo_list,
        # --- происхождение (аддитивно; потребители читают три ключа выше) ---
        "seed": seed,
        "source": _rel(source),
        "target_column": resolved_col,
        "anchors": anchors,
        "evaluation_universe": {
            "size": int(len(levels) + len(anchor_list) + dropped),
            "restricted": universe is not None,
            "outside_universe_dropped": int(dropped),
            "source": "OOF ∩ submission roster" if universe is not None else "вся панель",
        },
        "stratification": {
            "method": method,
            "quartiles": per_quartile,
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"источник: {payload['source']}  (целевая колонка: {payload['target_column']})")
    print(f"стратификация: {method}   seed={seed}   n_mo={n_mo}")
    print("якоря:")
    for label, mo in anchors.items():
        print(f"  {label:9s} -> {mo}")
    print("квартили (средние расходы, ₽):")
    for pq in per_quartile:
        print(
            f"  Q{pq['quartile']}: отобрано {pq['picked']:3d} из {pq['pool']:4d}  "
            f"[{pq['level_min']} … {pq['level_max']}]"
        )
    print(f"итого: {len(mo_list)} МО, уникальных {len(set(mo_list))}")
    print(f"wrote {_rel(out)} ({out.stat().st_size} байт)")
    return payload


def verify(path: Path, n_mo: int) -> None:
    """Контрольные инварианты выборки — читается независимо от генератора."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mo = data["mo_list"]
    assert data["n_mo"] == n_mo == len(mo), "n_mo расходится с длиной списка"
    assert len(set(mo)) == len(mo), "в выборке есть дубликаты"
    for label, needle in ANCHORS.items():
        matched = [m for m in mo if needle.lower() in m.lower()]
        assert len(matched) == 1, f"якорь {label} не представлен ровно один раз"
    assert all(isinstance(m, str) and m for m in mo), "пустые имена МО"
    print(f"verify OK: {path} — {len(mo)} МО, 3 якоря на месте, дубликатов нет")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Стратифицированная бенчмарк-выборка 150 МО")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-mo", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-col", default=None, help="по умолчанию: target, иначе y")
    ap.add_argument("--verify-only", action="store_true", help="только проверить готовый файл")
    ap.add_argument(
        "--eval-universe",
        choices=("auto", "none"),
        default="auto",
        help="auto: стратифицировать только по МО, которые есть и в OOF, и в сабмите",
    )
    args = ap.parse_args(argv)

    if args.verify_only:
        verify(args.out, args.n_mo)
        return 0
    universe = evaluation_universe() if args.eval_universe == "auto" else None
    if args.eval_universe == "auto" and universe is None:
        print("внимание: OOF/сабмит не найдены — стратификация по всей панели")
    build(args.source, args.out, args.n_mo, args.seed, args.target_col, universe=universe)
    verify(args.out, args.n_mo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
