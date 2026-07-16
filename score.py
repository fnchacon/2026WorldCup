#!/usr/bin/env python3
"""Reconcile the published 2026 World Cup group and R32 scorecards."""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results_2026.csv"
LEDGER = ROOT / "predictions" / "bracket_vs_model.csv"
GROUP_SCORECARD = ROOT / "predictions" / "group_stage_scorecard.csv"
R32_PREDICTIONS = ROOT / "predictions" / "r32_game_predictions.csv"
R32_SCORECARD = ROOT / "predictions" / "r32_scorecard.csv"

UNIFORM_RPS_BASELINE = 0.215
GROUP_POINTS = (3, 1)
R32_POINTS = (4, 2)

ALIASES = {
    "bosnia": "bosnia and herzegovina",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "cabo verde": "cape verde",
    "cape verde": "cape verde",
    "cote d'ivoire": "ivory coast",
    "ivory coast": "ivory coast",
    "czechia": "czech republic",
    "czech republic": "czech republic",
    "s. korea": "south korea",
    "south korea": "south korea",
    "turkiye": "turkey",
    "turkey": "turkey",
    "usa": "united states",
    "united states": "united states",
}

MONTH_SCORE = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

EXPECTED = {
    "group_md1_accuracy_pct": 50.0,
    "group_md2_accuracy_pct": 70.8,
    "group_md3_accuracy_pct": 54.0,
    "group_cumulative_accuracy_pct": 58.3,
    "group_md1_rps": 0.183,
    "group_md2_rps": 0.133,
    "group_md3_rps": 0.132,
    "group_cumulative_rps": 0.149,
    "group_overrides": (29, 6, 12, 11, -6),
    "group_override_round_net": (1, -2, -5),
    "group_disagreements": (16, 10, 4, 2),
    "r32_top_pick": (12, 16, 75.0),
    "r32_rps": 0.124,
    "r32_overrides": (6, 2, 2, 2, 0),
    "r32_head_to_head": (10, 16, 9, 16),
    "r32_exact": (3, 2),
    "r32_draws": (3, 2, "Germany-Paraguay"),
}

R32_NEWLY_FILLED = {
    "France-Sweden",
    "Côte d'Ivoire-Norway",
    "Mexico (H)-Ecuador",
    "USA (H)-Bosnia",
    "Belgium-Senegal",
    "England-DR Congo",
    "Portugal-Croatia",
    "Spain-Austria",
    "Switzerland-Algeria",
    "Argentina-Cabo Verde",
    "Australia-Egypt",
    "Colombia-Ghana",
}


@dataclass
class MatchResult:
    date: datetime
    home: str
    away: str
    home_score: int
    away_score: int
    row_index: int


def read_csv(path: Path) -> list[dict[str, str]]:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open(newline="", encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(str(path), b"", 0, 1, "unable to decode")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def deaccent(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def norm_team(value: str) -> str:
    value = value.replace("\ufeff", "").strip()
    value = re.sub(r"\s*\(H\)\s*$", "", value)
    value = value.replace("’", "'")
    value = deaccent(value).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return ALIASES.get(value, value)


def team_pair_key(home: str, away: str) -> frozenset[str]:
    return frozenset((norm_team(home), norm_team(away)))


def parse_date(value: str) -> datetime:
    value = value.strip()
    for fmt in ("%m/%d/%y", "%Y-%m-%d", "%d-%b"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(year=2026) if fmt == "%d-%b" else parsed
        except ValueError:
            pass
    raise ValueError(f"Unparseable date: {value!r}")


def parse_score(value: str) -> tuple[int, int] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    value = value.replace("–", "-").replace("—", "-")
    if re.fullmatch(r"\d+-\d+", value):
        h, a = value.split("-")
        return int(h), int(a)

    # Spreadsheet-damaged scores such as Jan-00 (= 1-0) or 1-Feb (= 2-1).
    m = re.fullmatch(r"([A-Za-z]{3})-00", value)
    if m and m.group(1).lower() in MONTH_SCORE:
        return MONTH_SCORE[m.group(1).lower()], 0
    m = re.fullmatch(r"(\d+)-([A-Za-z]{3})", value)
    if m and m.group(2).lower() in MONTH_SCORE:
        return MONTH_SCORE[m.group(2).lower()], int(m.group(1))
    return None


def fmt_score(score: tuple[int, int] | None) -> str:
    return "" if score is None else f"{score[0]}-{score[1]}"


def outcome(score: tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "home"
    if score[0] < score[1]:
        return "away"
    return "draw"


def outcome_from_probs(row: dict[str, str]) -> str:
    probs = {
        "home": float(row["p_home"]),
        "draw": float(row["p_draw"]),
        "away": float(row["p_away"]),
    }
    return max(probs, key=probs.get)


def score_points(pred: tuple[int, int], actual: tuple[int, int], exact_pts: int, result_pts: int) -> int:
    if pred == actual:
        return exact_pts
    if outcome(pred) == outcome(actual):
        return result_pts
    return 0


def split_match(match: str) -> tuple[str, str]:
    match = match.replace("Ð", "-").replace("�", "-").replace("–", "-").strip()
    parts = match.split("-")
    if len(parts) != 2:
        raise ValueError(f"Cannot split match name: {match!r}")
    return parts[0].strip(), parts[1].strip()


def result_label(model_right: bool, me_right: bool) -> str:
    if model_right and me_right:
        return "Both"
    if model_right:
        return "Model"
    if me_right:
        return "Me"
    return "Neither"


def model_pick_label(home: str, away: str, model_score: tuple[int, int]) -> str:
    out = outcome(model_score)
    if out == "home":
        return home
    if out == "away":
        return away
    return "Draw"


def load_results() -> list[MatchResult]:
    results = []
    for i, row in enumerate(read_csv(RESULTS)):
        if row.get("home_score", "") == "" or row.get("away_score", "") == "":
            continue
        results.append(
            MatchResult(
                date=parse_date(row["date"]),
                home=row["home_team"],
                away=row["away_team"],
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                row_index=i,
            )
        )
    return results


def actual_for_match(results_by_pair: dict[frozenset[str], MatchResult], home: str, away: str) -> tuple[int, int]:
    result = results_by_pair[team_pair_key(home, away)]
    if norm_team(result.home) == norm_team(home):
        return result.home_score, result.away_score
    return result.away_score, result.home_score


def load_ledger() -> list[dict[str, str]]:
    raw_rows = read_csv(LEDGER)
    rows = []
    for row in raw_rows:
        if row.get("match") in {"Match", "", None}:
            continue
        if "match" not in row:
            continue
        rows.append(row)
    return rows


def clean_and_score_ledger(rows: list[dict[str, str]], results: list[MatchResult]) -> list[dict[str, str]]:
    results_by_pair = {team_pair_key(r.home, r.away): r for r in results}
    cleaned = []
    for row in rows:
        scoring_row = len(cleaned) < 88
        home, away = split_match(row["match"])
        model_score = parse_score(row.get("model_proj_score", ""))
        bracket_score = parse_score(row.get("bracket_pick_score", ""))
        actual_score = None
        if team_pair_key(home, away) in results_by_pair:
            actual_score = actual_for_match(results_by_pair, home, away)
        else:
            actual_score = parse_score(row.get("actual_result", ""))

        overrode = "Yes" if row.get("overrode", "").strip().lower() == "yes" else ""
        out = {
            "match": f"{home}-{away}",
            "model_proj_score": fmt_score(model_score),
            "model_pick": row.get("model_pick", "").strip(),
            "bracket_pick_score": fmt_score(bracket_score),
            "overrode": overrode,
            "actual_result": fmt_score(actual_score),
            "winner_right": row.get("winner_right", "").strip().replace("You", "Me"),
            "exact_score_right": row.get("exact_score_right", "").strip().replace("You", "Me"),
            "override_effect": row.get("override_effect", "").strip().replace("—", ""),
        }

        if model_score is not None:
            out["model_pick"] = model_pick_label(home, away, model_score)

        if scoring_row and model_score is not None and bracket_score is not None and actual_score is not None:
            model_result_right = outcome(model_score) == outcome(actual_score)
            me_result_right = outcome(bracket_score) == outcome(actual_score)
            model_exact = model_score == actual_score
            me_exact = bracket_score == actual_score
            out["winner_right"] = result_label(model_result_right, me_result_right)
            out["exact_score_right"] = result_label(model_exact, me_exact)
            if overrode:
                exact_pts, result_pts = R32_POINTS if len(cleaned) >= 72 else GROUP_POINTS
                delta = score_points(bracket_score, actual_score, exact_pts, result_pts) - score_points(
                    model_score, actual_score, exact_pts, result_pts
                )
                if delta > 0:
                    out["override_effect"] = "Helped"
                elif delta < 0:
                    out["override_effect"] = "Hurt"
                else:
                    out["override_effect"] = "Neutral"
            else:
                out["override_effect"] = ""
        cleaned.append(out)
    return cleaned


def assign_group_matchdays(results: list[MatchResult]) -> dict[frozenset[str], int]:
    group_results = results[:72]
    by_team = defaultdict(list)
    for r in group_results:
        by_team[norm_team(r.home)].append(r)
        by_team[norm_team(r.away)].append(r)

    ranks = {}
    for games in by_team.values():
        for rank, game in enumerate(sorted(games, key=lambda g: (g.date, g.row_index)), start=1):
            ranks.setdefault(team_pair_key(game.home, game.away), rank)

    counts = Counter(ranks[team_pair_key(r.home, r.away)] for r in group_results)
    if counts != {1: 24, 2: 24, 3: 24}:
        raise AssertionError(f"Bad matchday bucketing: {dict(counts)}")
    return ranks


def group_scorecard_values() -> dict[str, tuple[float, float, int]]:
    values = {}
    for row in read_csv(GROUP_SCORECARD):
        values[row["round"]] = (
            float(row["outcome_accuracy_pct"]),
            float(row["rps"]),
            int(row["matches"]),
        )
    return values


def override_summary(rows: list[dict[str, str]], start: int, stop: int) -> tuple[int, int, int, int, int]:
    counts = Counter()
    for row in rows[start:stop]:
        if row["overrode"] != "Yes":
            continue
        counts["total"] += 1
        effect = row["override_effect"]
        if effect == "Helped":
            counts["helped"] += 1
        elif effect == "Hurt":
            counts["hurt"] += 1
        elif effect == "Neutral":
            counts["neutral"] += 1
    return (
        counts["total"],
        counts["helped"],
        counts["hurt"],
        counts["neutral"],
        counts["helped"] - counts["hurt"],
    )


def override_round_net(rows: list[dict[str, str]]) -> tuple[int, int, int]:
    nets = []
    for start in (0, 24, 48):
        _, helped, hurt, _, _ = override_summary(rows, start, start + 24)
        nets.append(helped - hurt)
    return tuple(nets)


def disagreement_summary(rows: list[dict[str, str]], start: int, stop: int) -> tuple[int, int, int, int]:
    model_right = me_right = both_wrong = total = 0
    for row in rows[start:stop]:
        model = parse_score(row["model_proj_score"])
        me = parse_score(row["bracket_pick_score"])
        actual = parse_score(row["actual_result"])
        if model is None or me is None or actual is None or outcome(model) == outcome(me):
            continue
        total += 1
        mr = outcome(model) == outcome(actual)
        br = outcome(me) == outcome(actual)
        if mr:
            model_right += 1
        elif br:
            me_right += 1
        else:
            both_wrong += 1
    return total, model_right, me_right, both_wrong


def rps(row: dict[str, str], actual: str) -> float:
    p_home = float(row["p_home"])
    p_draw = float(row["p_draw"])
    o_home = 1.0 if actual == "home" else 0.0
    o_draw = 1.0 if actual == "draw" else 0.0
    return 0.5 * ((p_home - o_home) ** 2 + ((p_home + p_draw) - (o_home + o_draw)) ** 2)


def r32_probability_score(results: list[MatchResult]) -> tuple[int, int, float, list[str]]:
    results_by_pair = {team_pair_key(r.home, r.away): r for r in results}
    predictions = read_csv(R32_PREDICTIONS)
    right = 0
    total_rps = 0.0
    missing = []
    for row in predictions:
        try:
            actual_score = actual_for_match(results_by_pair, row["home"], row["away"])
        except KeyError:
            missing.append(f"{row['home']}-{row['away']}")
            continue
        actual_outcome = outcome(actual_score)
        if outcome_from_probs(row) == actual_outcome:
            right += 1
        total_rps += rps(row, actual_outcome)
    total = len(predictions) - len(missing)
    return right, total, total_rps / total, missing


def r32_head_to_head(rows: list[dict[str, str]]) -> tuple[int, int, int, int, int, int]:
    me_result = model_result = me_exact = model_exact = total = 0
    for row in rows[72:88]:
        model = parse_score(row["model_proj_score"])
        me = parse_score(row["bracket_pick_score"])
        actual = parse_score(row["actual_result"])
        if model is None or me is None or actual is None:
            continue
        total += 1
        model_result += outcome(model) == outcome(actual)
        me_result += outcome(me) == outcome(actual)
        model_exact += model == actual
        me_exact += me == actual
    return me_result, total, model_result, total, model_exact, me_exact


def r32_draws(rows: list[dict[str, str]]) -> tuple[int, int, list[str]]:
    draw_games = []
    called = 0
    missed = []
    for row in rows[72:88]:
        model = parse_score(row["model_proj_score"])
        actual = parse_score(row["actual_result"])
        if model is None or actual is None or outcome(actual) != "draw":
            continue
        draw_games.append(row["match"])
        if outcome(model) == "draw":
            called += 1
        else:
            missed.append(row["match"])
    return len(draw_games), called, missed


def r32_projection_discrepancies(rows: list[dict[str, str]]) -> list[str]:
    by_pair = {team_pair_key(row["match"].split("-")[0], row["match"].split("-")[1]): row for row in rows[72:88]}
    out = []
    for pred in read_csv(R32_PREDICTIONS):
        key = team_pair_key(pred["home"], pred["away"])
        if key not in by_pair:
            continue
        ledger_score = parse_score(by_pair[key]["model_proj_score"])
        pred_score = parse_score(pred["proj_score"])
        if ledger_score != pred_score:
            same = "same outcome" if outcome(ledger_score) == outcome(pred_score) else "different outcome"
            out.append(
                f"{pred['home']}-{pred['away']}: ledger {fmt_score(ledger_score)} vs predictions {fmt_score(pred_score)} ({same})"
            )
    return out


def validate_new_r32_rows(rows: list[dict[str, str]]) -> list[str]:
    problems = []
    for row in rows[72:88]:
        if row["match"] not in R32_NEWLY_FILLED:
            continue
        model = parse_score(row["model_proj_score"])
        me = parse_score(row["bracket_pick_score"])
        actual = parse_score(row["actual_result"])
        if model is None or me is None or actual is None:
            problems.append(f"{row['match']}: missing score")
            continue
        expected_winner = result_label(outcome(model) == outcome(actual), outcome(me) == outcome(actual))
        expected_exact = result_label(model == actual, me == actual)
        model_pts = score_points(model, actual, *R32_POINTS)
        me_pts = score_points(me, actual, *R32_POINTS)
        expected_effect = ""
        if row["overrode"] == "Yes":
            expected_effect = "Helped" if me_pts > model_pts else "Hurt" if me_pts < model_pts else "Neutral"
        for field, expected in (
            ("winner_right", expected_winner),
            ("exact_score_right", expected_exact),
            ("override_effect", expected_effect),
        ):
            if row[field] != expected:
                problems.append(f"{row['match']}: {field}={row[field]!r}, expected {expected!r}")
    return problems


def passfail(name: str, actual, expected) -> bool:
    ok = actual == expected
    print(f"{'PASS' if ok else 'FAIL'} {name}: {actual} (expected {expected})")
    return ok


def passfail_float(name: str, actual: float, expected: float, places: int = 3) -> bool:
    rounded = round(actual, places)
    return passfail(name, rounded, expected)


def write_scorecards(r32_right: int, r32_total: int, r32_avg_rps: float) -> None:
    rows = [
        {
            "round": "Round of 32",
            "matches": str(r32_total),
            "outcome_accuracy_pct": f"{r32_right / r32_total * 100:.1f}",
            "rps": f"{r32_avg_rps:.3f}",
            "notes": f"top-pick/argmax accuracy; vs uniform baseline RPS {UNIFORM_RPS_BASELINE:.3f}",
        }
    ]
    write_csv(R32_SCORECARD, ["round", "matches", "outcome_accuracy_pct", "rps", "notes"], rows)


def main() -> int:
    results = load_results()
    assign_group_matchdays(results)
    ledger_rows = clean_and_score_ledger(load_ledger(), results)
    write_csv(
        LEDGER,
        [
            "match",
            "model_proj_score",
            "model_pick",
            "bracket_pick_score",
            "overrode",
            "actual_result",
            "winner_right",
            "exact_score_right",
            "override_effect",
        ],
        ledger_rows,
    )

    group = group_scorecard_values()
    r32_right, r32_total, r32_avg_rps, missing = r32_probability_score(results)
    write_scorecards(r32_right, r32_total, r32_avg_rps)

    print("Reconciliation report")
    print("=====================")
    ok = True
    ok &= passfail_float("Group MD1 top-pick accuracy pct", group["Matchday 1"][0], EXPECTED["group_md1_accuracy_pct"], 1)
    ok &= passfail_float("Group MD2 top-pick accuracy pct", group["Matchday 2"][0], EXPECTED["group_md2_accuracy_pct"], 1)
    ok &= passfail_float("Group MD3 top-pick accuracy pct", group["Matchday 3"][0], EXPECTED["group_md3_accuracy_pct"], 1)
    ok &= passfail_float(
        "Group cumulative top-pick accuracy pct", group["Cumulative"][0], EXPECTED["group_cumulative_accuracy_pct"], 1
    )
    ok &= passfail_float("Group MD1 RPS", group["Matchday 1"][1], EXPECTED["group_md1_rps"])
    ok &= passfail_float("Group MD2 RPS", group["Matchday 2"][1], EXPECTED["group_md2_rps"])
    ok &= passfail_float("Group MD3 RPS", group["Matchday 3"][1], EXPECTED["group_md3_rps"])
    ok &= passfail_float("Group cumulative RPS", group["Cumulative"][1], EXPECTED["group_cumulative_rps"])
    ok &= passfail("Group overrides total/helped/hurt/neutral/net", override_summary(ledger_rows, 0, 72), EXPECTED["group_overrides"])
    ok &= passfail("Group override net by matchday", override_round_net(ledger_rows), EXPECTED["group_override_round_net"])
    ok &= passfail("Group disagreements total/model/me/both-wrong", disagreement_summary(ledger_rows, 0, 72), EXPECTED["group_disagreements"])

    r32_acc = r32_right / r32_total * 100
    ok &= passfail("R32 top-pick right/total/accuracy pct", (r32_right, r32_total, round(r32_acc, 1)), EXPECTED["r32_top_pick"])
    ok &= passfail_float("R32 RPS", r32_avg_rps, EXPECTED["r32_rps"])
    ok &= passfail("R32 overrides total/helped/hurt/neutral/net", override_summary(ledger_rows, 72, 88), EXPECTED["r32_overrides"])
    me_r, me_t, model_r, model_t, model_exact, me_exact = r32_head_to_head(ledger_rows)
    ok &= passfail("R32 modal-result head-to-head me/model", (me_r, me_t, model_r, model_t), EXPECTED["r32_head_to_head"])
    ok &= passfail("R32 exact scores model/me", (model_exact, me_exact), EXPECTED["r32_exact"])
    draw_total, draw_called, draw_missed = r32_draws(ledger_rows)
    ok &= passfail("R32 draws total/model-scoreline-called/missed", (draw_total, draw_called, ", ".join(draw_missed)), EXPECTED["r32_draws"])

    new_row_problems = validate_new_r32_rows(ledger_rows)
    ok &= passfail("R32 newly-filled ledger row inconsistencies", len(new_row_problems), 0)
    for problem in new_row_problems:
        print(f"  - {problem}")

    if missing:
        ok = False
        print("FAIL R32 missing actuals:", ", ".join(missing))

    discrepancies = r32_projection_discrepancies(ledger_rows)
    if discrepancies:
        print("\nDocumented ledger-vs-probability projection differences:")
        for item in discrepancies:
            print(f"  - {item}")

    print(f"\nWrote {R32_SCORECARD.relative_to(ROOT)}")
    print(f"Updated {LEDGER.relative_to(ROOT)} with computed result/effect columns")
    print("\nOVERALL", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
