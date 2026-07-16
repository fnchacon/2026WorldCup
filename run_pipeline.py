#!/usr/bin/env python3
"""
run_pipeline.py — one-command refit -> predict -> simulate.

Usage:
    python run_pipeline.py            # use cached data/results_2026.csv
    python run_pipeline.py --update   # pull latest results from upstream first

After each matchday, append the new results to data/results_2026.csv
(or let the upstream martj42 dataset sync) and re-run. Because the model is
recency-weighted, World Cup matches carry the highest weight of anything in the
sample, so the posteriors sharpen fast as the tournament unfolds.
"""
import argparse
import csv
import math
from pathlib import Path


R32_PREDICTIONS = Path("predictions/r32_game_predictions.csv")


def poisson_goal_mode(lam, maxg=10):
    if math.isclose(lam, math.floor(lam)):
        return math.floor(lam)
    probs = []
    p = math.exp(-lam)
    probs.append(p)
    for k in range(1, maxg):
        p *= lam / k
        probs.append(p)
    return max(range(maxg), key=lambda k: probs[k])


def modal_scoreline_from_xg(xg_home, xg_away, maxg=10):
    """proj_score is the modal (most-likely) exact scoreline, not rounded xG."""
    return f"{poisson_goal_mode(xg_home, maxg)}-{poisson_goal_mode(xg_away, maxg)}"


def regenerate_r32_proj_score(path=R32_PREDICTIONS):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    changed = []
    for row in rows:
        old = row["proj_score"]
        row["proj_score"] = modal_scoreline_from_xg(float(row["xg_home"]), float(row["xg_away"]))
        if row["proj_score"] != old:
            changed.append((f"{row['home']}-{row['away']}", old, row["proj_score"]))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Regenerated {path}")
    if changed:
        print("Rows changed:")
        for match, old, new in changed:
            print(f"  - {match}: {old} -> {new}")
    else:
        print("Rows changed: none")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true', help='pull latest upstream results')
    ap.add_argument('--sims', type=int, default=200, help='posterior draws for Bayesian table')
    ap.add_argument(
        '--regenerate-r32-proj-score',
        action='store_true',
        help='rewrite predictions/r32_game_predictions.csv proj_score from modal Poisson goal counts',
    )
    args = ap.parse_args()

    if args.regenerate_r32_proj_score:
        regenerate_r32_proj_score()
        return

    import wc2026_model as wc

    rows = wc.load_matches(download=args.update)
    print(f"Fitting on {len(rows)} weighted international matches...")
    P = wc.fit(rows)
    print(f"  converged={P['conv']}  home x{2.718**P['hadv']:.2f}  "
          f"major-goal x{2.718**P['gam']:.2f}  Dixon-Coles rho={P['rho']:.3f}")

    # Per-game predictions for any fixture list live in predictions/.
    # The Bayesian bracket table is produced by bayes_layer (see README).
    print("\nModel ready. To regenerate the title-odds table, import bayes_layer and call")
    print("champion_table(P, laplace_covariance(P, rows), bracket_order, hosts).")
    print("See README.md for the full worked example and the running scorecard.")


if __name__ == '__main__':
    main()
