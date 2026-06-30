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
import wc2026_model as wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true', help='pull latest upstream results')
    ap.add_argument('--sims', type=int, default=200, help='posterior draws for Bayesian table')
    args = ap.parse_args()

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
