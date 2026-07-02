"""
pool_optimizer.py — expected-points-maximizing predictions for a score-exact pool.

The base model (wc2026_model.py) reports the MODAL scoreline: the single most
likely result. That is the right answer to "what will happen?" but the WRONG
answer to "what should I bet?" in a pool that pays asymmetrically for exact
scores vs. correct results.

This layer takes the model's full score-probability matrix and instead reports
the scoreline that maximizes EXPECTED POOL POINTS:

    EV(predict i-j) = P(exact i-j) * exact_pts
                    + P(correct outcome, wrong score) * result_pts

Because exact_pts >> result_pts (3:1 in groups, up to 15:6 in the final), and
because a draw prediction collects nothing on a 2-1, the EV-optimal pick is
often NOT the modal score. It tends to be:
  - bolder than the model's hedge in lopsided games (chase the exact bonus),
  - and it shifts with the round's payoff ratio and your standing in the pool.

The honest forecasting model stays untouched. This is a SEPARATE betting layer
that sits on top of it. Keep both: one tells the truth, one wins the office pool.
"""
import numpy as np
from scipy.stats import poisson
import wc2026_model as wc

# Pool points: (exact_score_pts, correct_result_pts) by phase.
POINTS = {
    'group':  (3, 1),
    'r32':    (4, 2),
    'r16':    (5, 2),
    'qf':     (7, 3),
    'sf':     (9, 4),
    'final':  (15, 6),
}


def score_matrix(P, home, away, major=True, maxg=12):
    """Full normalized P(i-j) matrix from the model (with Dixon-Coles tau)."""
    lh, la = wc.lambdas(P, home, away, major=major)
    M = np.outer(poisson.pmf(range(maxg), lh), poisson.pmf(range(maxg), la))
    rho = P['rho']
    M[0, 0] *= 1 - lh*la*rho; M[0, 1] *= 1 + lh*rho
    M[1, 0] *= 1 + la*rho;    M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None); M /= M.sum()
    return M


def expected_points(M, i, j, exact_pts, result_pts):
    """EV in pool points of predicting scoreline i-j, given score matrix M."""
    maxg = M.shape[0]
    p_exact = M[i, j]
    # outcome of the predicted scoreline
    pred = 0 if i > j else (1 if i == j else 2)
    # probability the actual outcome matches the predicted outcome
    if pred == 0:      # predicted home win
        p_outcome = np.tril(M, -1).sum()
    elif pred == 1:    # predicted draw
        p_outcome = np.trace(M)
    else:              # predicted away win
        p_outcome = np.triu(M, 1).sum()
    p_result_only = p_outcome - p_exact          # right outcome, wrong exact score
    return p_exact * exact_pts + p_result_only * result_pts


def optimal_prediction(P, home, away, phase='group', major=True, standing=None, maxg=10):
    """
    Return (best_score, ev, modal_score, table) where best_score maximizes
    expected pool points. `standing` optionally biases variance:
      'leading'  -> ties broken toward the safer (more probable) of near-equal EVs
      'trailing' -> ties broken toward the bolder (higher-exact) pick
    """
    exact_pts, result_pts = POINTS[phase]
    M = score_matrix(P, home, away, major=major, maxg=max(maxg, 12))
    # modal (what the honest model reports)
    modal = np.unravel_index(np.argmax(M), M.shape)
    cand = []
    for i in range(maxg):
        for j in range(maxg):
            ev = expected_points(M, i, j, exact_pts, result_pts)
            cand.append(((i, j), ev, M[i, j]))
    cand.sort(key=lambda x: -x[1])
    # standing-based tiebreak among near-optimal (within 2% EV of the top)
    top_ev = cand[0][1]
    near = [c for c in cand if c[1] >= top_ev - 0.02 * abs(top_ev)]
    if standing == 'leading':
        near.sort(key=lambda x: -x[2])           # most probable
    elif standing == 'trailing':
        near.sort(key=lambda x: (-x[2] if False else x[2]))  # boldest (lowest prob) among near-best
        near.sort(key=lambda x: x[2])
    best = near[0]
    return {
        'best_score': f"{best[0][0]}-{best[0][1]}",
        'best_ev': round(best[1], 3),
        'modal_score': f"{modal[0]}-{modal[1]}",
        'modal_ev': round(expected_points(M, modal[0], modal[1], exact_pts, result_pts), 3),
        'top5': [(f"{i}-{j}", round(ev, 3), round(p, 3)) for (i, j), ev, p in cand[:5]],
    }


if __name__ == '__main__':
    import pickle
    P = pickle.load(open('P72.pkl', 'rb'))
    demo = [('Brazil', 'Japan', 'r32'), ('Mexico', 'Ecuador', 'r32'),
            ('Argentina', 'Cape Verde', 'r32'), ('Netherlands', 'Morocco', 'r32')]
    print(f"{'Match':28}{'phase':6}{'MODAL':8}{'OPTIMAL':9}{'EV gain'}")
    for h, a, ph in demo:
        r = optimal_prediction(P, h, a, ph)
        gain = r['best_ev'] - r['modal_ev']
        print(f"{h+'-'+a:28}{ph:6}{r['modal_score']:8}{r['best_score']:9}+{gain:.3f}")
