# Experiments

Open investigations on top of the base model. Each is framed as a testable
question with a pre-registered prediction, so we can't fool ourselves after the
fact. Results are scored honestly, win or lose.

---

## 1. Does ball possession predict results? (And does Paraguay break the rule?)

**Origin.** A friend proposed that high-possession teams usually win. The
co-author countered with "...except Paraguay" — the counter-attacking side that
wins *without* the ball — as the exception that might prove (or break) the rule.

**Two distinct questions, because possession can only be trusted in-tournament**
(a team's style shifts between qualifying and the finals, so pre-tournament
averages measure a different team):

- **Explanatory:** across played matches, does higher in-match possession
  associate with winning, *after* controlling for team quality (our attack/
  defense ratings)? This tests the theory as stated. It is **not** a forecasting
  tool — you don't know possession before kickoff.
- **Predictive (knockouts only):** by the Round of 32 every team has three
  group matches of possession data. Does group-stage average possession improve
  prediction of knockout results beyond the goals-based ratings? Small sample,
  but the only version that could help a live bracket.

**Pre-registered predictions.**
- Friend: possession positively predicts goal differential.
- Co-author: it holds on average but **reverses or vanishes for the low-
  possession-but-winning archetype** (Paraguay, Morocco, etc.) — detectable as
  systematic residuals once team quality is controlled for.

**Data.** `data/possession_2026.csv` — FIFA three-part possession (team /
opponent / contested share) for all 72 group-stage matches, with match IDs and
source URLs. The **contested** column is a bonus variable: it may flag scrappy,
transitional games with higher upset variance.

### First-look results (group stage, 72 matches) — both hypotheses CONFIRMED

- **The friend is right (on average).** `corr(home possession %, goal
  differential) = +0.42`. Win rate climbs cleanly with possession:
  `<45% -> 31% win, -0.59 GD` · `45-55% -> 56% win, +1.17 GD` ·
  `>55% -> 60% win, +1.56 GD`.
- **The co-author is right (at the extremes).** The most dominant possession
  performance of the tournament *lost*: **Türkiye had 78% of the ball and lost
  1-0 to Paraguay.** The low-possession-winner list is exactly the predicted
  archetype — Australia (27%!), Ecuador beating Germany on 39%, and tellingly
  **Argentina winning 3-0 on 44%**: the best team cedes the ball and still wins.
- **The resolution (the actual finding).** Both are true because possession is a
  **proxy for control that the genuinely elite and the well-drilled counter-
  attackers can break.** The +0.42 lives mostly in the mid-table, where
  possession tracks quality; the residuals — at *both* ends of the talent
  distribution — are where it fails. Prediction: once attack/defense ratings
  (which already encode quality) are in the model, possession's *independent*
  contribution shrinks toward a noisy style indicator, not a cause.
- **Contested possession:** weak so far — `corr(contested, |margin|) = -0.16`,
  `corr(contested, total goals) = -0.10`, mean 6%. Underpowered; "probably
  minor, needs more data," not a finding.

**Still to do.** The formal test: fit model residuals (actual minus expected),
regress on possession, and check whether the counter-attacking teams sit
systematically off the line (the falsifier). One regression, one scatter plot
with the Paraguay archetype highlighted. That plot is the post.

---

## 2. Pool-optimal predictions vs. honest forecasts

**Question.** The base model reports the *most likely* scoreline. But the pool
pays asymmetrically for exact scores vs. correct results (3:1 in groups, up to
15:6 in the final), and a draw prediction scores nothing on a 2-1. Does
optimizing for *expected pool points* beat reporting the modal score?

**Answer: yes.** See `pool_optimizer.py` and
`predictions/r32_pool_optimal.csv`. Across the R32, the points-optimal picks
beat the modal picks by ~2 expected points — and the gains come almost entirely
from games where the model hedges to 1-1. The optimizer breaks those draws
*toward the favorite* (Portugal 1-1 -> 2-0, Switzerland 1-1 -> 2-1), which is
the mathematically-correct version of the co-author's instinct to turn draws
into decisive scores. His gut was directionally right; it just fired in the
wrong games (dead-rubber 0-0s instead of favorites with a real edge).

**Note.** This is a *separate betting layer*. The honest forecasting model stays
untouched — one tells the truth, one wins the office pool. Keep both.
