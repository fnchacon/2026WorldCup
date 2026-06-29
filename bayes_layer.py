"""
bayes_layer.py — Laplace-approximation Bayesian uncertainty layer.

Our ridge-penalized fit in wc2026_model.py is already a maximum-a-posteriori
(MAP) estimate under a Gaussian prior. This module computes the posterior
covariance from the Hessian at the mode (the Laplace approximation), draws
parameter sets from it, and propagates that uncertainty through the knockout
bracket. The payoff: thin-data teams get wide posteriors, and that uncertainty
flows into the title odds instead of being silently ignored.

This is the cheap, fast cousin of the full MCMC fit Luke Benz runs in Stan.
For a near-Gaussian posterior it lands in the same place; for the very thinnest
teams the tails differ slightly. Given how small the overall effect turns out
to be (~2 points off the favorite), the approximation is more than good enough.

See README.md for the math and the comparison to Benz (2021).
"""
import numpy as np
import scipy.sparse as sp
from scipy.stats import poisson
import wc2026_model as wc

RIDGE = 1.0


def laplace_covariance(P, rows):
    """Posterior covariance via the Hessian of the negative log-posterior at MAP."""
    tix = P['tix']; N = len(tix)
    atk, dfn, mu, hadv, gam = P['atk'], P['dfn'], P['mu'], P['hadv'], P['gam']
    Pn = 2 * N + 3
    ri, ci, dv, Wobs = [], [], [], []
    M = len(rows)
    for i, (h, a, gh, ga, neu, w, maj) in enumerate(rows):
        ih, ia = tix[h], tix[a]; home = 0.0 if neu else 1.0
        lh = np.exp(mu + atk[ih] - dfn[ia] + hadv * home + gam * maj)
        la = np.exp(mu + atk[ia] - dfn[ih] + gam * maj)
        ri += [i] * 5; ci += [ih, N + ia, 2 * N, 2 * N + 1, 2 * N + 2]
        dv += [1, -1, 1, home, 1.0 * maj]; Wobs.append(w * lh)
        ri += [M + i] * 4; ci += [ia, N + ih, 2 * N, 2 * N + 2]
        dv += [1, -1, 1, 1.0 * maj]; Wobs.append(w * la)
    X = sp.coo_matrix((dv, (ri, ci)), shape=(2 * M, Pn)).tocsr()
    Hlik = (X.T @ sp.diags(np.array(Wobs)) @ X).toarray()
    S = sum(r[5] for r in rows)
    Hpen = np.zeros((Pn, Pn))
    for i in range(2 * N):
        Hpen[i, i] += 2 * RIDGE / N
    Hpen[:N, :N] += 200.0 / N**2
    Hpen[N:2*N, N:2*N] += 200.0 / N**2
    Hg = Hlik + (S / 1000.0) * Hpen + np.eye(Pn) * 1e-6
    return np.linalg.inv(Hg)


def champion_table(P, Cov, bracket_order, hosts, n_draws=200, seed=7, maxg=8):
    """Average stage-reach probabilities over posterior draws (the Bayesian table)."""
    tix = P['tix']; N = len(tix); rho = P['rho']
    atk, dfn, mu, hadv, gam = P['atk'], P['dfn'], P['mu'], P['hadv'], P['gam']
    order = bracket_order
    sub_idx = np.array([tix[t] for t in order] + [N + tix[t] for t in order]
                       + [2 * N, 2 * N + 1, 2 * N + 2])
    mean_full = np.concatenate([atk, dfn, [mu, hadv, gam]])
    sub_mean = mean_full[sub_idx]
    sub_cov = Cov[np.ix_(sub_idx, sub_idx)]
    sub_cov = (sub_cov + sub_cov.T) / 2 + np.eye(len(sub_idx)) * 1e-9
    Lc = np.linalg.cholesky(sub_cov)
    hostflags = [order[i] in hosts for i in range(32)]
    ar = np.arange(maxg)

    def reach(av, dv, mu_s, h_s, g_s):
        a = {i: av[i] for i in range(32)}; dd = {i: dv[i] for i in range(32)}; cache = {}
        def padv(i, j):
            if (i, j) in cache: return cache[(i, j)]
            lh = np.exp(mu_s + a[i] - dd[j] + (h_s if hostflags[i] else 0) + g_s)
            la = np.exp(mu_s + a[j] - dd[i] + (h_s if hostflags[j] else 0) + g_s)
            Mx = np.outer(poisson.pmf(ar, lh), poisson.pmf(ar, la))
            Mx[0, 0] *= 1 - lh*la*rho; Mx[0, 1] *= 1 + lh*rho
            Mx[1, 0] *= 1 + la*rho; Mx[1, 1] *= 1 - rho
            Mx = np.clip(Mx, 0, None); Mx /= Mx.sum()
            pH = np.tril(Mx, -1).sum(); pD = np.trace(Mx); pA = np.triu(Mx, 1).sum()
            v = pH + pD * (0.5 + 0.10 * (pH - pA) / (pH + pA + 1e-9))
            cache[(i, j)] = v; cache[(j, i)] = 1 - v; return v
        r_ = np.ones(32); st = np.zeros((32, 5))
        for rd in range(5):
            bs = 1 << rd; nr = np.zeros(32)
            for i in range(32):
                blk = (i // (bs * 2)) * (bs * 2); sib = blk + bs if (i - blk) < bs else blk
                nr[i] = r_[i] * sum(r_[o] * padv(i, o) for o in range(sib, sib + bs))
            r_ = nr; st[:, rd] = r_
        return st

    rng = np.random.default_rng(seed); acc = np.zeros((32, 5))
    for _ in range(n_draws):
        z = sub_mean + Lc @ rng.standard_normal(len(sub_mean))
        acc += reach(z[:32], z[32:64], z[64], z[65], z[66])
    return acc / n_draws * 100  # cols: R16, QF, SF, Final, Champion
