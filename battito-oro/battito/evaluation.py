"""Validazione: il modello batte davvero le previsioni banali su dati mai visti?

Metriche per (modello, orizzonte):
  r2_oos        1 - MSE_modello / MSE_zero (random walk). >0 = meglio del "non so"
  r2_lo/hi      intervallo bootstrap a blocchi (rispetta l'autocorrelazione)
  skill_trend   1 - MSE_modello / MSE_trend_1f (batte anche il momentum banale?)
  ic, ic_p      correlazione previsione/esito su campione NON sovrapposto, p-value
  hit, hit_p    % direzione corretta quando il modello "agisce" (|p-0.5|>=margin)
  bss           Brier skill score vs p=0.5 (calibrazione probabilistica)
  stable        frazione di fold temporali con r2 > 0
  ph_alarms     allarmi Page-Hinkley: l'errore cresce = cambio di regime
  paper_bps     SIMULAZIONE: rendimento medio per decisione al netto dello spread
  edge_spread   |movimento previsto| mediano / spread mediano (<1 = inutilizzabile)
"""
from math import erfc, sqrt
import numpy as np
import pandas as pd


def _p_one_sided(z):
    return 0.5 * erfc(z / sqrt(2))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - w, c + w


def non_overlapping(t, h):
    keep = np.zeros(len(t), dtype=bool)
    last = -10 ** 18
    for i, ti in enumerate(t):
        if ti >= last + h:
            keep[i] = True
            last = ti
    return keep


def block_bootstrap_r2(e2m, e2z, block, nb, rng):
    n = len(e2m)
    if n < 2 * block:
        return np.nan, np.nan
    ext_m = np.concatenate([e2m, e2m[:block]])
    ext_z = np.concatenate([e2z, e2z[:block]])
    cm = np.concatenate([[0], np.cumsum(ext_m)])
    cz = np.concatenate([[0], np.cumsum(ext_z)])
    bm = cm[block:block + n] - cm[:n]
    bz = cz[block:block + n] - cz[:n]
    nblk = int(np.ceil(n / block))
    s = rng.integers(0, n, size=(nb, nblk))
    r2 = 1 - bm[s].sum(1) / bz[s].sum(1)
    return float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))


def page_hinkley(x, delta=0.05, lam=50.0):
    mean, cum, mn, n, alarms = 0.0, 0.0, 0.0, 0, 0
    for v in x:
        n += 1
        mean += (v - mean) / n
        cum += v - mean - delta
        mn = min(mn, cum)
        if cum - mn > lam:
            alarms += 1
            mean, cum, mn, n = 0.0, 0.0, 0.0, 0
    return alarms


def evaluate_one(g, cfg, rng, mse_trend=None):
    h = int(g.h.iloc[0])
    g = g.sort_values("t")
    y, mu, p = g.y.to_numpy(), g.mu.to_numpy(), g.p_up.to_numpy()
    yz, muz = g.y_z.to_numpy(), g.mu_z.to_numpy()
    e2m, e2z = (y - mu) ** 2, y ** 2
    mse = e2m.mean()
    r = {"n": len(g), "r2_oos": 1 - e2m.sum() / e2z.sum()}
    block = max(1, 10 * int(np.ceil(h / cfg.stride)))
    r["r2_lo"], r["r2_hi"] = block_bootstrap_r2(e2m, e2z, block, cfg.bootstrap, rng)
    r["skill_trend"] = 1 - mse / mse_trend if mse_trend else np.nan

    nov = non_overlapping(g.t.to_numpy(), h)
    a, b = muz[nov], yz[nov]
    if len(a) > 10 and a.std() > 0 and b.std() > 0:
        ic = float(np.corrcoef(a, b)[0, 1])
        tstat = ic * sqrt((len(a) - 2) / max(1e-12, 1 - ic * ic))
        r["ic"], r["ic_p"] = ic, _p_one_sided(tstat)
    else:
        r["ic"], r["ic_p"] = np.nan, np.nan

    act = (np.abs(p - 0.5) >= cfg.conf_margin) & (y != 0)
    r["coverage"] = act.mean()
    side = np.sign(p - 0.5)
    hit = side == np.sign(y)
    k, nn = int(hit[act].sum()), int(act.sum())
    r["hit"] = k / nn if nn else np.nan
    act_nov = act & nov
    k2, n2 = int(hit[act_nov].sum()), int(act_nov.sum())
    r["hit_p"] = _p_one_sided((k2 - n2 / 2) / sqrt(n2 / 4)) if n2 > 10 else np.nan
    r["hit_lo"], r["hit_hi"] = wilson(k2, n2)

    nz = y != 0
    brier = np.mean((p[nz] - (y[nz] > 0)) ** 2) if nz.any() else np.nan
    r["bss"] = 1 - brier / 0.25

    folds = np.array_split(np.arange(len(g)), cfg.n_folds)
    f_r2 = [1 - e2m[f].sum() / e2z[f].sum() for f in folds if len(f) and e2z[f].sum() > 0]
    r["stable"] = float(np.mean(np.array(f_r2) > 0)) if f_r2 else np.nan
    r["fold_r2"] = [round(float(v) * 100, 3) for v in f_r2]
    r["ph_alarms"] = page_hinkley((yz - muz) ** 2 - 1.0)

    pnl = side * y - g.spread_rel.to_numpy()
    if nn:
        r["paper_bps"] = float(pnl[act].mean() * 1e4)
        pn = pnl[act_nov]
        r["paper_t"] = float(pn.mean() / (pn.std(ddof=1) / sqrt(len(pn)))) if len(pn) > 10 else np.nan
        r["gross_bps"] = float((side * y)[act].mean() * 1e4)
    else:
        r["paper_bps"] = r["paper_t"] = r["gross_bps"] = np.nan
    sp = np.median(g.spread_rel)
    r["edge_spread"] = float(np.median(np.abs(mu)) / sp) if sp > 0 else np.inf
    return r


def evaluate(pred, cfg):
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for h, gh in pred.groupby("h"):
        tr = gh[gh.model == "trend_1f"]
        mse_trend = float(((tr.y - tr.mu) ** 2).mean()) if len(tr) else None
        for m, g in gh.groupby("model"):
            r = evaluate_one(g, cfg, rng, None if m == "trend_1f" else mse_trend)
            r.update(model=m, h=int(h))
            rows.append(r)
    s = pd.DataFrame(rows)
    n_tests = int((s.model != "zero").sum())
    a_bonf = cfg.alpha / max(1, n_tests)
    s["verdict_stat"] = s.apply(lambda r: _verdict_stat(r, cfg.alpha, a_bonf), axis=1)
    s["verdict_econ"] = s.apply(_verdict_econ, axis=1)
    s.attrs["alpha_bonferroni"] = a_bonf
    s.attrs["n_tests"] = n_tests
    return s


def _verdict_stat(r, alpha, a_bonf):
    if r.model == "zero":
        return "riferimento"
    if r.r2_lo > 0 and r.ic_p < a_bonf and r.stable >= 0.8:
        return "CANDIDATO"
    if (r.ic_p < alpha) or (r.r2_oos > 0):
        return "debole"
    return "assente"


def _verdict_econ(r):
    if r.model == "zero":
        return "-"
    if r.verdict_stat == "CANDIDATO" and r.paper_bps > 0 and r.paper_t > 2:
        return "POSITIVO(sim)"
    return "negativo"


def by_hour(pred):
    rows = []
    for (m, h, hr), g in pred.groupby(["model", "h", pred.time.dt.hour]):
        if m == "zero" or len(g) < 30:
            continue
        ic = np.corrcoef(g.mu_z, g.y_z)[0, 1] if g.mu_z.std() > 0 else np.nan
        e2m, e2z = ((g.y - g.mu) ** 2).sum(), (g.y ** 2).sum()
        hit = (np.sign(g.p_up - 0.5) == np.sign(g.y))[g.y != 0].mean()
        rows.append(dict(model=m, h=h, hour_utc=hr, n=len(g), ic=ic,
                         r2_oos=1 - e2m / e2z if e2z else np.nan, hit_all=hit))
    return pd.DataFrame(rows)
