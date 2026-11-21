"""Motore a replay: scorre il tempo in avanti come se fosse live.
Garanzia anti-lookahead: un modello riceve l'esito di una previsione fatta in t
solo quando il tempo simulato T ha raggiunto t + h.
"""
import time
from collections import deque
import numpy as np
import pandas as pd

from .features import analog_patterns
from .models import build_models


def run(ctx, cfg, only=None, verbose=True):
    n = len(ctx.logmid)
    D = np.flatnonzero(ctx.valid & (np.arange(n) % cfg.stride == 0))
    if len(D) == 0:
        raise ValueError("Nessun punto decisionale valido (dati troppo corti o troppo radi?)")
    pos = np.full(n, -1, dtype=np.int64)
    pos[D] = np.arange(len(D))
    patterns = analog_patterns(ctx, D, cfg)

    H = list(cfg.horizons)
    models = {h: build_models(ctx, patterns, pos, cfg, h, only) for h in H}
    names = [nm for nm, _ in models[H[0]]]
    pending = {h: deque() for h in H}
    recs = []
    t0 = time.time()
    step = max(1, len(D) // 10)
    for i, T in enumerate(D):
        for h in H:
            q = pending[h]
            yh, yv = ctx.y[h], ctx.y_valid[h]
            while q and q[0] + h <= T:
                tp = q.popleft()
                if yv[tp]:
                    yz = max(-8.0, min(8.0, yh[tp] / (ctx.sigma[tp] * np.sqrt(h))))
                    for _, m in models[h]:
                        m.update(tp, yz)
            for mi, (_, m) in enumerate(models[h]):
                out = m.predict(T)
                if out is not None:
                    recs.append((T, h, mi, out[0], out[1]))
            q.append(T)
        if verbose and i % step == 0:
            print(f"  replay {100 * i / len(D):5.1f}%  ({time.time() - t0:.0f}s)", flush=True)

    a = np.array(recs, dtype=float)
    t = a[:, 0].astype(np.int64)
    h = a[:, 1].astype(np.int64)
    df = pd.DataFrame({
        "t": t,
        "time": ctx.index[t],
        "h": h,
        "model": np.array(names)[a[:, 2].astype(int)],
        "mu_z": a[:, 3],
        "p_up": a[:, 4],
        "sigma": ctx.sigma[t],
        "spread_rel": ctx.spread_rel[t],
    })
    y = np.full(len(df), np.nan)
    yv = np.zeros(len(df), dtype=bool)
    for hh in H:
        m = h == hh
        y[m] = ctx.y[hh][t[m]]
        yv[m] = ctx.y_valid[hh][t[m]]
    df["y"] = y
    df = df[yv].reset_index(drop=True)
    df["y_z"] = df.y / (df.sigma * np.sqrt(df.h))
    df["mu"] = df.mu_z * df.sigma * np.sqrt(df.h)
    if verbose:
        print(f"  replay completato: {len(D)} decisioni, {len(df)} previsioni valutate "
              f"in {time.time() - t0:.0f}s")
    return df
