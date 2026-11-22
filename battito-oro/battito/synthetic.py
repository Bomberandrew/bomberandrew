"""Generatore di tick sintetici per verificare che il laboratorio sia onesto.

kind="rw"       : random walk puro con regimi di volatilita' -> NESSUN segnale deve emergere
kind="momentum" : rendimento futuro dipende dalla media degli ultimi mom_len secondi
                  -> il sistema DEVE trovarlo (controllo positivo)
"""
import numpy as np
import pandas as pd


def generate(days=2, kind="rw", seed=0, start="2026-01-05", price=4000.0,
             vol_day=0.012, spread_usd=0.25, tick_rate=3.0,
             momentum=0.3, mom_len=20):
    rng = np.random.default_rng(seed)
    sess_len, gap = 23 * 3600, 3600
    sig1 = vol_day / np.sqrt(86400)
    out = []
    t_start = pd.Timestamp(start, tz="UTC").value // 10 ** 6  # ms
    lp = np.log(price)
    logv = 0.0
    for d in range(days):
        n = sess_len
        eps = rng.standard_normal(n)
        vshock = rng.standard_normal(n)
        sec = np.arange(n)
        season = 1 + 0.5 * np.sin(2 * np.pi * sec / 86400 - 1.0)
        r = np.empty(n)
        buf = np.zeros(mom_len)
        bsum, bi = 0.0, 0
        for i in range(n):
            logv = 0.9995 * logv + 0.02 * vshock[i]
            s = sig1 * season[i] * np.exp(logv)
            ri = s * eps[i]
            if kind == "momentum":
                ri += momentum * bsum / mom_len
            r[i] = ri
            bsum += ri - buf[bi]
            buf[bi] = ri
            bi = (bi + 1) % mom_len
        path = lp + np.cumsum(r)
        lp = path[-1]
        cnt = rng.poisson(tick_rate * season)
        sidx = np.repeat(sec, cnt)
        ms = rng.integers(0, 1000, size=len(sidx))
        base = t_start + d * 86400 * 1000
        ts = base + sidx * 1000 + ms
        mid = np.exp(path[sidx])
        sp = spread_usd * (1 + 0.3 * rng.random(len(sidx)))
        out.append(pd.DataFrame({"timestamp": ts,
                                 "bid": np.round(mid - sp / 2, 2),
                                 "ask": np.round(mid + sp / 2, 2)}))
    df = pd.concat(out, ignore_index=True)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
