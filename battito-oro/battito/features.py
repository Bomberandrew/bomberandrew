"""Feature strettamente causali (al tempo t usano solo dati <= t) ed etichette.

Due famiglie separate, per confrontarle ad armi pari:
  AUTO : descrittori grezzi del "battito" (rendimenti multi-scala, accelerazione,
         regime di volatilita', intensita' tick, spread, ora) -> nessuna figura imposta
  TA   : indicatori classici di analisi tecnica (MACD, RSI, Bollinger, Donchian)
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

SIGMA_FLOOR = 2e-6
SCALES = (1, 5, 15, 60, 300)


def _ewm(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def _shift(x, k):
    out = np.empty_like(x)
    out[:k] = x[0]
    out[k:] = x[:-k]
    return out


@dataclass
class Context:
    index: pd.DatetimeIndex
    logmid: np.ndarray
    spread_rel: np.ndarray
    sigma: np.ndarray          # volatilita' per secondo (EWMA, causale)
    valid: np.ndarray          # punto utilizzabile per decidere (dato fresco + warmup)
    X_auto: np.ndarray
    X_ta: np.ndarray
    auto_names: list
    ta_names: list
    past_z: dict = field(default_factory=dict)   # h -> rendimento passato su h, normalizzato
    y: dict = field(default_factory=dict)        # h -> rendimento futuro log
    y_valid: dict = field(default_factory=dict)  # h -> etichetta valida


def build_context(grid: pd.DataFrame, cfg) -> Context:
    lm = np.log(grid["mid"].to_numpy())
    sess = grid["session"].to_numpy()
    since = grid["since_start"].to_numpy()
    gvalid = grid["valid"].to_numpy()
    n = len(lm)

    r1 = np.diff(lm, prepend=lm[0])
    r1[since == 0] = 0.0
    sigma = np.maximum(np.sqrt(_ewm(r1 ** 2, cfg.vol_span)), SIGMA_FLOOR)
    sig_s = np.maximum(np.sqrt(_ewm(r1 ** 2, 30)), SIGMA_FLOOR)
    sig_l = np.maximum(np.sqrt(_ewm(r1 ** 2, 1800)), SIGMA_FLOOR)

    def ret(k):
        return lm - _shift(lm, k)

    # ---------- AUTO ----------
    auto, an = [], []
    for k in SCALES:
        auto.append(ret(k) / (sigma * np.sqrt(k)))
        an.append(f"z_ret_{k}s")
    r5 = ret(5)
    auto.append((r5 - _shift(r5, 5)) / (sigma * np.sqrt(10)))
    an.append("accel_5s")
    r15 = ret(15)
    auto.append((r15 - _shift(r15, 15)) / (sigma * np.sqrt(30)))
    an.append("accel_15s")
    auto.append(np.log(sig_s / sig_l))
    an.append("vol_regime")
    tk = grid["ticks"].to_numpy().astype(float)
    auto.append(np.log1p(_ewm(tk, 30)) - np.log1p(_ewm(tk, 1800)))
    an.append("tick_intensity")
    sp = grid["spread_rel"].to_numpy()
    sp_m = _ewm(sp, 1800)
    sp_sd = np.sqrt(np.maximum(_ewm((sp - sp_m) ** 2, 1800), 1e-20))
    auto.append((sp - sp_m) / sp_sd)
    an.append("spread_z")
    hr = grid.index.hour.to_numpy() + grid.index.minute.to_numpy() / 60
    auto.append(np.sin(2 * np.pi * hr / 24))
    auto.append(np.cos(2 * np.pi * hr / 24))
    an += ["hour_sin", "hour_cos"]
    X_auto = np.clip(np.column_stack(auto), -5, 5)

    # ---------- TA ----------
    ta, tn = [], []
    ta.append((_ewm(lm, 30) - _ewm(lm, 120)) / (sigma * np.sqrt(60)))
    tn.append("macd_30_120")
    up = _ewm(np.maximum(r1, 0), 60)
    dn = _ewm(np.maximum(-r1, 0), 60)
    ta.append((up / (up + dn + 1e-18) - 0.5) * 2)
    tn.append("rsi_60")
    s = pd.Series(lm)
    m120 = s.rolling(120, min_periods=10).mean().to_numpy()
    sd120 = s.rolling(120, min_periods=10).std().to_numpy()
    ta.append(np.nan_to_num((lm - m120) / np.maximum(sd120, 1e-12)))
    tn.append("boll_z_120")
    hi = s.rolling(300, min_periods=10).max().to_numpy()
    lo = s.rolling(300, min_periods=10).min().to_numpy()
    ta.append(np.nan_to_num(2 * (lm - lo) / np.maximum(hi - lo, 1e-12) - 1))
    tn.append("donchian_300")
    ta.append(ret(60) / (sigma * np.sqrt(60)))
    tn.append("roc_60")
    X_ta = np.clip(np.column_stack(ta), -5, 5)

    valid = gvalid & (since >= cfg.warmup)
    ctx = Context(grid.index, lm, sp, sigma, valid, X_auto, X_ta, an, tn)

    for h in cfg.horizons:
        ctx.past_z[h] = np.clip(ret(h) / (sigma * np.sqrt(h)), -5, 5)
        y = np.full(n, np.nan)
        y[:-h] = lm[h:] - lm[:-h]
        yv = np.zeros(n, dtype=bool)
        yv[:-h] = gvalid[:-h] & gvalid[h:] & (sess[:-h] == sess[h:])
        ctx.y[h] = y
        ctx.y_valid[h] = yv & valid
    return ctx


def analog_patterns(ctx: Context, D: np.ndarray, cfg) -> np.ndarray:
    """Forma degli ultimi analog_len secondi, normalizzata per volatilita', compressa PAA."""
    L, P = cfg.analog_len, cfg.analog_paa
    w = np.lib.stride_tricks.sliding_window_view(ctx.logmid, L + 1)[D - L]
    path = (w[:, 1:] - w[:, :1]) / (ctx.sigma[D] * np.sqrt(L))[:, None]
    return np.clip(path.reshape(len(D), P, L // P).mean(axis=2), -6, 6)
