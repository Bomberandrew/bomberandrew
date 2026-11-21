"""Caricamento tick, controllo qualita' e griglia a 1 secondo.

Formati CSV accettati (rilevamento automatico, colonne case-insensitive):
  - timestamp,bid,ask            (timestamp ISO o epoch s/ms/us/ns)
  - Gmt time,Ask,Bid,...         (export Dukascopy, "dd.mm.yyyy HH:MM:SS.fff")
  - timestamp,price | mid | last (senza bid/ask: spread assunto = 0)
"""
import numpy as np
import pandas as pd

TIME_COLS = ["timestamp", "time", "gmt time", "datetime", "date", "ts", "local time"]
PRICE_COLS = ["mid", "price", "last", "close"]


def _parse_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype("float64")
        mx = np.nanmax(np.abs(v.to_numpy()))
        unit = "ns" if mx > 1e17 else "us" if mx > 1e14 else "ms" if mx > 1e11 else "s"
        return pd.to_datetime(v.astype("int64") if unit != "s" else v, unit=unit, utc=True)
    s = s.astype(str).str.strip()
    for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"):
        try:
            return pd.to_datetime(s, format=fmt, utc=True)
        except (ValueError, TypeError):
            pass
    # "... GMT+0200" (Dukascopy local time) -> rimuove la parte testuale
    s2 = s.str.replace(r"\s*GMT([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
    return pd.to_datetime(s2, utc=True, format="mixed")


def load_ticks(path: str) -> pd.DataFrame:
    return load_ticks_df(pd.read_csv(path))


def load_ticks_df(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in raw.columns}
    tcol = next((cols[c] for c in TIME_COLS if c in cols), None)
    if tcol is None:
        raise ValueError(f"Colonna tempo non trovata. Colonne: {list(raw.columns)}")
    out = pd.DataFrame({"ts": _parse_time(raw[tcol])})
    if "bid" in cols and "ask" in cols:
        out["bid"] = pd.to_numeric(raw[cols["bid"]], errors="coerce").to_numpy()
        out["ask"] = pd.to_numeric(raw[cols["ask"]], errors="coerce").to_numpy()
    else:
        pcol = next((cols[c] for c in PRICE_COLS if c in cols), None)
        if pcol is None:
            raise ValueError("Servono colonne bid+ask oppure price/mid/last/close")
        p = pd.to_numeric(raw[pcol], errors="coerce").to_numpy()
        out["bid"], out["ask"] = p, p
    return out


def clean_ticks(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    q = {"n_raw": int(len(df))}
    df = df.dropna().sort_values("ts", kind="stable")
    q["nan_removed"] = q["n_raw"] - len(df)
    n0 = len(df)
    df = df.drop_duplicates(subset=["ts", "bid", "ask"])
    q["duplicates_removed"] = n0 - len(df)
    bad = (df.bid <= 0) | (df.ask <= 0) | (df.ask < df.bid)
    q["invalid_or_crossed"] = int(bad.sum())
    df = df[~bad]
    spread = df.ask - df.bid
    med = spread.rolling(500, min_periods=20).median().bfill()
    sp_out = spread > cfg.spread_outlier * med.clip(lower=1e-9)
    q["spread_outliers"] = int(sp_out.sum())
    df = df[~sp_out]
    # spike isolati: salto grande seguito da ritorno immediato
    mid = np.log((df.bid.to_numpy() + df.ask.to_numpy()) / 2)
    r = np.diff(mid, prepend=mid[0])
    # scala robusta calcolata solo sui tick che muovono il prezzo (molti tick hanno r=0)
    absr = pd.Series(np.where(r != 0, np.abs(r), np.nan))
    mad = absr.rolling(1000, min_periods=50).median().ffill().bfill().to_numpy() + 1e-12
    r_next = np.append(r[1:], 0.0)
    spike = (np.abs(r) > 30 * mad) & (np.abs(r + r_next) < 0.2 * np.abs(r))
    q["spikes_removed"] = int(spike.sum())
    df = df[~spike].reset_index(drop=True)
    q["n_clean"] = int(len(df))
    if len(df):
        q["first"] = str(df.ts.iloc[0])
        q["last"] = str(df.ts.iloc[-1])
        sp_rel = (df.ask - df.bid) / ((df.ask + df.bid) / 2)
        q["median_spread_bps"] = round(float(sp_rel.median() * 1e4), 3)
    return df, q


def build_grid(ticks: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    """Griglia regolare (1 s, o 1 min in modalita' barre): ultimo bid/ask del secondo, ffill limitato, sessioni, validita'."""
    sec = ticks.ts.dt.floor(cfg.freq)
    g = ticks.groupby(sec)
    last = g[["bid", "ask"]].last()
    cnt = g.size()
    idx = pd.date_range(last.index[0], last.index[-1], freq=cfg.freq)
    grid = last.reindex(idx)
    has = grid.bid.notna().to_numpy()
    n = len(idx)
    pos = np.arange(n)
    lastpos = pd.Series(np.where(has, pos, np.nan)).ffill().to_numpy()
    age = pos - lastpos
    grid = grid.ffill()
    grid["ticks"] = cnt.reindex(idx).fillna(0).to_numpy()
    grid["age"] = age
    # sessioni: nuovo blocco quando arriva un tick dopo un buco > session_gap
    tick_pos = pos[has]
    starts = np.zeros(n, dtype=bool)
    starts[0] = True
    starts[tick_pos[1:][np.diff(tick_pos) > cfg.session_gap]] = True
    session = np.cumsum(starts) - 1
    start_pos = pos[starts]
    grid["session"] = session
    grid["since_start"] = pos - start_pos[session]
    grid["valid"] = age <= cfg.max_stale
    grid["mid"] = (grid.bid + grid.ask) / 2
    grid["spread_rel"] = (grid.ask - grid.bid) / grid["mid"]
    info = {
        "grid_seconds": int(n),
        "valid_seconds": int(grid.valid.sum()),
        "sessions": int(session[-1] + 1),
        "seconds_with_ticks_pct": round(100 * has.mean(), 2),
    }
    return grid, info
