"""Download barre OHLC da Twelve Data (sola lettura).

Chiave: variabile d'ambiente TWELVEDATA_API_KEY oppure --apikey (mai scriverla nel codice).
Piano free: max 5000 barre per richiesta, ~8 richieste/minuto -> pausa tra richieste.
"""
import json
import os
import time
import urllib.parse
import urllib.request
import pandas as pd

URL = "https://api.twelvedata.com/time_series"


def fetch(symbol, interval, start, end, apikey, pause=8.0, chunk_days=3, verbose=True):
    apikey = apikey or os.environ.get("TWELVEDATA_API_KEY")
    if not apikey:
        raise SystemExit("Manca la chiave: export TWELVEDATA_API_KEY=... oppure --apikey")
    t0, t_end = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    while t0 < t_end:
        t1 = min(t0 + pd.Timedelta(days=chunk_days), t_end)
        q = urllib.parse.urlencode({
            "symbol": symbol, "interval": interval, "timezone": "UTC", "outputsize": 5000,
            "start_date": t0.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": t1.strftime("%Y-%m-%d %H:%M:%S"), "apikey": apikey})
        for attempt in range(4):
            with urllib.request.urlopen(f"{URL}?{q}", timeout=60) as r:
                d = json.load(r)
            if d.get("status") == "ok":
                break
            if d.get("code") == 429:          # limite richieste: attende e riprova
                time.sleep(60)
                continue
            if "No data" in str(d.get("message", "")):
                d = {"values": []}
                break
            raise RuntimeError(f"Twelve Data: {d}")
        v = d.get("values", [])
        if verbose:
            print(f"  {t0:%Y-%m-%d} -> {t1:%Y-%m-%d}: {len(v)} barre")
        if v:
            frames.append(pd.DataFrame(v))
        t0 = t1
        if t0 < t_end:
            time.sleep(pause)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"datetime": "timestamp"})
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def bars_to_quotes(bars, spread_bps):
    """Barre -> quote sintetiche sul close con spread relativo costante (costo ipotizzato)."""
    s = spread_bps / 1e4
    return pd.DataFrame({"ts": bars["timestamp"],
                         "bid": bars["close"] * (1 - s / 2),
                         "ask": bars["close"] * (1 + s / 2)})


def market_hours(bars):
    """Rimuove barre fuori mercato (spesso prezzi derivati dal fornitore):
    pausa giornaliera 17:00-18:00 New York e weekend da ven 17:00 a dom 18:00 NY (DST incluso)."""
    ny = bars["timestamp"].dt.tz_convert("America/New_York")
    h, dow = ny.dt.hour, ny.dt.dayofweek
    off = (h == 17) | (dow == 5) | ((dow == 4) & (h >= 17)) | ((dow == 6) & (h < 18))
    return bars[~off].reset_index(drop=True), int(off.sum())
