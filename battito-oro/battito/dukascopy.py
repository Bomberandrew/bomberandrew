"""Download tick storici gratuiti Dukascopy (file orari .bi5, LZMA), sola lettura.

URL: https://datafeed.dukascopy.com/datafeed/XAUUSD/YYYY/MM/DD/HHh_ticks.bi5
     ATTENZIONE: MM e' 0-based (gennaio = 00).
Record 20 byte big-endian: ms dall'inizio ora, ask, bid (uint32, /1000 per XAUUSD), vol ask, vol bid.
I file scaricati sono messi in cache in data/duka/ (ri-esecuzioni istantanee).
"""
import lzma
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed"
DT = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("av", ">f4"), ("bv", ">f4")])
PAUSE = 3.0          # secondi tra richieste: Dukascopy risponde 429 se si va troppo veloci
POINT = {"XAUUSD": 1000.0, "XAGUSD": 1000.0}


def _hour(symbol, ts, cache):
    path = os.path.join(cache, symbol, ts.strftime("%Y%m%d_%H") + ".bi5")
    if not os.path.exists(path):
        url = f"{BASE}/{symbol}/{ts.year}/{ts.month - 1:02d}/{ts.day:02d}/{ts.hour:02d}h_ticks.bi5"
        raw = None
        for attempt in range(30):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    raw = r.read()
                time.sleep(PAUSE)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raw = b""
                    break
                time.sleep(min(300, 30 * (attempt + 1)) if e.code == 429 else 2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
        if raw is None:
            raise RuntimeError(f"download fallito: {url}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw)
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        return None
    a = np.frombuffer(lzma.decompress(raw), dtype=DT)
    p = POINT.get(symbol, 1000.0)
    t0 = ts.value // 10 ** 6
    return pd.DataFrame({"timestamp": t0 + a["ms"].astype(np.int64),
                         "bid": a["bid"] / p, "ask": a["ask"] / p})


def fetch(symbol, start, end, cache="data/duka", workers=1, verbose=True):
    hours = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                          freq="h", inclusive="left")
    out, done = [], 0
    with ThreadPoolExecutor(workers) as ex:
        for df in ex.map(lambda h: _hour(symbol, h, cache), hours):
            done += 1
            if df is not None:
                out.append(df)
            if verbose and done % 24 == 0:
                print(f"  {done}/{len(hours)} ore", flush=True)
    if not out:
        raise RuntimeError("nessun tick nel periodo richiesto")
    return pd.concat(out, ignore_index=True)
