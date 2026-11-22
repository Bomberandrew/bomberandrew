"""CLI.
  python -m battito synth --kind rw --days 3 --out data/synth_rw.csv
  python -m battito run --csv data/xauusd_ticks.csv --out results/run1
  python -m battito run --synthetic momentum --days 3
  python -m battito selftest
  python -m battito ticks --from 2026-11-02 --to 2026-11-14 --out data/xau_ticks.csv
  python -m battito fetch --from 2026-09-01 --to 2026-11-21 --out data/xau_1min.csv
  python -m battito run --bars data/xau_1min.csv --holdout-from 2026-11-01
"""
import argparse
import sys
import time

import pandas as pd

from .config import Config, bar_config
from . import data, features, engine, evaluation, report, synthetic, twelvedata, dukascopy


def pipeline(ticks, cfg, out=None, only=None, verbose=True, holdout_from=None):
    t0 = time.time()
    ticks, q = data.clean_ticks(ticks, cfg)
    grid, gi = data.build_grid(ticks, cfg)
    if verbose:
        print("[1/4] qualita' dato:", q)
        print("      griglia:", gi)
    ctx = features.build_context(grid, cfg)
    if verbose:
        print(f"[2/4] feature AUTO={len(ctx.auto_names)} TA={len(ctx.ta_names)}  "
              f"punti validi={int(ctx.valid.sum())}")
        print("[3/4] replay walk-forward...")
    pred = engine.run(ctx, cfg, only=only, verbose=verbose)
    summary = evaluation.evaluate(pred, cfg)
    hours = evaluation.by_hour(pred)
    if verbose:
        print("[4/4] risultati (r2/skill/bss/coverage/hit/stable in %):")
        print(report.fmt_table(summary))
        print(f"\nTest multipli: {summary.attrs['n_tests']}  alpha Bonferroni="
              f"{summary.attrs['alpha_bonferroni']:.2e}")
        print("CONCLUSIONE:", report.conclusion(summary))
        print(f"tempo totale {time.time() - t0:.0f}s")
    if holdout_from:
        ho = pred[pred.time >= pd.Timestamp(holdout_from, tz="UTC")]
        hs = evaluation.evaluate(ho, cfg)
        if verbose:
            print(f"\n=== HOLD-OUT da {holdout_from} ({ho.t.nunique()} decisioni) ===")
            print(report.fmt_table(hs))
            print("CONCLUSIONE HOLD-OUT:", report.conclusion(hs))
        if out:
            import os
            os.makedirs(out, exist_ok=True)
            hs.to_csv(os.path.join(out, "summary_holdout.csv"), index=False)
    if out:
        report.save(out, cfg, q, gi, summary, pred, hours)
        if verbose:
            print("salvato in", out)
    return summary, pred


def main(argv=None):
    ap = argparse.ArgumentParser(prog="battito")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="genera tick sintetici")
    s.add_argument("--kind", choices=["rw", "momentum"], default="rw")
    s.add_argument("--days", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)

    f = sub.add_parser("fetch", help="scarica barre da Twelve Data")
    f.add_argument("--symbol", default="XAU/USD")
    f.add_argument("--interval", default="1min")
    f.add_argument("--from", dest="date_from", required=True)
    f.add_argument("--to", dest="date_to", required=True)
    f.add_argument("--apikey", default=None, help="meglio: export TWELVEDATA_API_KEY=...")
    f.add_argument("--out", required=True)

    k = sub.add_parser("ticks", help="scarica tick Dukascopy (gratis, con bid/ask)")
    k.add_argument("--symbol", default="XAUUSD")
    k.add_argument("--from", dest="date_from", required=True)
    k.add_argument("--to", dest="date_to", required=True, help="esclusa")
    k.add_argument("--out", required=True)

    r = sub.add_parser("run", help="esperimento walk-forward")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="tick: timestamp,bid,ask")
    src.add_argument("--bars", help="barre 1 min: timestamp,open,high,low,close")
    src.add_argument("--synthetic", choices=["rw", "momentum"])
    r.add_argument("--days", type=int, default=3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--from", dest="date_from")
    r.add_argument("--to", dest="date_to")
    r.add_argument("--horizons", default=None, help="default 5,30,120 (s) o 1,5,15 (barre)")
    r.add_argument("--spread-bps", type=float, default=0.8, help="costo ipotizzato con --bars")
    r.add_argument("--holdout-from", default=None, help="data: tabella separata dal giorno in poi")
    r.add_argument("--stride", type=int, default=5)
    r.add_argument("--models", default=None, help="es. trend_1f,auto_rls,analog")
    r.add_argument("--out", default="results/latest")

    sub.add_parser("selftest", help="controlli di onesta' del laboratorio")
    a = ap.parse_args(argv)

    if a.cmd == "synth":
        synthetic.generate(days=a.days, kind=a.kind, seed=a.seed).to_csv(a.out, index=False)
        print("scritto", a.out)
        return 0
    if a.cmd == "selftest":
        from .selftest import main as st
        return st()

    if a.cmd == "fetch":
        df = twelvedata.fetch(a.symbol, a.interval, a.date_from, a.date_to, a.apikey)
        df.to_csv(a.out, index=False)
        print(f"scritte {len(df)} barre in {a.out}")
        return 0

    if a.cmd == "ticks":
        df = dukascopy.fetch(a.symbol, a.date_from, a.date_to)
        df.to_csv(a.out, index=False)
        print(f"scritti {len(df)} tick in {a.out}")
        return 0

    hz = tuple(int(x) for x in a.horizons.split(",")) if a.horizons else None
    if a.bars:
        cfg = bar_config(hz or (1, 5, 15))
        bars = pd.read_csv(a.bars)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars, n_off = twelvedata.market_hours(bars)
        print(f"barre fuori orario di mercato rimosse: {n_off}")
        ticks = twelvedata.bars_to_quotes(bars, a.spread_bps)
    else:
        cfg = Config(horizons=hz or (5, 30, 120), stride=a.stride).check()
        if a.csv:
            ticks = data.load_ticks(a.csv)
    if a.synthetic:
        ticks = data.load_ticks_df(synthetic.generate(days=a.days, kind=a.synthetic, seed=a.seed))
    if a.date_from:
        ticks = ticks[ticks.ts >= a.date_from]
    if a.date_to:
        ticks = ticks[ticks.ts < a.date_to]
    only = set(a.models.split(",")) if a.models else None
    pipeline(ticks.reset_index(drop=True), cfg, out=a.out, only=only, holdout_from=a.holdout_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
