"""CLI.
  python -m battito synth --kind rw --days 3 --out data/synth_rw.csv
  python -m battito run --csv data/xauusd_ticks.csv --out results/run1
  python -m battito run --synthetic momentum --days 3
  python -m battito selftest
"""
import argparse
import sys
import time

from .config import Config
from . import data, features, engine, evaluation, report, synthetic


def pipeline(ticks, cfg, out=None, only=None, verbose=True):
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

    r = sub.add_parser("run", help="esperimento walk-forward")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--synthetic", choices=["rw", "momentum"])
    r.add_argument("--days", type=int, default=3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--from", dest="date_from")
    r.add_argument("--to", dest="date_to")
    r.add_argument("--horizons", default="5,30,120")
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

    cfg = Config(horizons=tuple(int(x) for x in a.horizons.split(",")), stride=a.stride).check()
    if a.csv:
        ticks = data.load_ticks(a.csv)
    else:
        ticks = data.load_ticks_df(synthetic.generate(days=a.days, kind=a.synthetic, seed=a.seed))
    if a.date_from:
        ticks = ticks[ticks.ts >= a.date_from]
    if a.date_to:
        ticks = ticks[ticks.ts < a.date_to]
    only = set(a.models.split(",")) if a.models else None
    pipeline(ticks.reset_index(drop=True), cfg, out=a.out, only=only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
