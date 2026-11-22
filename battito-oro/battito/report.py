import json
import os
import numpy as np
import pandas as pd

COLS = ["model", "h", "n", "coverage", "r2_oos", "r2_lo", "r2_hi", "skill_trend", "ic", "ic_p",
        "hit", "hit_p", "bss", "stable", "ph_alarms", "gross_bps", "paper_bps", "paper_t",
        "edge_spread", "verdict_stat", "verdict_econ"]


def fmt_table(s):
    t = s[COLS].copy()
    for c in ["r2_oos", "r2_lo", "r2_hi", "skill_trend", "bss"]:
        t[c] = (t[c] * 100).round(3)          # in punti percentuali
    for c in ["coverage", "hit", "stable"]:
        t[c] = (t[c] * 100).round(1)
    for c in ["ic", "gross_bps", "paper_bps", "paper_t", "edge_spread"]:
        t[c] = t[c].round(3)
    for c in ["ic_p", "hit_p"]:
        t[c] = t[c].map(lambda v: f"{v:.1e}" if pd.notna(v) else "nan")
    return t.sort_values(["h", "model"]).to_string(index=False)


def save(out_dir, cfg, quality, grid_info, summary, pred, hours):
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    hours.to_csv(os.path.join(out_dir, "by_hour.csv"), index=False)
    pred.to_csv(os.path.join(out_dir, "predictions.csv.gz"), index=False, compression="gzip")
    cand = summary[summary.verdict_stat == "CANDIDATO"]
    rep = {
        "config": cfg.to_dict(),
        "data_quality": quality,
        "grid": grid_info,
        "multiple_testing": {"n_tests": summary.attrs.get("n_tests"),
                             "alpha_bonferroni": summary.attrs.get("alpha_bonferroni")},
        "candidates": cand[["model", "h", "r2_oos", "ic", "hit", "paper_bps"]].to_dict("records"),
        "conclusion": conclusion(summary),
        "summary": json.loads(summary.to_json(orient="records")),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)


def conclusion(s):
    c = s[s.verdict_stat == "CANDIDATO"]
    if c.empty:
        return ("NESSUN SEGNALE DIMOSTRATO: nessun modello batte il random walk in modo "
                "significativo e stabile su dati fuori campione.")
    lines = [f"{r.model}@{r.h}s: R2_oos={r.r2_oos * 100:.3f}%, IC={r.ic:.3f}, "
             f"economico={r.verdict_econ}" for r in c.itertuples()]
    return ("SEGNALI CANDIDATI (da confermare su un periodo NUOVO mai usato): " + "; ".join(lines))
