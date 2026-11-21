"""Controlli di onesta'. Se uno fallisce, i risultati su dati reali NON sono affidabili.
  1. causalita': le feature in t non cambiano se si tagliano i dati dopo t
  2. controllo negativo: su random walk nessun modello deve risultare CANDIDATO
  3. controllo positivo: con momentum iniettato almeno un modello deve trovarlo
"""
import numpy as np

from .config import Config
from . import data, features, synthetic
from .__main__ import pipeline


def _ctx(ticks, cfg):
    t, _ = data.clean_ticks(ticks, cfg)
    g, _ = data.build_grid(t, cfg)
    return g, features.build_context(g, cfg)


def test_causality(cfg):
    ticks = data.load_ticks_df(synthetic.generate(days=1, kind="momentum", seed=1))
    g, full = _ctx(ticks, cfg)
    cut = g.index[len(g) // 2]
    _, part = _ctx(ticks[ticks.ts < cut + np.timedelta64(1, "s")].reset_index(drop=True), cfg)
    n = len(part.logmid) - 1
    ok = (np.allclose(full.X_auto[:n], part.X_auto[:n])
          and np.allclose(full.X_ta[:n], part.X_ta[:n])
          and np.allclose(full.sigma[:n], part.sigma[:n]))
    print(f"[causalita'] {'OK' if ok else 'FALLITO'}")
    return ok


def main():
    cfg = Config().check()
    res = [test_causality(cfg)]
    print("[controllo negativo] random walk, 3 giorni...")
    s, _ = pipeline(data.load_ticks_df(synthetic.generate(days=3, kind="rw", seed=7)),
                    cfg, verbose=False)
    neg = not (s.verdict_stat == "CANDIDATO").any()
    print(s[["model", "h", "r2_oos", "ic", "ic_p", "verdict_stat"]].to_string(index=False))
    print(f"[controllo negativo] {'OK' if neg else 'FALLITO (falso positivo)'}")
    res.append(neg)
    print("[controllo positivo] momentum iniettato, 3 giorni...")
    s, _ = pipeline(data.load_ticks_df(synthetic.generate(days=3, kind="momentum", seed=7)),
                    cfg, verbose=False)
    pos = (s.verdict_stat == "CANDIDATO").any()
    print(s[["model", "h", "r2_oos", "ic", "ic_p", "verdict_stat"]].to_string(index=False))
    print(f"[controllo positivo] {'OK' if pos else 'FALLITO (segnale non trovato)'}")
    res.append(pos)
    print("SELFTEST", "SUPERATO" if all(res) else "NON SUPERATO")
    return 0 if all(res) else 1
