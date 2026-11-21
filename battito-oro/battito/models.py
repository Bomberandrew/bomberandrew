"""Modelli online. Interfaccia comune:
    predict(t) -> (mu_z, p_up) | None   (None = non pronto / astensione strutturale)
    update(t, y_z)                       (chiamato SOLO quando l'esito e' noto: t + h <= adesso)
mu_z = rendimento atteso normalizzato per volatilita' (y / (sigma*sqrt(h))).
"""
from math import erf, sqrt
import numpy as np


def ncdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


class ZeroModel:
    """Random walk: il prossimo movimento e' imprevedibile. Riferimento per R2_oos."""
    def __init__(self, *a, **k):
        pass

    def predict(self, t):
        return 0.0, 0.5

    def update(self, t, y):
        pass


class RLS:
    """Regressione lineare ricorsiva con oblio esponenziale (si adatta ai cambi di regime)."""
    def __init__(self, X, cfg):
        self.X = X
        d = X.shape[1] + 1
        self.d = d
        self.w = np.zeros(d)
        self.P = np.eye(d) * cfg.rls_delta
        self.lam = cfg.rls_lambda
        self.rv = 1.0
        self.n = 0
        self.min_train = cfg.min_train

    def _phi(self, t):
        x = np.empty(self.d)
        x[0] = 1.0
        x[1:] = self.X[t]
        return x

    def predict(self, t):
        if self.n < self.min_train:
            return None
        mu = float(self._phi(t) @ self.w)
        mu = max(-3.0, min(3.0, mu))
        return mu, ncdf(mu / sqrt(self.rv))

    def update(self, t, y):
        x = self._phi(t)
        Px = self.P @ x
        g = Px / (self.lam + x @ Px)
        e = y - x @ self.w
        self.rv = 0.995 * self.rv + 0.005 * e * e
        self.w += g * e
        self.P = (self.P - np.outer(g, Px)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)
        tr = np.trace(self.P)
        cap = 1e3 * self.d
        if tr > cap:
            self.P *= cap / tr
        self.n += 1


class Analog:
    """Sequenze simili: k vicini piu' prossimi tra i pattern passati con esito gia' noto.
    Dichiara incertezza tramite p_up = frazione (smussata) di vicini seguiti da rialzo."""
    def __init__(self, patterns, pos, cfg):
        self.pat = patterns       # (nD, P)
        self.pos = pos            # t -> riga in patterns
        M = cfg.analog_memory
        self.mem = np.zeros((M, patterns.shape[1]))
        self.out = np.zeros(M)
        self.cnt = 0
        self.ptr = 0
        self.M = M
        self.k = cfg.analog_k
        self.min_mem = cfg.analog_min_mem

    def predict(self, t):
        if self.cnt < self.min_mem:
            return None
        q = self.pat[self.pos[t]]
        m = self.mem[: self.cnt]
        d = ((m - q) ** 2).sum(axis=1)
        k = min(self.k, self.cnt)
        idx = np.argpartition(d, k - 1)[:k]
        f = self.out[: self.cnt][idx]
        p = (np.count_nonzero(f > 0) + 0.5 * np.count_nonzero(f == 0) + 1.0) / (k + 2.0)
        return float(f.mean()), p

    def update(self, t, y):
        self.mem[self.ptr] = self.pat[self.pos[t]]
        self.out[self.ptr] = y
        self.ptr = (self.ptr + 1) % self.M
        self.cnt = min(self.cnt + 1, self.M)


def build_models(ctx, patterns, pos, cfg, h, only=None):
    """Restituisce [(nome, modello)] per l'orizzonte h."""
    specs = [
        ("zero", lambda: ZeroModel()),
        ("trend_1f", lambda: RLS(ctx.past_z[h][:, None], cfg)),  # 1 sola feature: momentum/reversion
        ("ta_rls", lambda: RLS(ctx.X_ta, cfg)),                  # analisi tecnica classica
        ("auto_rls", lambda: RLS(ctx.X_auto, cfg)),              # descrittori grezzi
        ("analog", lambda: Analog(patterns, pos, cfg)),           # sequenze simili
    ]
    return [(n, f()) for n, f in specs if only is None or n in only or n == "zero"]
