from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- orizzonti e cadenza decisionale (secondi) ---
    horizons: tuple = (5, 30, 120)
    stride: int = 5              # una decisione ogni N secondi
    # --- qualita' dato / griglia 1s ---
    max_stale: int = 10          # oltre N s senza tick il punto non e' valido
    session_gap: int = 60        # buco > N s = nuova sessione (chiusure, weekend)
    warmup: int = 600            # s di storia richiesti dopo l'inizio sessione
    spread_outlier: float = 20.0 # spread > N x mediana mobile -> tick scartato
    vol_span: int = 300          # EWMA volatilita' per secondo
    # --- modello analogico (sequenze simili) ---
    analog_len: int = 60         # lunghezza pattern in secondi
    analog_paa: int = 12         # punti dopo compressione PAA (deve dividere analog_len)
    analog_k: int = 200          # vicini
    analog_memory: int = 20000   # pattern in memoria (ring buffer)
    analog_min_mem: int = 1000
    # --- regressione online RLS ---
    rls_lambda: float = 0.9999   # fattore di oblio (1 = memoria infinita)
    rls_delta: float = 10.0
    min_train: int = 500         # aggiornamenti minimi prima di prevedere
    # --- decisione / astensione ---
    conf_margin: float = 0.05    # agisce solo se |p_up - 0.5| >= margin
    # --- validazione ---
    n_folds: int = 5
    bootstrap: int = 500
    alpha: float = 0.05
    seed: int = 42

    def check(self):
        assert self.analog_len % self.analog_paa == 0, "analog_paa deve dividere analog_len"
        assert max(self.horizons) <= self.warmup, "warmup deve essere >= orizzonte massimo"
        assert self.analog_len <= self.warmup and 300 <= self.warmup
        return self

    def to_dict(self):
        d = asdict(self)
        d["horizons"] = list(self.horizons)
        return d
