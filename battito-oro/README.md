# Battito dell'Oro — laboratorio di previsione XAU/USD

Laboratorio **offline** che replica i tick di XAU/USD in ordine temporale, produce previsioni
probabilistiche a 5 s / 30 s / 120 s e verifica, su dati mai usati per l'apprendimento,
se battono le previsioni banali. **Non invia ordini e non è un segnale operativo.**

Domanda a cui risponde: *esiste un segnale predittivo misurabile, stabile e superiore ai costi?*
La risposta di default è **no** finché i dati non dimostrano il contrario.

## Avvio rapido (Mac, Python ≥ 3.10)

```bash
cd battito-oro
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m battito selftest                       # 1) il laboratorio è onesto? (~3-5 min)
python -m battito run --synthetic momentum --days 2   # 2) demo su dati sintetici
python -m battito run --csv data/xauusd.csv --out results/run1   # 3) dati reali
```

### Dati reali gratuiti (tick Dukascopy)

```bash
npx dukascopy-node -i xauusd -from 2026-10-01 -to 2026-10-08 -t tick -f csv -dir data
python -m battito run --csv data/<file>.csv --out results/ott_w1
```

CSV accettati (rilevamento automatico): `timestamp,bid,ask` (ISO o epoch s/ms/us/ns),
export Dukascopy (`Gmt time,Ask,Bid,...`), oppure `timestamp,price`.
Consiglio: iniziare con **1 settimana** (~5 min di elaborazione), poi estendere.

Opzioni: `--from 2026-10-02 --to 2026-10-05`, `--horizons 5,30,120`, `--stride 5`,
`--models trend_1f,auto_rls,analog`.

## Architettura (modulare, un file = una responsabilità)

```
tick CSV ─► data.py ─► features.py ─► engine.py (replay) ─► evaluation.py ─► report.py
            pulizia     feature         modelli online       metriche e      summary.csv
            griglia 1s  causali +       aggiornati solo      verdetti        by_hour.csv
            sessioni    etichette       con esiti noti                       report.json
```

| Modulo | Cosa fa |
|---|---|
| `data.py` | parsing, duplicati, quote incrociate, outlier di spread, spike isolati, griglia 1 s, sessioni (buchi > 60 s), dato "stale" > 10 s = non valido |
| `features.py` | **AUTO**: rendimenti multi-scala (1-300 s), accelerazione, regime di volatilità, intensità tick, spread, ora. **TA**: MACD, RSI, Bollinger, Donchian, ROC. Tutte causali |
| `models.py` | `zero` (random walk), `trend_1f` (1 feature, momentum/reversion), `ta_rls`, `auto_rls` (regressione ricorsiva con oblio), `analog` (k vicini su sequenze simili, PAA, normalizzate per volatilità) |
| `engine.py` | replay: in T il modello riceve l'esito della previsione fatta in t solo se t+h ≤ T → **niente lookahead per costruzione**. Stesso codice riutilizzabile per un feed live in sola lettura |
| `evaluation.py` | R²_oos vs random walk con bootstrap a blocchi, IC e hit-rate su campioni non sovrapposti, Brier skill, stabilità per fold, Page-Hinkley, correzione Bonferroni, simulazione al netto dello spread |
| `selftest.py` | causalità, controllo negativo (random walk → nessun segnale), controllo positivo (momentum iniettato → trovato) |

## Come leggere i risultati

| Colonna | Significato | Soglia utile |
|---|---|---|
| `r2_oos` (%) | errore vs "non so" (random walk) | > 0 e `r2_lo` > 0 |
| `skill_trend` (%) | batte anche il momentum banale? | > 0 |
| `ic`, `ic_p` | correlazione previsione/esito, p-value | `ic_p` < alpha Bonferroni |
| `hit` (%) | direzione giusta quando agisce | > 50 con `hit_p` piccolo |
| `stable` (%) | fold temporali con R² > 0 | ≥ 80 |
| `ph_alarms` | cambi di regime rilevati sull'errore | confronta col `zero` |
| `paper_bps` | **simulazione** per decisione, al netto dello spread | > 0 con `paper_t` > 2 |
| `edge_spread` | movimento previsto / spread | < 1 = non sfruttabile |

Verdetto statistico: `CANDIDATO` solo se R²_oos > 0 con intervallo tutto positivo, IC significativo
dopo Bonferroni e stabile su ≥ 80 % dei fold. `CANDIDATO` **non** è una scoperta: va riconfermato
su un periodo successivo mai usato (hold-out finale).

Esempio su dati sintetici con momentum iniettato: i modelli trovano il segnale a 5 s e 30 s
(R² 2-3 %, verdetto CANDIDATO) ma il verdetto economico è `negativo`: il movimento previsto
è circa 0,1-0,5 volte lo spread. È esattamente il tipo di trappola che il laboratorio deve mostrare.

## Scelta di impostazione

Ricerca **guidata dai dati** (AUTO + analog) come strada principale, con l'analisi tecnica (TA)
come **gruppo di controllo** valutato con le stesse regole. Se TA e AUTO ottengono risultati
equivalenti, l'informazione è la stessa (momentum/volatilità) e non serve complessità aggiuntiva.

## Protocollo consigliato

1. `selftest` deve risultare SUPERATO.
2. Sviluppo su un mese (es. ottobre): esplora orizzonti, stride, modelli.
3. **Congela** configurazione e codice (commit).
4. Esegui una sola volta su un mese nuovo (es. novembre). Solo questo risultato conta.
5. Se ci sono CANDIDATI: verifica `by_hour.csv` (il segnale è concentrato in certe ore?)
   e `edge_spread` prima di qualunque ulteriore passo.

## Verifiche suggerite (anche da un'altra AI)

- Rileggere `engine.run`: l'unico punto in cui un esito entra in un modello è `q[0] + h <= T`.
- `features.build_context`: nessuna operazione deve usare indici > t (solo `ewm`, `rolling`, `_shift`).
- Provare `rls_lambda=1.0` (memoria infinita) vs `0.9999` per misurare la non stazionarietà.
- Sostituire il kNN di `Analog` con `stumpy` (motif/discord) o il modello RLS con `river`
  mantenendo la stessa interfaccia `predict(t)` / `update(t, y)`: il resto non cambia.
- Aggiungere costo di slippage oltre allo spread in `evaluate_one` (`pnl`).

## Prossimi passi possibili

- Adattatore live in sola lettura (WebSocket/REST) che alimenta lo stesso `engine` tick per tick.
- Pagina web (Netlify) che mostra le previsioni correnti con le loro incertezze.
- Salvataggio previsioni/esiti su Supabase per una validazione continua nel tempo.
