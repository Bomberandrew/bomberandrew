# Contesto del progetto (passaggio di consegne)

Leggere insieme al README. Questo file riassume decisioni e risultati della prima fase,
svolta in una sessione cloud. Il lavoro ora prosegue sul server Hetzner `mar-auto`.

## Obiettivo
Laboratorio di ricerca (NON bot di trading): osservare il "battito" di XAU/USD, imparare
regolarità e verificare su dati mai visti se battono previsioni banali, al netto dei costi.
Andrea (utente) lavora con Claude e ChatGPT in parallelo: risposte tecniche, in italiano,
pratiche, con codice completo e verifiche trasferibili all'altra AI.

## Decisioni prese
- Strada principale guidata dai dati (`auto_rls`, `analog`), analisi tecnica (`ta_rls`) come
  gruppo di controllo, baseline `zero` (random walk) e `trend_1f`. Stesse regole per tutti.
- Solo numpy + pandas (niente STUMPY/River per ora: stessa interfaccia predict/update,
  sostituibili in seguito).
- Validazione: replay walk-forward senza lookahead, R2_oos con bootstrap a blocchi, IC su
  campioni non sovrapposti, Bonferroni, stabilita' per fold, Page-Hinkley, simulazione netta
  di spread. Verdetto CANDIDATO solo se tutto supera le soglie; poi conferma su hold-out.
- Protocollo: sviluppa su un periodo, congela il codice, valuta UNA volta su periodo nuovo.

## Risultati finora
1. Selftest SUPERATO: causalita' OK; random walk -> nessun falso positivo; momentum
   iniettato -> trovato a 5s e 30s (R2 1.4-2.9%), ma economicamente negativo (edge < spread).
2. Twelve Data, barre 1 min, 1 set - 20 nov 2026 (orizzonti 1/5/15 min, hold-out da 1 nov):
   NESSUN SEGNALE. Unico effetto: auto_rls@1min IC 0.038 (0.22 bps lordi vs 0.8 bps costo).
   Il feed Twelve Data ha barre anche a mercato chiuso (rimosse con market_hours) e
   un'autocorrelazione lag-2 di -0.095 che NON compare su Dukascopy nello stesso periodo
   (+0.048): probabile artefatto di aggregazione del fornitore.
3. Dukascopy tick reali, solo 34 ore (2-3 nov 2026), orizzonti 5/30/120 s: NESSUN SEGNALE.
   Spread mediano reale 1.39 bps; movimento previsto 0.004-0.26 volte lo spread.

## In corso (al 21 nov 2026, ~16:30 UTC)
Sessione tmux `battito` sul server: download Dukascopy 2-13 nov (lento per limiti 429,
alcune ore), poi `battito run --holdout-from 2026-11-09 --out results/duka_2w`.
NON cancellare run.log ne' fermare la sessione finche' non compare "salvato in".
Sessione tmux `claude`: Remote Control.

## Note operative
- Dukascopy blocca IP con raffiche di richieste: downloader sequenziale, pausa 3 s,
  backoff su 429, cache in data/duka/, salta le ore del weekend.
- Chiave Twelve Data: solo via `export TWELVEDATA_API_KEY=...`; la chiave condivisa in chat
  va rigenerata.
- Server: riavvio per nuovo kernel in sospeso, farlo solo a esperimento finito.

## Prossimi passi proposti
1. Analizzare results/duka_2w (summary.csv, summary_holdout.csv, by_hour.csv).
2. Se nessun segnale: provare orizzonti piu' lunghi (5-60 min) su tick, filtri per sessione
   (apertura Londra/NY, news), e feature di microstruttura (squilibrio bid/ask, volumi tick).
3. Solo con un CANDIDATO confermato su hold-out: osservatore live in sola lettura sul server
   che usa lo stesso engine, con previsioni salvate (es. Supabase) e validazione continua.
