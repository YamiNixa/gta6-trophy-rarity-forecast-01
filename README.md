# GTA VI trophy rarity forecast

Forecast, for every Grand Theft Auto VI PlayStation 5 trophy, its earn
rate on Exophase one week after launch. Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01).

## Program

`main.py` (Python 3.13, standard library only) reads every `*.in` case
from `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one `<id>.out`
forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output/`): five
quantiles (5/25/50/75/95) of each trophy's earn rate in percent at the
case target. Deterministic, no network, runs in ~0.05 s per case.

## Model

An earn rate is earners / players-tracked. Both move after launch: new
players dilute every rate down, while story progression pushes the
early-trophy rates back up. On the logit scale the rates move roughly
linearly against log(1 + hours since release), so the forecast fits
that line per trophy and reads it at the target:

- distinct page states only (consecutive identical reads collapsed,
  since Exophase refreshes its counts every few hours), least-squares
  fit of logit(rate) on log(1 + hours since release) over the most
  recent 48 hours of states;
- slope shrunk toward zero when few states exist and damped past the
  last observed state, re-centred on the window mean;
- spread from a random-walk term growing with the horizon, the fit's
  residual scatter, and the slope's standard error times the
  extrapolation distance.

When the page never moved at all the counts are stale: the next
refresh reprices every rate, so the median is the last rate scaled by
the launch-night first-refresh dilution (players 638 -> 3868, about
0.20x) with a wide spread — except a trophy already at 100%, which a
refresh leaves at 100%.

## Local checks

Practice cases (answers proxied by the launch-night refresh, calibrated
so the house baseline scores its stated ~1.51):

| case | this entry | house baseline |
|------|-----------|----------------|
| p00  | 0.1196    | 1.4982         |
| p01  | 0.1196    | 1.4982         |

On a synthetic final-like case (2.5 days of reads, 4.5-day horizon):
0.2097 vs the house baseline's 0.3197. The model also beat the
baseline across 9 synthetic variants (slower/faster progression,
player growth, refresh cadence, read noise).
