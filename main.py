"""Forecast GTA VI trophy earn rates on Exophase at a future target.

Reads every ``*.in`` case file from $MAKEFUN_INPUT_DIR (else /input/) and
writes one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR
(else /output/).  Each forecast gives five quantiles (5/25/50/75/95) of
every trophy's earn rate in percent at the case target.

Model, per trophy
-----------------
An earn rate is earners / players-tracked.  Both move after launch: new
players dilute every rate down, while story progression pushes the
early-trophy rates back up.  On the logit scale the rates move roughly
linearly against log(1 + hours since release), so the forecast fits that
line and reads it at the target:

* the page's distinct states (consecutive identical reads collapsed,
  because Exophase only refreshes its counts every few hours) feed a
  least-squares fit of logit(rate) on log(1 + hours since release),
  using the most recent 48 hours of states when there are enough;
* the fitted slope is shrunk toward zero when few states exist and
  damped the further the target lies past the last observed state, and
  the line is re-centred on the window mean so shrinking the slope
  cannot drag the level;
* the spread combines a random-walk term growing with the horizon, the
  fit's residual scatter, and the slope's standard error times the
  extrapolation distance, floored and capped.

Two degenerate histories need their own handling.  When the page never
changed at all (every read identical for hours) no trend exists to fit,
and the page is stale: Exophase's next refresh reprices every rate, so
the median is the last rate scaled by the first-refresh dilution seen
on launch night (players 638 -> 3868, earners growing only slightly,
about a 0.20x factor) with a wide spread, except a trophy already at
100%, which a refresh leaves at 100%.  When the page is flat over only
a short span, no refresh is due yet and the last rate is held.

Only the standard library is used.  No network, no randomness, no
clock reads at forecast time: the same input always yields the same
output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard-normal quantiles for the 5/25/50/75/95 levels.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Earn rates are printed to 2 decimals, so 0% and 100% have no logit.
# Clamp into (0, 100) before taking logits, as the scorer does.
PCT_LO = 0.01
PCT_HI = 99.99

# Launch-night first refresh: players tracked 638 -> 3868 while the
# earners behind each rate had barely grown, so a rate repriced to
# about a fifth of its stale value.  Used only when the page in a case
# never moved at all.
STALE_REFRESH_FACTOR = 0.20
STALE_SPREAD = 0.6
STALE_MIN_SPAN_HOURS = 2.0

# Trend-fit tuning.
FIT_WINDOW_HOURS = 48.0
MAX_LOGIT_MOVE = 3.5       # never move a median more than this from the last rate
SPREAD_FLOOR = 0.45
SPREAD_CAP = 1.5
WALK_COEF = 0.35           # random-walk logit sigma per sqrt(day)
RESID_FLOOR = 0.15
EXTRAP_DAMP = 0.25         # slope damping per log-hour past the last state


def clamp_pct(value):
    """Clamp a percent into the range the logit can handle."""
    return min(PCT_HI, max(PCT_LO, value))


def logit_pct(value):
    """logit of a percent, with the scorer's 0.01/99.99 clamping."""
    p = clamp_pct(value) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    """Inverse logit back to a percent."""
    if value > 20.0:
        return PCT_HI
    if value < -20.0:
        return PCT_LO
    return 100.0 / (1.0 + math.exp(-value))


def parse_time(text):
    """Parse an ISO-8601 timestamp (date-only strings become midnight)."""
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def hours_since(moment, origin):
    """Hours from origin to moment, floored at zero."""
    return max(0.0, (moment - origin).total_seconds() / 3600.0)


def least_squares(xs, ys):
    """Fit y = a + b*x; return (a, b, resid_sd, slope_se)."""
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return y_bar, 0.0, 0.0, 0.0
    sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    resid_var = sum((y - (intercept + slope * x)) ** 2
                    for x, y in zip(xs, ys)) / max(n - 2, 1)
    slope_se = math.sqrt(resid_var / sxx) if n > 2 else 0.0
    return intercept, slope, math.sqrt(max(resid_var, 0.0)), slope_se


def distinct_states(case, release):
    """Collapse consecutive identical page reads into distinct states.

    Returns [(x, rates)] with x = log(1 + hours since release).
    A read that repeats the previous page exactly carries no new
    information: Exophase had not refreshed between the two.
    """
    states = []
    previous = None
    for read in case.get("history", []):
        rates = read.get("rates")
        if rates is None or "at" not in read:
            continue
        key = (tuple(rates), read.get("players_tracked"))
        if key == previous:
            continue
        previous = key
        moment = parse_time(read["at"])
        states.append((math.log1p(hours_since(moment, release)),
                       read["rates"]))
    return states


def forecast_trophy(last_rate, series, x_target, horizon_days, span_hours):
    """Five quantiles for one trophy.

    series: [(x, rate)] over the case's distinct page states.
    """
    if last_rate is None:
        last_rate = 0.5
    seen = {rate for _x, rate in series}
    if len(seen) <= 1:
        # The page never moved.  A long flat span means the displayed
        # counts are stale and the next refresh reprices them; a short
        # one means nothing is due yet.
        if last_rate >= 100.0:
            return [PCT_HI] * 5
        if span_hours >= STALE_MIN_SPAN_HOURS:
            centre = logit_pct(last_rate) + math.log(STALE_REFRESH_FACTOR)
            spread = STALE_SPREAD
        else:
            centre = logit_pct(last_rate)
            spread = 0.5 * math.sqrt(horizon_days)
        return [round(clamp_pct(inv_logit_pct(centre + z * spread)), 4)
                for z in Z_SCORES]

    # Trend fit on the recent window of states.
    x_max = max(x for x, _rate in series)
    hours_max = math.expm1(x_max)
    cut = math.log1p(max(0.0, hours_max - FIT_WINDOW_HOURS))
    window = [(x, rate) for x, rate in series if x >= cut]
    if len(window) < 3:
        window = list(series)
    xs = [x for x, _rate in window]
    ys = [logit_pct(rate) for _x, rate in window]
    _intercept, slope, resid_sd, slope_se = least_squares(xs, ys)
    count = len(window)
    x_bar = sum(xs) / count
    y_bar = sum(ys) / count

    # Shrink the slope when states are few, damp it past the last
    # observed state, and re-centre so the level comes from the data.
    slope *= count / (count + 3.0)
    beyond = max(0.0, x_target - x_max)
    slope *= 1.0 / (1.0 + EXTRAP_DAMP * beyond)
    centre = y_bar + slope * (x_target - x_bar)

    # Never let the extrapolation run away from the last observed rate.
    last_logit = logit_pct(last_rate)
    centre = max(last_logit - MAX_LOGIT_MOVE,
                 min(last_logit + MAX_LOGIT_MOVE, centre))
    centre = max(logit_pct(PCT_LO), min(logit_pct(PCT_HI), centre))

    spread = math.sqrt(
        (WALK_COEF * math.sqrt(horizon_days)) ** 2
        + max(resid_sd, RESID_FLOOR) ** 2
        + (slope_se * max(0.0, x_target - x_bar)) ** 2
    )
    spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread))
    return [round(clamp_pct(inv_logit_pct(centre + z * spread)), 4)
            for z in Z_SCORES]


def forecast_case(case):
    """Forecast every trophy of one case."""
    release = parse_time(case.get("game", {}).get("release_date",
                                                 "2026-11-19"))
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(hours_since(target, release))

    history = case.get("history", [])
    span_hours = 0.0
    if len(history) >= 2:
        span_hours = (parse_time(history[-1]["at"])
                      - parse_time(history[0]["at"])).total_seconds() / 3600.0
    states = distinct_states(case, release)
    last_rates = history[-1]["rates"] if history else []

    rows = []
    for index, trophy in enumerate(case["trophies"]):
        last = last_rates[index] if index < len(last_rates) else None
        series = [(x, rates[index]) for x, rates in states
                  if rates[index] is not None]
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(last, series, x_target,
                                          horizon_days, span_hours)})
    return {"trophies": rows}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(in_dir)
                   if name.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
