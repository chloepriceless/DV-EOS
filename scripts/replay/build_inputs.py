#!/usr/bin/env python3
"""Build per-day genetic optimizer input JSONs for the overnight-reserve replay.

Reads an hourly pipe-separated dump of the DVhub prod timeseries (price / PV /
load / battery SoC, Berlin wall-clock hours) and writes one
``GeneticOptimizationParameters`` JSON per (day, import-price-variant) for
``scripts/replay/ablate.py`` to run gate-OFF vs gate-ON on.

Faithfulness to Christin's prod setup (live-verified /v1/config 2026-06-19):
  - 43 kWh battery, min_soc 10 %, round-trip ~0.92 (charge/discharge 0.959 each),
    max charge 18 kW; inverter 29.7 kW with max_ac_charge_power_w=0 → NO grid
    charging (prod allowGridCharge=false: the battery only ever charges from PV).
  - Feed-in = spot price ×1 (prod tariff.feedInMode=spot), per-hour array.
  - Import price: two variants — "fix" = flat 26.9 ct/kWh gross (Christin's fixed
    tariff, FALL 1, what the eos-forecast-bridge pushes); "spot" = spot import
    (sensitivity, since the brief valued night re-buy at spot ~12-19 ct).
  - EV neutralized (initial 100 % / min 0 %) — the chosen days are EV-free.

APPROXIMATION (labelled in the report): PV and load use the *actual* measured
power as the forecast (perfect foresight). This is a backtest convenience; the
gate-OFF-vs-ON delta is unaffected because both runs see identical inputs — only
the absolute euro level is approximate. Spot price is genuinely day-ahead-known.

Real consumption/SoC data stays in /tmp (PII) — only this generic builder is
committed.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

CT_PER_KWH_TO_EUR_PER_WH = 1e-5  # ct/kWh × 1e-5 = EUR/Wh
FIXED_IMPORT_CT = 26.9  # Christin's fixed gross import tariff (FALL 1)
HORIZON = 48


def parse_psv(path: Path) -> dict[str, dict[str, float]]:
    series: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, ts, val = line.split("|")
        series.setdefault(key, {})[ts.strip()] = float(val)
    return series


def hourly_series(series: dict[str, float], hours: list[str], *, default=None,
                  ffill=True) -> list[float]:
    """Pull values for the given hour-keys; forward/back-fill gaps."""
    out: list[float | None] = [series.get(h) for h in hours]
    if ffill:
        # forward fill
        last = None
        for i, v in enumerate(out):
            if v is None:
                out[i] = last
            else:
                last = v
        # back fill leading None
        first = next((v for v in out if v is not None), default)
        out = [first if v is None else v for v in out]
    return [default if v is None else v for v in out]


def build_day(series, day: str, *, import_variant: str) -> dict:
    midnight = dt.datetime.strptime(day, "%Y-%m-%d")
    hours = [(midnight + dt.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M")
             for i in range(HORIZON)]

    spot_ct = hourly_series(series["price_ct_kwh"], hours, default=0.0)
    pv_w = hourly_series(series["pv_total_w"], hours, default=0.0)
    load_w = hourly_series(series["load_power_w"], hours, default=0.0)

    # initial SoC at the optimization start hour (10:00 of the day)
    soc_key = (midnight + dt.timedelta(hours=10)).strftime("%Y-%m-%d %H:%M")
    soc_series = series["battery_soc_pct"]
    initial_soc = soc_series.get(soc_key)
    if initial_soc is None:  # nearest available within the day
        for off in range(1, 11):
            for cand in (10 + off, 10 - off):
                k = (midnight + dt.timedelta(hours=cand)).strftime("%Y-%m-%d %H:%M")
                if k in soc_series:
                    initial_soc = soc_series[k]
                    break
            if initial_soc is not None:
                break
    initial_soc = int(round(initial_soc if initial_soc is not None else 50))

    feed_in = [round(p * CT_PER_KWH_TO_EUR_PER_WH, 10) for p in spot_ct]
    if import_variant == "fix":
        strompreis = [round(FIXED_IMPORT_CT * CT_PER_KWH_TO_EUR_PER_WH, 10)] * HORIZON
    elif import_variant == "spot":
        strompreis = [round(max(p, 0.0) * CT_PER_KWH_TO_EUR_PER_WH, 10) for p in spot_ct]
    else:
        raise ValueError(import_variant)

    return {
        "ems": {
            "preis_euro_pro_wh_akku": 0.0,
            "einspeiseverguetung_euro_pro_wh": feed_in,
            "gesamtlast": [round(x, 2) for x in load_w],
            "pv_prognose_wh": [round(x, 2) for x in pv_w],
            "strompreis_euro_pro_wh": strompreis,
        },
        # Exact prod EOS device config (live-verified from /v1/config, 2026-06-19).
        "pv_akku": {
            "device_id": "battery1",
            "capacity_wh": 43000,
            "initial_soc_percentage": initial_soc,
            "min_soc_percentage": 10,  # prod live = 10 (5% Victron blackout + ~5% EOS)
            "max_soc_percentage": 100,
            "charging_efficiency": 0.9591663046625439,   # prod RT ~0.92 (0.959^2)
            "discharging_efficiency": 0.9591663046625439,
            "max_charge_power_w": 18000,
            # NOTE: prod levelized_cost_of_storage_kwh=0.024 is NOT settable on the
            # input battery model -> omitted. Effect ~-0.05 EUR (extra cycled ~2 kWh
            # x 2.4 ct wear), i.e. the reported delta is a tiny overestimate.
        },
        "inverter": {
            "device_id": "inverter1",
            "max_power_wh": 29700,  # prod max_power_w=29700
            "battery_id": "battery1",
            "ac_to_dc_efficiency": 1.0,
            "dc_to_ac_efficiency": 1.0,
            "max_ac_charge_power_w": 0.0,  # prod: no grid charging (allowGridCharge=false)
        },
        "eauto": {
            "device_id": "ev1",
            "capacity_wh": 60000,
            "charging_efficiency": 0.95,
            "max_charge_power_w": 11040,
            "initial_soc_percentage": 100,  # full → inert (EV-free days)
            "min_soc_percentage": 0,
        },
        "_meta": {
            "day": day, "import_variant": import_variant, "initial_soc": initial_soc,
            "spot_ct": [round(p, 2) for p in spot_ct],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psv", default="/tmp/eos_replay_raw.psv")
    ap.add_argument("--out", default="/tmp/eos_replay")
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["fix", "spot"])
    args = ap.parse_args()

    series = parse_psv(Path(args.psv))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for day in args.days:
        for variant in args.variants:
            doc = build_day(series, day, import_variant=variant)
            meta = doc.pop("_meta")
            path = outdir / f"{day}_{variant}.json"
            path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            ev = [i for i, v in enumerate(meta["spot_ct"]) if 18 <= (i % 24) <= 22]
            print(f"{path.name}: soc0={meta['initial_soc']}% "
                  f"spot_midday(idx11-14)={[meta['spot_ct'][i] for i in (11,12,13,14)]} "
                  f"spot_eve(idx18-22)={[meta['spot_ct'][i] for i in (18,19,20,21,22)]}")


if __name__ == "__main__":
    main()
