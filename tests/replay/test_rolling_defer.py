"""Rolling per-step harness for the Finding-1 morning charge-timing defer.

NOT a CI test — skipped unless ``EOS_ROLL_CSV`` is set. Driven by
``scripts/replay/rolling_defer.py``.

The receding-horizon bug only manifests across the rolling 15-min re-opt, which
single-shot cannot reproduce (validated 2026-06-19). This harness replays the
REAL per-step prod state from Völtchen's data anchor
(``finding1-rolling-input-2026-06-18.csv``: per-15-min ``soc_seed_pct`` = prod's
actual SoC, ``feedin_ct_dayahead`` = the constant day-ahead spot EOS saw all day,
``pv_realized_w``, and ``setpoint_w_actual`` = prod's actuation = GROUND TRUTH).

At each step T it builds the optimizer input as prod saw it (PV=realized →
perfect-foresight, the SHARPEST test: if the defer emerges even with perfect
foresight, it is purely the commit-first-slot rolling mechanism, not forecast
error), seeds the REAL ``soc_seed_pct[T]``, runs ``optimize()`` at prod-exact
params, and reads slot-T's committed action from the solution's per-slot grid
feed-in (``Netzeinspeisung_Wh_pro_Stunde[0]`` — start-slot-relative, so [0] == T):
``> 0`` ⇒ EXPORTED the surplus (deferred charging, the bug), ``≈ 0`` ⇒ STORED it.

gate-OFF must reproduce prod's export defer through the cheap trough; gate-ON
must instead store the trough. The gate is read at import → one arm per process.
"""
import csv
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.cache import CacheEnergyManagementStore
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization
from akkudoktoreos.optimization.genetic.geneticparams import (
    GeneticOptimizationParameters,
)
from akkudoktoreos.utils.datetimeutil import to_datetime

_CSV = os.environ.get("EOS_ROLL_CSV")
CT_TO_EUR_PER_WH = 1e-5
FIXED_IMPORT_CT = 26.9
FLAT_LOAD_W = 800.0  # 06-18 was a low-load day; surplus is PV-dominated (~18 kW).
SLOTS_PER_DAY = 96  # 24h × 15-min, slot-of-day aligned from 00:00 UTC.


def _load_day(csv_path: str):
    """Build slot-of-day-aligned (00:00 UTC = slot 0) 96-slot 15-min arrays from
    the per-step CSV. Energy arrays are Wh-per-slot (W × 0.25); price/feed-in are
    per-Wh rates. Missing edge slots (pre-dawn / post-data) are PV=0, feed-in
    edge-filled. Returns arrays + the real SoC per slot + prod setpoint."""
    rows = list(csv.DictReader(open(csv_path)))
    pv = [0.0] * SLOTS_PER_DAY
    feedin_ct = [None] * SLOTS_PER_DAY
    soc = [None] * SLOTS_PER_DAY
    setpoint = [None] * SLOTS_PER_DAY
    for r in rows:
        t = r["slot_utc"]  # YYYY-MM-DDTHH:MM
        hh, mm = int(t[11:13]), int(t[14:16])
        i = hh * 4 + mm // 15
        pv[i] = float(r["pv_realized_w"]) * 0.25  # W → Wh per 15-min slot
        feedin_ct[i] = float(r["feedin_ct_dayahead"])
        soc[i] = float(r["soc_seed_pct"])
        setpoint[i] = float(r["setpoint_w_actual"])
    # edge-fill feed-in (nearest known) so padded night slots carry a sane rate.
    known = [i for i, v in enumerate(feedin_ct) if v is not None]
    lo, hi = known[0], known[-1]
    for i in range(SLOTS_PER_DAY):
        if feedin_ct[i] is None:
            feedin_ct[i] = feedin_ct[lo] if i < lo else feedin_ct[hi]
    # Optionally replicate day 1 as day 2..N to give the optimizer a multi-day
    # (48h+) horizon — needed to test the under-determination hypothesis (a):
    # over 2 PV days, storing TODAY's specific cheap trough is under-determined
    # (tomorrow can fill the pack too), which may let the GA defer/export it.
    days = int(os.environ.get("EOS_ROLL_DAYS", "1"))
    pv = pv * days
    feedin_ct = feedin_ct * days
    load = [FLAT_LOAD_W * 0.25] * (SLOTS_PER_DAY * days)
    imp = [FIXED_IMPORT_CT * CT_TO_EUR_PER_WH] * (SLOTS_PER_DAY * days)
    feed = [c * CT_TO_EUR_PER_WH for c in feedin_ct]
    return dict(pv=pv, load=load, imp=imp, feed=feed, feedin_ct=feedin_ct,
                soc=soc + [None] * (SLOTS_PER_DAY * (days - 1)),
                setpoint=setpoint + [None] * (SLOTS_PER_DAY * (days - 1)), known=(lo, hi))


def _params(day, initial_soc_pct):
    return GeneticOptimizationParameters(**{
        "ems": {
            "preis_euro_pro_wh_akku": 0.0,
            "einspeiseverguetung_euro_pro_wh": day["feed"],
            "gesamtlast": [round(x, 2) for x in day["load"]],
            "pv_prognose_wh": [round(x, 2) for x in day["pv"]],
            "strompreis_euro_pro_wh": day["imp"],
        },
        "pv_akku": {
            "device_id": "battery1", "capacity_wh": 43000,
            "initial_soc_percentage": int(round(initial_soc_pct)),
            "min_soc_percentage": 10, "max_soc_percentage": 100,
            "charging_efficiency": 0.9219544457292887,
            "discharging_efficiency": 0.9219544457292887,
            "max_charge_power_w": 18000,
        },
        "inverter": {
            "device_id": "inverter1", "max_power_wh": 29700, "battery_id": "battery1",
            "ac_to_dc_efficiency": 1.0, "dc_to_ac_efficiency": 1.0,
            "max_ac_charge_power_w": 0.0,
        },
        "eauto": {
            "device_id": "ev1", "capacity_wh": 60000, "charging_efficiency": 0.95,
            "max_charge_power_w": 11040, "initial_soc_percentage": 100,
            "min_soc_percentage": 0,
        },
    })


@pytest.mark.skipif(not _CSV, reason="rolling defer harness — set EOS_ROLL_CSV to run")
def test_rolling_defer(config_eos: ConfigEOS):
    ngen = int(os.environ.get("EOS_ROLL_NGEN", "400"))
    seed = int(os.environ.get("EOS_ROLL_SEED", "1"))
    step_lo = int(os.environ.get("EOS_ROLL_STEP_LO", "36"))   # 09:00 UTC slot-of-day
    step_hi = int(os.environ.get("EOS_ROLL_STEP_HI", "48"))   # 12:00 UTC slot-of-day
    step_every = int(os.environ.get("EOS_ROLL_STEP_EVERY", "1"))
    out_path = os.environ["EOS_ROLL_OUT"]

    days = int(os.environ.get("EOS_ROLL_DAYS", "1"))
    pred_hours = 24 * days
    config_eos.merge_settings_from_dict({
        "general": {"timezone": "UTC"},  # so start_datetime maps directly to slot-of-day
        "prediction": {"hours": pred_hours},
        "optimization": {
            "horizon_hours": pred_hours, "interval": 900,
            "genetic": {"individuals": 300, "generations": ngen,
                        "penalties": {"ev_soc_miss": 10, "ac_charge_break_even": 0}},
        },
        "devices": {"max_electric_vehicles": 1,
                    "electric_vehicles": [{"charge_rates": [0.0, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]}]},
    })

    day = _load_day(_CSV)
    from akkudoktoreos.utils import visualize as _viz
    trace = []
    for T in range(step_lo, step_hi + 1, step_every):
        if day["soc"][T] is None:
            continue
        params = _params(day, day["soc"][T])
        hh, mm = divmod(T * 15, 60)
        ems = get_ems(init=True)
        ems.set_start_datetime(to_datetime().set(hour=hh, minute=mm, second=0))
        CacheEnergyManagementStore().clear()
        opt = GeneticOptimization(fixed_seed=seed)
        viz = str(Path(out_path).with_suffix(f".{T}.viz.pdf"))
        with patch("akkudoktoreos.utils.visualize.prepare_visualize",
                   side_effect=lambda parameters, results, *a, **k: _viz.prepare_visualize(
                       parameters, results, filename=viz, **k)):
            sol = opt.optimierung_ems(parameters=params, start_hour=hh, ngen=ngen)
        Path(viz).unlink(missing_ok=True)
        r = sol.result
        # result arrays are start-slot-relative → index 0 == slot T (the committed slot).
        export0 = float(r.Netzeinspeisung_Wh_pro_Stunde[0])
        soc0_res = float(r.akku_soc_pro_stunde[0]) if len(r.akku_soc_pro_stunde) else None
        surplus_wh = max(day["pv"][T] - day["load"][T], 0.0)
        stored = export0 < 0.5 * surplus_wh and surplus_wh > 100  # stored most of the surplus
        trace.append({
            "slot": T, "utc": f"{hh:02d}:{mm:02d}",
            "feedin_ct": round(day["feedin_ct"][T], 2),
            "soc_seed_pct": day["soc"][T],
            "surplus_wh": round(surplus_wh),
            "my_export_wh": round(export0),
            "my_decision": "STORE" if stored else "EXPORT",
            "prod_setpoint_w": day["setpoint"][T],
            "prod_action": "EXPORT" if (day["setpoint"][T] is not None and day["setpoint"][T] < -1000) else "hold/charge",
        })
        Path(out_path).write_text(json.dumps(
            {"gate_pv_charge_window": os.environ.get("EOS_PV_CHARGE_WINDOW", "0"),
             "ngen": ngen, "seed": seed, "trace": trace}, indent=2), encoding="utf-8")
    assert trace, "no steps run"
