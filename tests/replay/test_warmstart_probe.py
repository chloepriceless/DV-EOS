"""Dev probe for the Finding-1 PV store/export warm-start partition.

NOT a CI test — skipped unless ``EOS_PROBE_INPUT`` is set. Driven by
``scripts/replay/warmstart_arms.py`` (ga mode) and run directly (seed mode).

Two modes (``EOS_PROBE_MODE``):

* ``seed`` (LAYER-A litmus, deterministic) — sets up the optimizer like the
  canonical replay, then at the prepared, pre-GA state builds the warm-start seed
  with the partition gate OFF and ON in ONE process (toggling the module global
  ``_PV_CHARGE_WINDOW_ENABLED`` so both seeds run against the SAME prepared
  simulation + toolbox), evaluates each seed's fitness, and simulates each to
  record per-slot grid feed-in + SoC. Proves the substantive claim: does the
  partition make the seed FITNESS-SUPERIOR and store the CHEAPEST surplus /
  export the DEARER (INVARIANT A, keyed on the SIMULATED ``Netzeinspeisung``)?

* ``ga`` (DRIFT arm) — runs the REAL GA (``EOS_PROBE_NGEN`` gens) with the gate
  state taken from the import-time env, then records ``hof[0]``'s decoded plan
  simulated via ``evaluate_inner`` (start-slot-relative indexing). The driver
  loops RNG seeds × gate arms to measure whether the returned plan stores the
  cheap trough (the fix) or exports it (the drift bug).

Both record arrays indexed RELATIVE to ``_start_day_slot()`` (result[0] == start
slot), so absolute slot ``h`` maps to index ``h - start_slot``.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import akkudoktoreos.optimization.genetic.genetic as G
from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.cache import CacheEnergyManagementStore
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization, creator
from akkudoktoreos.optimization.genetic.geneticparams import (
    GeneticOptimizationParameters,
)
from akkudoktoreos.utils.datetimeutil import to_datetime

_INPUT = os.environ.get("EOS_PROBE_INPUT")
_MODE = os.environ.get("EOS_PROBE_MODE", "seed")  # "seed" (Layer-A) | "ga" (drift arm)


def _config(config_eos):
    interval = int(os.environ.get("EOS_PROBE_INTERVAL", "3600"))  # 900 = prod 15-min (192 slots)
    config_eos.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {
                "horizon_hours": 48,
                "interval": interval,
                "genetic": {
                    "individuals": 300,
                    "generations": 400,
                    "penalties": {"ev_soc_miss": 10, "ac_charge_break_even": 0},
                },
            },
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [
                    {"charge_rates": [0.0, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]},
                ],
            },
        }
    )


def _load_input():
    with Path(_INPUT).open("r") as f_in:
        return GeneticOptimizationParameters(**json.load(f_in))


@pytest.mark.skipif(not _INPUT or _MODE != "seed", reason="seed probe — EOS_PROBE_INPUT + MODE=seed")
def test_warmstart_seed_probe(config_eos: ConfigEOS):
    start_hour = int(os.environ.get("EOS_PROBE_START_HOUR", "10"))
    out_path = os.environ["EOS_PROBE_OUT"]
    _config(config_eos)
    input_data = _load_input()
    ems = get_ems(init=True)
    ems.set_start_datetime(to_datetime().set(hour=start_hour))
    CacheEnergyManagementStore().clear()
    opt = GeneticOptimization(fixed_seed=42)

    captured: dict = {}
    orig_optimize = GeneticOptimization.optimize

    def probe_optimize(self, start_solution=None, ngen: int = 200):
        n = self.total_slots
        sslot = self._start_day_slot()
        bat = self.simulation.battery
        revenue = np.asarray(self.simulation.elect_revenue_per_hour_arr, float)[:n]
        pv = np.asarray(self.simulation.pv_prediction_wh, float)[:n]
        load = np.asarray(self.simulation.load_energy_array, float)[:n]

        def build(flag: bool):
            G._PV_CHARGE_WINDOW_ENABLED = flag
            seed = self._greedy_discharge_seed()
            if seed is None:
                return None
            fitness = float(self.toolbox.evaluate(creator.Individual(list(seed)))[0])
            res = self.evaluate_inner(creator.Individual(list(seed)))
            return {
                "fitness": fitness,
                "feedin_wh": [float(x) for x in res["Netzeinspeisung_Wh_pro_Stunde"]],
                "soc_pct": [float(x) for x in res["akku_soc_pro_stunde"]],
                "charge_genes": [int(x) for x in seed[:n]],
            }

        captured["off"] = build(False)
        captured["on"] = build(True)
        G._PV_CHARGE_WINDOW_ENABLED = False
        surplus = [h for h in range(sslot, n) if pv[h] - load[h] > 0.0]
        order = sorted(surplus, key=lambda h: (revenue[h], h))
        captured["meta"] = {
            "start_slot": sslot, "n": n, "min_soc_wh": float(bat.min_soc_wh),
            "surplus_slots": surplus,
            "cheapest_surplus_slots": order[: max(1, len(order) // 3)],
            "dearest_surplus_slots": order[-max(1, len(order) // 3):],
            "revenue_eur_per_wh": [float(x) for x in revenue],
            "pv_wh": [float(x) for x in pv], "load_wh": [float(x) for x in load],
        }
        Path(out_path).write_text(json.dumps(captured, indent=2), encoding="utf-8")
        return orig_optimize(self, start_solution, ngen=1)

    viz = str(Path(out_path).with_suffix(".viz.pdf"))
    from akkudoktoreos.utils import visualize as _viz

    with patch.object(GeneticOptimization, "optimize", probe_optimize), patch(
        "akkudoktoreos.utils.visualize.prepare_visualize",
        side_effect=lambda parameters, results, *a, **k: _viz.prepare_visualize(
            parameters, results, filename=viz, **k
        ),
    ):
        opt.optimierung_ems(parameters=input_data, start_hour=start_hour, ngen=1)
    Path(viz).unlink(missing_ok=True)
    assert captured.get("on") is not None, "ON seed must build on the targeted day"


@pytest.mark.skipif(not _INPUT or _MODE != "ga", reason="ga arm probe — EOS_PROBE_INPUT + MODE=ga")
def test_warmstart_ga_arm(config_eos: ConfigEOS):
    start_hour = int(os.environ.get("EOS_PROBE_START_HOUR", "4"))
    ga_ngen = int(os.environ.get("EOS_PROBE_NGEN", "400"))
    seed = int(os.environ.get("EOS_PROBE_SEED", "1"))
    out_path = os.environ["EOS_PROBE_OUT"]
    _config(config_eos)
    input_data = _load_input()
    ems = get_ems(init=True)
    ems.set_start_datetime(to_datetime().set(hour=start_hour))
    CacheEnergyManagementStore().clear()
    opt = GeneticOptimization(fixed_seed=seed)

    captured: dict = {}
    orig_optimize = GeneticOptimization.optimize

    def probe_optimize(self, start_solution=None, ngen: int = 200):
        # Run the REAL GA with the import-time gate state, then record hof[0]'s
        # decoded plan simulated via evaluate_inner (start-slot-relative arrays).
        n = self.total_slots
        sslot = self._start_day_slot()
        result = orig_optimize(self, start_solution, ngen=ga_ngen)
        best = result[0]
        res = self.evaluate_inner(creator.Individual(list(best)))
        revenue = np.asarray(self.simulation.elect_revenue_per_hour_arr, float)[:n]
        pv = np.asarray(self.simulation.pv_prediction_wh, float)[:n]
        load = np.asarray(self.simulation.load_energy_array, float)[:n]
        captured.update({
            "start_slot": sslot, "n": n, "seed": seed, "ngen": ga_ngen,
            "gate_pv_charge_window": os.environ.get("EOS_PV_CHARGE_WINDOW", "0"),
            "elite_k": os.environ.get("EOS_PV_CHARGE_WINDOW_ELITE_K", "2"),
            "gesamtbilanz_eur": float(res["Gesamtbilanz_Euro"]),
            "feedin_wh": [float(x) for x in res["Netzeinspeisung_Wh_pro_Stunde"]],
            "soc_pct": [float(x) for x in res["akku_soc_pro_stunde"]],
            "charge_genes": [int(x) for x in best[:n]],
            "revenue_eur_per_wh": [float(x) for x in revenue],
            "pv_wh": [float(x) for x in pv], "load_wh": [float(x) for x in load],
        })
        Path(out_path).write_text(json.dumps(captured, indent=2), encoding="utf-8")
        return result

    viz = str(Path(out_path).with_suffix(".viz.pdf"))
    from akkudoktoreos.utils import visualize as _viz

    with patch.object(GeneticOptimization, "optimize", probe_optimize), patch(
        "akkudoktoreos.utils.visualize.prepare_visualize",
        side_effect=lambda parameters, results, *a, **k: _viz.prepare_visualize(
            parameters, results, filename=viz, **k
        ),
    ):
        opt.optimierung_ems(parameters=input_data, start_hour=start_hour, ngen=ga_ngen)
    Path(viz).unlink(missing_ok=True)
    assert captured.get("feedin_wh"), "GA arm must produce a plan"
