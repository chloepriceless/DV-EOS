"""Dev replay harness for the price-aware overnight reserve fix (Finding-2).

This is NOT a CI test — it is skipped unless ``EOS_REPLAY_INPUT`` is set. It is
driven by ``scripts/replay/ablate.py``, which invokes it once per (day,
gate-config) in a *separate process*, so the import-time env gates
(``EOS_RESERVE_PRICE_AWARE`` and friends, read in ``genetic.py`` at import) take
effect per run.

It reuses the full pytest ``config_eos`` fixture so the EOS config / singleton
plumbing is set up exactly like the canonical optimizer tests, loads a single
day's ``GeneticOptimizationParameters`` JSON, runs the genetic optimizer
deterministically (fixed seed), and writes the resulting plan's EUR balance plus
the SoC / grid-export / grid-import trajectory to ``EOS_REPLAY_OUT`` for the
driver to diff gate-OFF vs gate-ON.

Env knobs:
  EOS_REPLAY_INPUT       path to the day's optimize-input JSON  (required)
  EOS_REPLAY_OUT         path to write the result JSON           (required)
  EOS_REPLAY_START_HOUR  optimization start hour (default 10)
  EOS_REPLAY_NGEN        GA generations (default 400)
  EOS_REPLAY_INDIVIDUALS GA population (default 300)
  EOS_REPLAY_SEED        GA fixed seed (default 42)
  + the genetic gate envs (EOS_RESERVE_PRICE_AWARE, EOS_OVERNIGHT_RESERVE, ...)
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.cache import CacheEnergyManagementStore
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization
from akkudoktoreos.optimization.genetic.geneticparams import (
    GeneticOptimizationParameters,
)
from akkudoktoreos.utils.datetimeutil import to_datetime

_INPUT = os.environ.get("EOS_REPLAY_INPUT")


@pytest.mark.skipif(not _INPUT, reason="replay harness — set EOS_REPLAY_INPUT to run")
def test_replay(config_eos: ConfigEOS):
    start_hour = int(os.environ.get("EOS_REPLAY_START_HOUR", "10"))
    ngen = int(os.environ.get("EOS_REPLAY_NGEN", "400"))
    individuals = int(os.environ.get("EOS_REPLAY_INDIVIDUALS", "300"))
    seed = int(os.environ.get("EOS_REPLAY_SEED", "42"))
    out_path = os.environ["EOS_REPLAY_OUT"]

    # Match the canonical optimizer test config: 48h horizon, real GA budget,
    # one EV with the standard charge-rate ladder (the input neutralizes it).
    config_eos.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {
                "horizon_hours": 48,
                "genetic": {
                    "individuals": individuals,
                    "generations": ngen,
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

    with Path(_INPUT).open("r") as f_in:
        input_data = GeneticOptimizationParameters(**json.load(f_in))

    ems = get_ems(init=True)
    ems.set_start_datetime(to_datetime().set(hour=start_hour))
    CacheEnergyManagementStore().clear()

    opt = GeneticOptimization(fixed_seed=seed)

    # The optimizer renders a visualization PDF as a side-effect; redirect it to a
    # throwaway temp file so the replay leaves no artifacts behind.
    viz = str(Path(out_path).with_suffix(".viz.pdf"))
    from akkudoktoreos.utils import visualize as _viz

    with patch(
        "akkudoktoreos.utils.visualize.prepare_visualize",
        side_effect=lambda parameters, results, *a, **k: _viz.prepare_visualize(
            parameters, results, filename=viz, **k
        ),
    ):
        sol = opt.optimierung_ems(parameters=input_data, start_hour=start_hour, ngen=ngen)

    r = sol.result
    out = {
        "input": _INPUT,
        "start_hour": start_hour,
        "ngen": ngen,
        "individuals": individuals,
        "seed": seed,
        "gate_price_aware": os.environ.get("EOS_RESERVE_PRICE_AWARE", "0"),
        "gate_overnight_reserve": os.environ.get("EOS_OVERNIGHT_RESERVE", "1"),
        "Gesamtbilanz_Euro": r.Gesamtbilanz_Euro,
        "Gesamteinnahmen_Euro": r.Gesamteinnahmen_Euro,
        "Gesamtkosten_Euro": r.Gesamtkosten_Euro,
        "akku_soc_pro_stunde": list(r.akku_soc_pro_stunde),
        "Netzeinspeisung_Wh_pro_Stunde": list(r.Netzeinspeisung_Wh_pro_Stunde),
        "Netzbezug_Wh_pro_Stunde": list(r.Netzbezug_Wh_pro_Stunde),
    }
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    try:
        Path(viz).unlink()
    except OSError:
        pass
