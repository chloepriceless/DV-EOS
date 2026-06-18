#!/usr/bin/env python3
"""Drive the overnight-reserve replay ablation: gate-OFF vs gate-ON per day.

For each input JSON (a day × import-price variant) this runs the genetic
optimizer twice in *separate processes* — once with the price-aware reserve gate
OFF (current prod = price-blind reserve) and once ON (the Finding-2 fix) — with
all other DV gates in their prod-default ON state and the same fixed seed / GA
budget. It then diffs the two plans: euro balance, grid export/import, and the
evening SoC trajectory. The gate is read at import time, hence the subprocess.

Usage:
  python scripts/replay/ablate.py --inputs /tmp/eos_replay/2026-06-01_fix.json ... \
      --ngen 400 --individuals 300 --report /tmp/eos_replay/report.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = "tests/replay/test_reserve_replay.py"

# Prod-default DV gates (all ON); only the price-aware reserve gate is toggled.
BASE_GATES = {
    "EOS_OVERNIGHT_RESERVE": "1",
    "EOS_BATTERY_GRID_EXPORT": "1",
    "EOS_SELF_CONSUMPTION_PRIORITY": "1",
}


def run(input_path: str, price_aware: bool, ngen: int, individuals: int,
        start_hour: int, seed: int) -> dict:
    tag = "on" if price_aware else "off"
    out = str(Path(input_path).with_suffix("")) + f".result_{tag}.json"
    env = {
        **os.environ, **BASE_GATES,
        "EOS_RESERVE_PRICE_AWARE": "1" if price_aware else "0",
        "EOS_REPLAY_INPUT": input_path,
        "EOS_REPLAY_OUT": out,
        "EOS_REPLAY_NGEN": str(ngen),
        "EOS_REPLAY_INDIVIDUALS": str(individuals),
        "EOS_REPLAY_START_HOUR": str(start_hour),
        "EOS_REPLAY_SEED": str(seed),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", HARNESS, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not Path(out).exists():
        raise RuntimeError(
            f"replay failed for {input_path} (price_aware={price_aware}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-1000:]}"
        )
    return json.loads(Path(out).read_text())


def summarize(off: dict, on: dict, start_hour: int) -> dict:
    # Gesamtbilanz_Euro = cost - revenue (lower = better). Fix savings = OFF - ON.
    savings = off["Gesamtbilanz_Euro"] - on["Gesamtbilanz_Euro"]
    # Evening = plan indices for absolute hours 18..24 -> relative to start_hour.
    def rel(h):
        return h - start_hour
    eve = slice(rel(18), rel(25))
    soc_off, soc_on = off["akku_soc_pro_stunde"], on["akku_soc_pro_stunde"]
    exp_off = sum(off["Netzeinspeisung_Wh_pro_Stunde"])
    exp_on = sum(on["Netzeinspeisung_Wh_pro_Stunde"])
    imp_off = sum(off["Netzbezug_Wh_pro_Stunde"])
    imp_on = sum(on["Netzbezug_Wh_pro_Stunde"])
    return {
        "bilanz_off": round(off["Gesamtbilanz_Euro"], 4),
        "bilanz_on": round(on["Gesamtbilanz_Euro"], 4),
        "fix_savings_eur": round(savings, 4),
        "revenue_off": round(off["Gesamteinnahmen_Euro"], 4),
        "revenue_on": round(on["Gesamteinnahmen_Euro"], 4),
        "export_wh_off": round(exp_off), "export_wh_on": round(exp_on),
        "export_delta_wh": round(exp_on - exp_off),
        "import_wh_off": round(imp_off), "import_wh_on": round(imp_on),
        "soc_evening_off": [round(x, 1) for x in soc_off[eve]],
        "soc_evening_on": [round(x, 1) for x in soc_on[eve]],
        "soc_min_off": round(min(soc_off), 1), "soc_min_on": round(min(soc_on), 1),
        "soc_end_off": round(soc_off[-1], 1), "soc_end_on": round(soc_on[-1], 1),
        "identical": off["Gesamtbilanz_Euro"] == on["Gesamtbilanz_Euro"]
        and soc_off == soc_on,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--ngen", type=int, default=400)
    ap.add_argument("--individuals", type=int, default=300)
    ap.add_argument("--start-hour", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default="/tmp/eos_replay/report.json")
    args = ap.parse_args()

    report = {}
    for inp in args.inputs:
        name = Path(inp).stem
        print(f"\n=== {name} (ngen={args.ngen}, ind={args.individuals}) ===", flush=True)
        off = run(inp, False, args.ngen, args.individuals, args.start_hour, args.seed)
        on = run(inp, True, args.ngen, args.individuals, args.start_hour, args.seed)
        s = summarize(off, on, args.start_hour)
        report[name] = s
        verdict = "NO-OP (identical)" if s["identical"] else f"fix saves {s['fix_savings_eur']:+.3f} €"
        print(f"  bilanz OFF={s['bilanz_off']:+.3f}€  ON={s['bilanz_on']:+.3f}€  -> {verdict}")
        print(f"  export OFF={s['export_wh_off']}Wh ON={s['export_wh_on']}Wh (Δ{s['export_delta_wh']:+})  "
              f"min-SoC OFF={s['soc_min_off']}% ON={s['soc_min_on']}%")
        print(f"  SoC evening OFF={s['soc_evening_off']}")
        print(f"  SoC evening ON ={s['soc_evening_on']}")

    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
