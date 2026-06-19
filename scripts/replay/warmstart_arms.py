#!/usr/bin/env python3
"""Drive the Finding-1 warm-start drift validation: gate arms × N RNG seeds.

For the targeted high-PV crash day (2026-06-18), the receding-horizon bug stores
the DEAR morning surplus and exports the CHEAP midday trough (backwards). The
coherent store/export partition + selBest elitism should make the GA reliably
STORE the cheap trough and EXPORT the dear morning. Started in the MORNING
(before the trough), this exercises whether the GA holds the partitioned
front-load or drifts off it — the §4b tractable proxy for the rolling+stochastic
harness (the full rolling loop is compute-prohibitive at ngen=400 before the PV
ramp; this is the time-boxed G1, transparently noted).

Each run drives ``tests/replay/test_warmstart_probe.py`` in ``ga`` mode in a
SEPARATE process (the gate is read at import) and reads the recorded ``hof[0]``
plan (simulated ``Netzeinspeisung``, start-slot-relative arrays).

Arms (reserve price-aware gate held ON = prod state for all):
  A_off                 EOS_PV_CHARGE_WINDOW=0                  (today's behaviour)
  B_partition           EOS_PV_CHARGE_WINDOW=1  ELITE_K=0       (partition only)
  C_partition_elitism   EOS_PV_CHARGE_WINDOW=1  ELITE_K=2       (partition + elitism)

Metric (keyed on the SIMULATED Netzeinspeisung per INVARIANT A, NOT the gene):
  trough_export = Σ feed-in over the cheapest daytime surplus (abs) slots,
  indexed by (abs - start_slot). "trough stored" if below the threshold.

Usage:
  python scripts/replay/warmstart_arms.py --input /tmp/eos_replay/2026-06-18_fix.json \
      --start-hour 4 --ngen 400 --seeds 1 2 3 4 5 6 7 8 9 10 \
      --cheap-slots 12 13 14 --report /tmp/eos_replay/arms_0618.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = "tests/replay/test_warmstart_probe.py"

BASE_GATES = {
    "EOS_OVERNIGHT_RESERVE": "1",
    "EOS_BATTERY_GRID_EXPORT": "1",
    "EOS_SELF_CONSUMPTION_PRIORITY": "1",
    "EOS_RESERVE_PRICE_AWARE": "1",  # Finding-2 is live in prod -> baseline ON
}

ARMS = {
    "A_off": {"EOS_PV_CHARGE_WINDOW": "0"},
    "B_partition": {"EOS_PV_CHARGE_WINDOW": "1", "EOS_PV_CHARGE_WINDOW_ELITE_K": "0"},
    "C_partition_elitism": {"EOS_PV_CHARGE_WINDOW": "1", "EOS_PV_CHARGE_WINDOW_ELITE_K": "2"},
}


def run(input_path, arm_env, ngen, start_hour, seed, tag):
    out = str(Path(input_path).with_suffix("")) + f".arm_{tag}.json"
    env = {
        **os.environ, **BASE_GATES, **arm_env,
        "EOS_PROBE_MODE": "ga",
        "EOS_PROBE_INPUT": input_path,
        "EOS_PROBE_OUT": out,
        "EOS_PROBE_NGEN": str(ngen),
        "EOS_PROBE_START_HOUR": str(start_hour),
        "EOS_PROBE_SEED": str(seed),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", HARNESS, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not Path(out).exists():
        raise RuntimeError(
            f"arm {tag} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-800:]}"
        )
    return json.loads(Path(out).read_text())


def trough_export(result, cheap_slots):
    """Σ simulated feed-in over the cheap (abs) slots; arrays are relative to the
    recorded start_slot, so index by (abs - start_slot)."""
    fe = result["feedin_wh"]
    ss = result["start_slot"]
    return sum(float(fe[h - ss]) for h in cheap_slots if 0 <= h - ss < len(fe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--start-hour", type=int, default=4)
    ap.add_argument("--ngen", type=int, default=400)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--cheap-slots", nargs="+", type=int, required=True)
    ap.add_argument("--store-threshold-wh", type=float, default=5000.0)
    ap.add_argument("--arms", nargs="+", default=["A_off", "C_partition_elitism"])
    ap.add_argument("--report", default="/tmp/eos_replay/arms.json")
    args = ap.parse_args()

    report = {"input": args.input, "start_hour": args.start_hour, "ngen": args.ngen,
              "cheap_slots": args.cheap_slots, "store_threshold_wh": args.store_threshold_wh,
              "arms": {}}
    for arm in args.arms:
        runs = []
        for seed in args.seeds:
            r = run(args.input, ARMS[arm], args.ngen, args.start_hour, seed, f"{arm}_s{seed}")
            te = trough_export(r, args.cheap_slots)
            stored = te < args.store_threshold_wh
            runs.append({"seed": seed, "bilanz": round(r["gesamtbilanz_eur"], 4),
                         "trough_export_wh": round(te), "trough_stored": stored,
                         "start_slot": r["start_slot"]})
            print(f"  {arm} seed={seed}: bilanz={r['gesamtbilanz_eur']:+.3f} "
                  f"trough_export={te:.0f}Wh stored={stored}", flush=True)
        n_stored = sum(1 for x in runs if x["trough_stored"])
        bilanz_mean = sum(x["bilanz"] for x in runs) / len(runs)
        report["arms"][arm] = {"runs": runs, "n_stored": n_stored, "n": len(runs),
                               "bilanz_mean": round(bilanz_mean, 4)}
        print(f"=== {arm}: trough stored {n_stored}/{len(runs)} | mean bilanz {bilanz_mean:+.4f}",
              flush=True)

    if "A_off" in report["arms"] and "C_partition_elitism" in report["arms"]:
        a = report["arms"]["A_off"]; c = report["arms"]["C_partition_elitism"]
        report["summary"] = {
            "armA_stored": f"{a['n_stored']}/{a['n']}",
            "armC_stored": f"{c['n_stored']}/{c['n']}",
            "fix_savings_mean_eur": round(a["bilanz_mean"] - c["bilanz_mean"], 4),
        }
        print(f"\nSUMMARY: A stored {a['n_stored']}/{a['n']}, C stored {c['n_stored']}/{c['n']}, "
              f"fix saves mean {a['bilanz_mean'] - c['bilanz_mean']:+.4f} EUR")
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
