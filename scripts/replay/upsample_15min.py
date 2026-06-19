#!/usr/bin/env python3
"""Upsample an hourly replay input (48 slots) to prod 15-min resolution (192 slots).

Prod EOS runs at ``interval=900`` → ``total_slots = prediction.hours × 4 = 192``.
The hourly replay inputs (built by ``build_inputs.py`` at interval=3600) converge
easily because the search space is small; the prod 192-slot problem (same
individuals=300 / generations=400) is ~4× larger and may UNDER-CONVERGE — the
leading hypothesis (Völtchen) for why prod exports the cheap midday trough while
the hourly single-shot stores it. This builds the 192-slot input to test that
single-shot at prod resolution.

Faithful upsample: energy arrays (gesamtlast, pv_prognose_wh) are split into 4
quarter-hour slots (value/4 each, so the hour's total energy is preserved); price
and feed-in arrays are per-Wh RATES → repeated as-is (no division). Battery /
inverter / EV params and initial SoC are unchanged. Run the optimizer with
``optimization.interval=900`` on the output.

NOTE: a faithful prod replay uses the REAL 15-min per-step forecast/SoC (from
prod ``optimizer_run_series``); this upsample only varies the SLOT COUNT to
isolate the convergence effect. Use the real per-step data for the rolling harness.

Usage:
  python scripts/replay/upsample_15min.py /tmp/eos_replay/2026-06-18_fix.json \
      /tmp/eos_replay/2026-06-18_fix_15min.json
"""
import json
import sys
from pathlib import Path


def upsample(src: str, dst: str) -> None:
    d = json.load(open(src))
    ems = d["ems"]

    def rep_energy(a):
        return [round(v / 4.0, 4) for v in a for _ in range(4)]

    def rep_rate(a):
        return [v for v in a for _ in range(4)]

    ems["gesamtlast"] = rep_energy(ems["gesamtlast"])
    ems["pv_prognose_wh"] = rep_energy(ems["pv_prognose_wh"])
    ems["strompreis_euro_pro_wh"] = rep_rate(ems["strompreis_euro_pro_wh"])
    fi = ems["einspeiseverguetung_euro_pro_wh"]
    ems["einspeiseverguetung_euro_pro_wh"] = rep_rate(fi) if isinstance(fi, list) else fi
    json.dump(d, open(dst, "w"), indent=1)
    n = len(ems["pv_prognose_wh"])
    print(f"{Path(dst).name}: {n} slots (pv/load split /4, price/feed-in rates repeated)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: upsample_15min.py <src_hourly.json> <dst_15min.json>", file=sys.stderr)
        sys.exit(2)
    upsample(sys.argv[1], sys.argv[2])
