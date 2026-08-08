#!/usr/bin/env python3
"""Treiber für den Replay-Harness (DVhub-Portierung 2026-08-08).

Die drei Tests unter ``tests/replay/`` verweisen auf dieses Skript, es war aber
nie eingecheckt — hier neu gebaut.

Warum ein eigener Prozess je Lauf: ``genetic.py`` liest seine Feature-Schalter
(``EOS_RESERVE_PRICE_AWARE`` und Geschwister) **beim Import** als Modul-Konstanten.
In einem Prozess umzuschalten ginge nur mit importlib-Turnerei, die still
danebengehen kann. Ein Prozess pro Schalterstellung ist die ehrliche Variante.

Aufruf:
    python scripts/replay/ablate.py --input tag.json [--out-dir ergebnisse/]
                                    [--start-hour 10] [--ngen 400] [--seed 42]
                                    [--gates "A=1,B=0" --gates "A=0"]

Ohne ``--gates`` läuft der kanonische A/B: Nacht-Reserve AUS gegen AN.

Ergebnis: eine Tabelle mit EUR-Bilanz, Netzbezug, Einspeisung und SoC-Minimum je
Schalterstellung — plus die Differenz zur ersten Zeile (der Basislinie).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Kanonischer A/B. Erste Zeile ist die Basislinie, gegen die verglichen wird.
DEFAULT_GATES = [
    ("Reserve AUS", {"EOS_OVERNIGHT_RESERVE": "0"}),
    ("Reserve AN", {"EOS_OVERNIGHT_RESERVE": "1"}),
]


def parse_gates(specs):
    out = []
    for spec in specs:
        env = {}
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue
            k, _, v = token.partition("=")
            env[k.strip()] = v.strip()
        out.append((spec, env))
    return out


def run_one(label, env_extra, args, out_dir):
    out_file = out_dir / f"{label.replace(' ', '_').replace('=', '-')}.json"
    env = dict(os.environ)
    env.update(
        {
            "EOS_REPLAY_INPUT": str(Path(args.input).resolve()),
            "EOS_REPLAY_OUT": str(out_file),
            "EOS_REPLAY_START_HOUR": str(args.start_hour),
            "EOS_REPLAY_NGEN": str(args.ngen),
            "EOS_REPLAY_INDIVIDUALS": str(args.individuals),
            "EOS_REPLAY_SEED": str(args.seed),
        }
    )
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/replay/test_reserve_replay.py", "-q", "--no-header"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if not out_file.exists():
        print(f"  ! {label}: kein Ergebnis geschrieben (rc={proc.returncode})")
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-4:]
        for line in tail:
            print(f"    {line}")
        return None
    return json.loads(out_file.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="optimize-input JSON eines Tages")
    ap.add_argument("--out-dir", default="/tmp/eos-replay")
    ap.add_argument("--start-hour", type=int, default=10)
    ap.add_argument("--ngen", type=int, default=400)
    ap.add_argument("--individuals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gates", action="append", default=[])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = parse_gates(args.gates) if args.gates else DEFAULT_GATES

    print(f"Eingabe: {args.input}")
    print(f"{'Schalterstellung':28}{'EUR':>10}{'Bezug Wh':>11}{'Einsp. Wh':>11}{'SoC min':>9}{'Δ EUR':>9}")
    print("-" * 78)

    base = None
    for label, env_extra in gates:
        res = run_one(label, env_extra, args, out_dir)
        if res is None:
            continue
        # Schluesselnamen wie sie tests/replay/test_reserve_replay.py schreibt.
        num = lambda xs: [x for x in (xs or []) if isinstance(x, (int, float))]
        eur = res.get("Gesamtbilanz_Euro")
        imp = sum(num(res.get("Netzbezug_Wh_pro_Stunde")))
        exp = sum(num(res.get("Netzeinspeisung_Wh_pro_Stunde")))
        soc_list = num(res.get("akku_soc_pro_stunde"))
        soc = min(soc_list) if soc_list else None
        if base is None:
            base = eur
        delta = (eur - base) if (eur is not None and base is not None) else None
        fmt = lambda v, w, p=2: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
        print(
            f"{label:28}{fmt(eur,10)}{fmt(imp,11,0)}{fmt(exp,11,0)}{fmt(soc,9,1)}"
            f"{fmt(delta,9) if delta is not None else '        -'}"
        )
    print("-" * 78)
    print("negativer EUR-Wert = Gewinn. Δ ist der Abstand zur ersten Zeile.")


if __name__ == "__main__":
    main()
