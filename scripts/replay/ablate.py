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


def run_one(label, env_extra, args, out_dir, seed):
    slug = f"{label.replace(' ', '_').replace('=', '-')}_seed{seed}"
    out_file = out_dir / f"{slug}.json"
    env = dict(os.environ)
    env.update(
        {
            "EOS_REPLAY_INPUT": str(Path(args.input).resolve()),
            "EOS_REPLAY_OUT": str(out_file),
            "EOS_REPLAY_START_HOUR": str(args.start_hour),
            "EOS_REPLAY_NGEN": str(args.ngen),
            "EOS_REPLAY_INDIVIDUALS": str(args.individuals),
            "EOS_REPLAY_SEED": str(seed),
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
    ap.add_argument(
        "--seeds",
        default="42",
        help="Komma-Liste fester GA-Seeds, z.B. 42,7,1234. Mehrere Seeds sind der "
        "einzige Weg, den Schalter-Effekt vom GA-Rauschen zu trennen — ein "
        "Einzellauf-Delta unterhalb der Seed-Streuung ist keine Aussage.",
    )
    ap.add_argument("--gates", action="append", default=[])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = parse_gates(args.gates) if args.gates else DEFAULT_GATES
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"Eingabe: {args.input}")
    print(f"Seeds:   {seeds}   Generationen: {args.ngen}   Individuen: {args.individuals}")
    print()
    print(
        f"{'Schalterstellung':22}{'EUR Mittel':>12}{'Spanne':>9}{'Bezug Wh':>11}"
        f"{'Einsp. Wh':>11}{'SoC min':>9}{'Δ EUR':>9}"
    )
    print("-" * 83)

    num = lambda xs: [x for x in (xs or []) if isinstance(x, (int, float))]
    fmt = lambda v, w, p=2: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")

    base_mean = None
    spreads = []
    for label, env_extra in gates:
        runs = [r for r in (run_one(label, env_extra, args, out_dir, s) for s in seeds) if r]
        if not runs:
            continue
        eurs = [r["Gesamtbilanz_Euro"] for r in runs]
        mean = sum(eurs) / len(eurs)
        spread = max(eurs) - min(eurs)
        spreads.append(spread)
        imp = sum(sum(num(r.get("Netzbezug_Wh_pro_Stunde"))) for r in runs) / len(runs)
        exp = sum(sum(num(r.get("Netzeinspeisung_Wh_pro_Stunde"))) for r in runs) / len(runs)
        socs = [min(num(r.get("akku_soc_pro_stunde")) or [None]) for r in runs]
        socs = [s for s in socs if s is not None]
        soc = min(socs) if socs else None
        if base_mean is None:
            base_mean = mean
        delta = mean - base_mean
        print(
            f"{label:22}{fmt(mean,12)}{fmt(spread,9)}{fmt(imp,11,0)}"
            f"{fmt(exp,11,0)}{fmt(soc,9,1)}{fmt(delta,9)}"
        )
    print("-" * 83)
    print("negativer EUR-Wert = Gewinn. Δ ist der Abstand zur ersten Zeile.")
    if len(seeds) > 1 and spreads:
        worst = max(spreads)
        print(
            f"Seed-Streuung innerhalb einer Schalterstellung: bis {worst:.2f} EUR. "
            "Ein Δ darunter ist Rauschen, keine Wirkung."
        )
    else:
        print("Nur ein Seed — das Δ kann reines GA-Rauschen sein. Mit --seeds 42,7,1234 pruefen.")


if __name__ == "__main__":
    main()
