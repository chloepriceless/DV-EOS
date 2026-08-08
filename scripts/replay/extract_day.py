#!/usr/bin/env python3
"""Zieht einen echten Tag aus einer laufenden EOS-Instanz als Replay-Eingabe.

Hintergrund: DVhub schickt EOS keinen fertigen ``/optimize``-Payload mehr. Es
schiebt Prognosen per ``PUT /v1/prediction/import/...`` hinein, EOS baut sich die
``GeneticOptimizationParameters`` in ``geneticparams.prepare()`` selbst zusammen.
Diese Parameter werden nirgends persistiert — geprueft am 2026-08-08 auf .66:
weder unter ``/opt/dvhub`` noch ``/var/lib/dvhub`` liegt ein optimize-Payload,
und ``/v1/energy-management/optimization/solution`` haelt nur das *letzte*
Ergebnis im Speicher.

Dieses Skript baut die Parameter deshalb aus genau denselben Quellen nach, aus
denen ``prepare()`` sie baut — nur ueber die HTTP-API statt im Prozess:

  Prognosen   GET /v1/prediction/dataframe   (pv, preis, last, einspeisung, temp)
  Geraete     GET /v1/config                 (Akku, Wechselrichter, EV)
  Start-SoC   GET /v1/measurement/series     (battery1-soc-factor)

Alles davon ist **lesend**. Es geht kein Schreibzugriff an die Anlage.

Aufruf gegen Prod (EOS lauscht dort nur auf localhost, deshalb ueber ssh):

    python3 scripts/replay/extract_day.py --ssh root@192.168.20.66 \\
        --out /tmp/prod-heute.json

Oder gegen eine lokal erreichbare Instanz:

    python3 scripts/replay/extract_day.py --url http://127.0.0.1:8504 --out x.json

Das Ergebnis ist eine Datei, die ``scripts/replay/ablate.py --input`` frisst.
"""

import argparse
import json
import shlex
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

# Genau die Reihenfolge/Namen, die geneticparams.prepare() verwendet.
PREDICTION_KEYS = [
    "pvforecast_ac_power",
    "elecprice_marketprice_wh",
    "loadforecast_power_w",
    "feed_in_tariff_wh",
    "weather_temp_air",
]


class EosReader:
    """Liest lesend aus einer EOS-Instanz — direkt per HTTP oder ueber ssh+curl."""

    def __init__(self, url=None, ssh=None, port=8503, timeout=60):
        self.url = url.rstrip("/") if url else None
        self.ssh = ssh
        self.port = port
        self.timeout = timeout

    def get(self, path):
        if self.ssh:
            target = f"http://127.0.0.1:{self.port}{path}"
            cmd = ["ssh", "-o", "ConnectTimeout=10", self.ssh,
                   f"curl -s -m {self.timeout} {shlex.quote(target)}"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ssh/curl fehlgeschlagen: {proc.stderr.strip()}")
            body = proc.stdout
        else:
            with urllib.request.urlopen(self.url + path, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        body = body.strip()
        if not body.startswith(("{", "[")):
            raise RuntimeError(f"EOS antwortete nicht mit JSON auf {path}: {body[:200]}")
        return json.loads(body)


def fetch_frame(reader, start_dt, end_dt):
    """Holt die fuenf Prognose-Reihen als stundenscharfen Rahmen."""
    query = [("keys", k) for k in PREDICTION_KEYS]
    if start_dt:
        query.append(("start_datetime", start_dt))
    if end_dt:
        query.append(("end_datetime", end_dt))
    path = "/v1/prediction/dataframe?" + urllib.parse.urlencode(query)
    return reader.get(path)["data"]


def series_to_columns(data, hours):
    """Dreht {zeit: {key: wert}} in {key: [werte]} und schneidet auf ``hours``.

    Fehlende Stuenden werden nicht stillschweigend aufgefuellt — sie fliegen als
    Fehler hoch. Eine Luecke in der Mitte wuerde den Replay sonst um Stunden
    verschieben, ohne dass es jemand merkt.
    """
    stamps = sorted(data.keys())
    if len(stamps) < hours:
        raise RuntimeError(
            f"EOS lieferte nur {len(stamps)} Stunden, gebraucht werden {hours}. "
            "Prognose-Horizont in der EOS-Config zu kurz?"
        )
    stamps = stamps[:hours]
    cols = {k: [] for k in PREDICTION_KEYS}
    for stamp in stamps:
        row = data[stamp]
        for key in PREDICTION_KEYS:
            value = row.get(key)
            if value is None:
                raise RuntimeError(f"Luecke in '{key}' bei {stamp}")
            cols[key].append(float(value))
    return stamps, cols


def soc_at(reader, key, stamps, start_hour):
    """Start-SoC in Prozent (0..100, int) — wie prepare() ihn setzt.

    Nicht der SoC um Mitternacht, sondern der zur *Optimierungs-Startstunde*:
    prepare() liest ihn zu ``ems.start_datetime``, und das ist im Replay
    ``start_hour``. Die Reihen sind auf 00:00 verankert, der Startzustand nicht.
    """
    path = "/v1/measurement/series?" + urllib.parse.urlencode({"key": key})
    try:
        data = reader.get(path)["data"]
    except Exception as exc:  # kein SoC = kein sinnvoller Replay
        raise RuntimeError(f"Start-SoC ({key}) nicht lesbar: {exc}") from exc
    if not data:
        raise RuntimeError(f"Start-SoC ({key}) ist leer")
    # Der letzte Messwert vor (oder auf) der Startstunde.
    start = stamps[min(start_hour, len(stamps) - 1)]
    before = [t for t in sorted(data.keys()) if t <= start]
    stamp = before[-1] if before else sorted(data.keys())[0]
    factor = float(data[stamp])
    if not 0.0 <= factor <= 1.0:
        raise RuntimeError(f"Unplausibler SoC-Faktor {factor} bei {stamp}")
    return int(factor * 100), stamp


def check_plausible(cols, stamps, allow_empty=False):
    """Faengt Attrappen ab, bevor sie als echter Tag durchgehen.

    EOS haelt die Prognose-Reihen nur wenige Tage vor (auf .66 am 2026-08-08
    gemessen: 08-06 echt, 08-04 leer). Weiter zurueck liefert die API **keinen
    Fehler**, sondern eine glatt aufgefuellte Reihe — PV 0,0 kWh ueber 48
    Sommerstunden, Einspeiseverguetung als Konstante. Ein Replay darauf sieht
    tadellos aus und sagt nichts. Deshalb hier die Reissleine.

    Rueckgabe: Liste von Warnungen (leer = unauffaellig).
    """
    warnings = []
    pv_sum = sum(cols["pvforecast_ac_power"])
    if pv_sum <= 0.0 and not allow_empty:
        raise RuntimeError(
            f"PV-Summe ist {pv_sum:.1f} Wh ueber {len(stamps)} Stunden — das ist keine "
            f"echte Prognose, sondern eine aufgefuellte Leerreihe. EOS haelt nur wenige "
            f"Tage vor; {stamps[0]} liegt vermutlich dahinter. Mit --allow-empty "
            "trotzdem schreibbar."
        )
    for key, values in cols.items():
        if len(set(values)) == 1:
            warnings.append(f"'{key}' ist ueber alle {len(values)} Stunden konstant ({values[0]})")
    return warnings


def build_params(cfg, cols, initial_soc, terminal_value_kwh):
    batteries = (cfg.get("devices") or {}).get("batteries") or []
    inverters = (cfg.get("devices") or {}).get("inverters") or []
    if not batteries:
        raise RuntimeError("Keine Batterie in der EOS-Config — Replay waere sinnlos")

    bat = batteries[0]
    pv_akku = {
        "device_id": bat["device_id"],
        "capacity_wh": bat["capacity_wh"],
        "charging_efficiency": bat.get("charging_efficiency"),
        "discharging_efficiency": bat.get("discharging_efficiency"),
        "levelized_cost_of_storage_kwh": bat.get("levelized_cost_of_storage_kwh"),
        "max_charge_power_w": bat.get("max_charge_power_w"),
        "min_soc_percentage": bat.get("min_soc_percentage"),
        "max_soc_percentage": bat.get("max_soc_percentage"),
        "charge_rates": bat.get("charge_rates"),
        "initial_soc_percentage": initial_soc,
    }
    pv_akku = {k: v for k, v in pv_akku.items() if v is not None}

    inverter = None
    if inverters:
        inv = inverters[0]
        inverter = {
            "device_id": inv["device_id"],
            # prepare() mappt max_power_w -> max_power_wh. Nicht schoen, aber so.
            "max_power_wh": inv.get("max_power_w"),
            "battery_id": inv.get("battery_id"),
            "ac_to_dc_efficiency": inv.get("ac_to_dc_efficiency"),
            "dc_to_ac_efficiency": inv.get("dc_to_ac_efficiency"),
            "max_ac_charge_power_w": inv.get("max_ac_charge_power_w"),
        }
        inverter = {k: v for k, v in inverter.items() if v is not None}

    return {
        "ems": {
            "pv_prognose_wh": cols["pvforecast_ac_power"],
            "strompreis_euro_pro_wh": cols["elecprice_marketprice_wh"],
            "einspeiseverguetung_euro_pro_wh": cols["feed_in_tariff_wh"],
            "gesamtlast": cols["loadforecast_power_w"],
            "preis_euro_pro_wh_akku": terminal_value_kwh / 1000.0,
        },
        "temperature_forecast": cols["weather_temp_air"],
        "pv_akku": pv_akku,
        "inverter": inverter,
        "eauto": None,
        "home_appliances": None,
    }


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Basis-URL einer erreichbaren EOS-Instanz")
    src.add_argument("--ssh", help="ssh-Ziel, z.B. root@192.168.20.66 (EOS auf localhost)")
    ap.add_argument("--port", type=int, default=8503, help="EOS-Port auf dem ssh-Ziel")
    ap.add_argument("--out", required=True, help="Zieldatei fuer den Replay-Input")
    ap.add_argument("--hours", type=int, default=48, help="Stunden Horizont (default 48)")
    ap.add_argument(
        "--start",
        help="Startzeitpunkt ISO. Muss Mitternacht sein: geneticparams.prepare() "
        "verankert die Reihen auf 00:00 des Starttages (start_datetime.set(hour=0)) "
        "und ``start_hour`` indiziert von dort aus. Default ist Mitternacht heute.",
    )
    ap.add_argument(
        "--start-hour",
        type=int,
        default=10,
        help="Optimierungs-Startstunde. Muss zu ablate.py --start-hour passen; "
        "sie bestimmt, zu welchem Zeitpunkt der Start-SoC gelesen wird.",
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="Leere/aufgefuellte Prognose-Reihen trotzdem schreiben (normalerweise Abbruch).",
    )
    ap.add_argument(
        "--allow-offgrid-start",
        action="store_true",
        help="Startzeitpunkt abseits von 00:00 zulassen (verschiebt den Replay um Stunden).",
    )
    ap.add_argument(
        "--terminal-value-kwh",
        type=float,
        default=0.0,
        help="EUR/kWh Restwert der Akkuladung am Horizont-Ende. Aeltere EOS-Staende "
        "kennen optimization.terminal_value_euro_per_kwh nicht; dann greift dieser Wert.",
    )
    args = ap.parse_args()

    reader = EosReader(url=args.url, ssh=args.ssh, port=args.port)

    # Verankerung auf Mitternacht ist keine Kosmetik: prepare() setzt
    # parameter_start_datetime = start_datetime.set(hour=0), und ``start_hour``
    # im Replay indiziert genau in diese Reihen hinein. Ein Rahmen ab 02:00
    # wuerde jede Stunde um zwei verrutschen, ohne dass etwas auffaellt.
    if args.start:
        start_dt = datetime.fromisoformat(args.start)
    else:
        start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if (start_dt.hour, start_dt.minute) != (0, 0) and not args.allow_offgrid_start:
        ap.error(
            f"--start {start_dt.isoformat()} ist nicht Mitternacht. Die Reihen muessen "
            "auf 00:00 verankert sein, sonst verschiebt sich der Replay. "
            "Mit --allow-offgrid-start trotzdem erzwingbar."
        )
    start = start_dt.isoformat()
    end = (start_dt + timedelta(hours=args.hours)).isoformat()

    print(f"Lese Prognosen ({args.hours} h ab {start}) ...", file=sys.stderr)
    frame = fetch_frame(reader, start, end)
    stamps, cols = series_to_columns(frame, args.hours)
    warnings = check_plausible(cols, stamps, allow_empty=args.allow_empty)

    print("Lese Geraete-Config ...", file=sys.stderr)
    cfg = reader.get("/v1/config")

    soc_key = ((cfg.get("devices") or {}).get("batteries") or [{}])[0].get(
        "measurement_key_soc_factor", "battery1-soc-factor"
    )
    print(f"Lese Start-SoC ({soc_key}) zur Stunde {args.start_hour} ...", file=sys.stderr)
    initial_soc, soc_stamp = soc_at(reader, soc_key, stamps, args.start_hour)

    terminal = (cfg.get("optimization") or {}).get("terminal_value_euro_per_kwh")
    if terminal is None:
        terminal = args.terminal_value_kwh
        note = f"(nicht in der Quell-Config, genommen: {terminal} EUR/kWh)"
    else:
        note = "(aus der Quell-Config)"

    params = build_params(cfg, cols, initial_soc, float(terminal))
    with open(args.out, "w", encoding="utf-8") as f_out:
        json.dump(params, f_out, indent=2)

    price = cols["elecprice_marketprice_wh"]
    feed = cols["feed_in_tariff_wh"]
    print(f"\nGeschrieben: {args.out}", file=sys.stderr)
    print(f"  Zeitraum      {stamps[0]}  bis  {stamps[-1]}", file=sys.stderr)
    print(f"  Start-SoC     {initial_soc} %  (Messwert von {soc_stamp})", file=sys.stderr)
    print(f"  Akku          {params['pv_akku']['capacity_wh']} Wh", file=sys.stderr)
    print(
        f"  Bezugspreis   {min(price)*100000:.1f} .. {max(price)*100000:.1f} ct/kWh",
        file=sys.stderr,
    )
    print(
        f"  Einspeisung   {min(feed)*100000:.1f} .. {max(feed)*100000:.1f} ct/kWh",
        file=sys.stderr,
    )
    print(f"  PV-Summe      {sum(cols['pvforecast_ac_power'])/1000:.1f} kWh", file=sys.stderr)
    print(f"  Last-Summe    {sum(cols['loadforecast_power_w'])/1000:.1f} kWh", file=sys.stderr)
    print(f"  Restwert Akku {terminal} EUR/kWh {note}", file=sys.stderr)
    for warning in warnings:
        print(f"  ! Verdaechtig: {warning}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:  # erwartbare Abbrueche ohne Traceback melden
        print(f"\nAbbruch: {exc}", file=sys.stderr)
        sys.exit(2)
