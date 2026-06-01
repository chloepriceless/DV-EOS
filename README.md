# DV-EOS

**A fork of [Akkudoktor-EOS](https://github.com/Akkudoktor-EOS/EOS) v0.3.0** with
operator-critical features for German **Spot-/Direktvermarktung** households that
run a large battery and want the optimizer to actually arbitrage it.

DV-EOS is the EOS engine behind **DVhub**. EOS is used here **display/advisory
only** — it computes the plan; DVhub + the Victron inverter execute it.

> Fork base: Akkudoktor-EOS v0.3.0 · License: Apache-2.0 (upstream, retained).
> Original copyright Dr. Andreas Schmitz et al. — see `LICENSE`.

## What this fork adds

| # | Feature | Why vanilla EOS can't | Activation |
|---|---------|----------------------|------------|
| 1 | **15-min optimization slots** | vanilla hard-clamps `optimization.interval` to 3600 s | config `optimization.interval ∈ {900,1800,3600}` |
| 2 | **`FeedInTariffEnergyCharts` provider** | only Fixed/Import feed-in providers exist | `feedintariff.provider` |
| 3 | **Battery→grid arbitrage discharge** | inverter `process_energy` has **no** battery→grid path; `grid_export` is PV-only | env `EOS_BATTERY_GRID_EXPORT` (default on) |
| 4 | **Forecast-aware overnight reserve** | nothing stops the optimizer selling the battery empty at the evening peak and buying the night back from the grid | env `EOS_OVERNIGHT_RESERVE`, `EOS_OVERNIGHT_RESERVE_MARGIN` |
| 5 | **Self-consumption priority** | for a fixed-tariff operator (import > any spot feed-in) covering load from the battery always beats importing — price-independent | env `EOS_SELF_CONSUMPTION_PRIORITY` (default on) |
| 6 | **Slot-aware battery/inverter math** | power caps weren't scaled to slot length → 4× too-fast charging at 15 min | automatic with (1) |
| 7 | **Pydantic `json.dumps` fix** in `/v1/prediction/import/{id}` | vanilla returns HTTP 400 when the body validates as a Pydantic model | automatic |

### The headline: battery → grid arbitrage (feature 3–5)

Vanilla EOS' genetic inverter (`devices/genetic/inverter.py` → `process_energy`)
has **no battery→grid discharge path**: `grid_export` is only ever PV surplus;
in Case 2 (load > PV) the battery may only cover the local shortfall. So the
`discharge` gene / `GRID_SUPPORT_EXPORT` label can express "sell from battery"
but the simulator never realises it — the battery can't be arbitrage-discharged
at the evening peak, and can't be emptied overnight to make room for next-day PV.

DV-EOS implements it cleanly in `process_energy` Case 2 (a single
`discharge_energy()` call: load first, surplus → `grid_export` up to inverter AC
headroom), with a forecast-derived **overnight reserve** (`genetic.py`
`_compute_overnight_reserve`) that keeps enough charge to self-consume the night,
and a **self-consumption priority** that covers load from the battery before any
grid import. See `eos-patches/UPSTREAM-battery-grid-export.md` (carried into this
repo) for the upstream-PR write-up.

## Activation today (env) → config tomorrow (upstream)

The feature gates are **environment variables** for now (zero-risk, reversible):

```bash
EOS_BATTERY_GRID_EXPORT=1          # battery→grid arbitrage discharge
EOS_OVERNIGHT_RESERVE=1            # keep forecast night load before exporting
EOS_OVERNIGHT_RESERVE_MARGIN=1.1   # safety margin on the reserve
EOS_SELF_CONSUMPTION_PRIORITY=1    # battery covers load before grid import
```

**Upstream-PR roadmap:** promote these env gates to proper EOS Pydantic config
settings (e.g. under `devices`/`optimization`) so they're activatable from the
EOS settings UI and mergeable into Akkudoktor-EOS. The env reads stay as a
fallback. Disable any gate (`=0`) + restart for vanilla behaviour — with all
gates off the math is byte-identical to upstream.

## Patch inventory

See `eos-patches/` in the DVhub repo (`apply.sh`) for the canonical, idempotent
patch set this fork was assembled from. Files touched:
`optimization/genetic/{genetic,geneticparams}.py`,
`devices/genetic/{inverter,battery,homeappliance}.py`,
`prediction/{feedintariff,prediction,feedintariffenergycharts}.py`,
`server/eos.py`.

## Status

Running in production (display-only) behind DVhub. Not yet a clean upstream PR —
the config-gating conversion (above) is the gating item before proposing to
Akkudoktor-EOS.
