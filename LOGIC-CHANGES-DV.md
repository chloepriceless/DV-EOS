# DV-EOS — concrete logic changes for Direktvermarktung compatibility

This document lists the **exact code-logic changes** (file · function · what · why)
that make EOS work for a German **Direktvermarktung / dynamic-spot** household
that arbitrages a battery. Written so the upstream maintainers can evaluate
merging them. Every change is **gated and defaults to behaviour-preserving**:
with the gates off / vanilla inputs, the simulation is byte-identical to
upstream v0.3.0.

Convention: "vanilla" = Akkudoktor-EOS v0.3.0.

---

## 1. Battery → grid export path (the core DV feature)
**File:** `src/akkudoktoreos/devices/genetic/inverter.py` · `Inverter.process_energy`
**Gate:** env `EOS_BATTERY_GRID_EXPORT` (default on)

- **Vanilla:** `grid_export` is *only ever* PV surplus. Case 1 (gen ≥ cons): PV
  surplus charges battery, remainder exported. Case 2 (gen < cons): the battery
  may **only** discharge to cover the local load shortfall — `grid_export` stays
  `0`. There is **no path** for stored battery energy to reach the grid, so the
  `discharge` gene / `GRID_SUPPORT_EXPORT` mode label can express "sell from
  battery" but the simulator never realises it.
- **DV-EOS:** in Case 2, after covering load, if the slot's discharge gene is set
  the inverter discharges **additional** battery energy into `grid_export`, up to
  the remaining inverter AC headroom (`max_power_wh − generation`) and the
  battery power/SoC limits. Implemented as a **single** `discharge_energy()` call
  (load first, surplus → export) so the per-slot battery power cap and inverter
  throughput never compound. Revenue then falls out of the existing
  `feedin_energy × elect_revenue_per_hour` term unchanged — the GA now optimises
  *when* to sell stored energy (evening spot peak) on its own.

## 2. Spot feed-in valuation + negative-price curtailment
**Files:** `prediction/feedintariffenergycharts.py` (new) · `prediction/feedintariff.py`,
`prediction/prediction.py` (registration) · `optimization/genetic/genetic.py` · `GeneticSimulation.simulate`

- **Vanilla:** feed-in providers are only `Fixed` (one static value) or `Import`
  (self-managed). A spot-marketed operator's feed-in revenue is `EPEX × factor`,
  which can't be expressed without a second data feed.
- **DV-EOS:** `FeedInTariffEnergyCharts` derives `feed_in_tariff_wh` from the
  `elecprice_marketprice_wh` series the elec provider already populated, × a
  `spot_factor` (no extra HTTP fetch). **And** `simulate()` curtails export when
  the feed-in price is negative: `if elect_revenue_per_hour[h] < 0: feedin = 0`
  (counted as a loss, never exported) — a DV operator receives **no Marktprämie**
  at negative prices, so exporting is a pure loss; the GA then stops planning it.

## 3. Overnight self-consumption reserve
**Files:** `optimization/genetic/genetic.py` · `_compute_overnight_reserve` →
`inverter.process_energy(export_reserve_ac_wh=…)`
**Gate:** env `EOS_OVERNIGHT_RESERVE`, `EOS_OVERNIGHT_RESERVE_MARGIN`

- **Problem the export path creates:** once the battery *can* sell to grid, a pure
  revenue optimiser empties it into the evening price peak and then buys the
  whole night's load back from the grid.
- **DV-EOS:** before the sim loop, `_compute_overnight_reserve` walks the load/PV
  arrays backward and computes, per slot, the forecast net load (`load − pv`)
  from that slot until PV next covers load (next morning), × a safety margin.
  The inverter's **export** branch may not drain the battery below
  `min_soc + reserve` (reserve converted to SoC via the discharge/inverter
  efficiencies). Self-consumption (covering load) may still use the reserve. Net:
  the battery sells only the genuine surplus at the peak and rides the night on
  self-consumption.

## 4. Self-consumption priority (price-independent)
**Files:** `devices/genetic/battery.py` · `discharge_energy(ignore_gate=…)` ·
`devices/genetic/inverter.py` · `process_energy`
**Gate:** env `EOS_SELF_CONSUMPTION_PRIORITY` (default on)

- **Insight:** for a **fixed import-tariff** operator the grid import price is
  always higher than any spot feed-in, so covering load from a charged battery
  is unconditionally cheaper than importing — at any price level. A residual-
  value threshold can't express this robustly (it breaks when spot leaves the
  assumed band).
- **DV-EOS:** `battery.discharge_energy(..., ignore_gate=True)` bypasses the
  per-slot `discharge_array` gate; the inverter uses it to cover the house load
  from the battery **regardless** of the genetic's discharge gene (down to the
  hard floor). The discharge gene is thus repurposed to govern only grid
  **export** timing. Eliminates "buys the night back from the grid while the
  battery is charged" without any price tuning.

## 5. 15-min slots + slot-aware power math (enabler, not DV-specific)
**Files:** `optimization/genetic/{genetic,geneticparams}.py`,
`devices/genetic/{battery,inverter,homeappliance}.py`

- **Vanilla:** hard-clamps `optimization.interval` to 3600 s; device power caps
  are treated as Wh-per-hour.
- **DV-EOS:** allows `{900, 1800, 3600}`; every power cap (`max_power_wh`,
  `max_charge_power_w`, discharge cap) is scaled by `slot_duration_h`, so at
  15-min resolution a slot can only move ¼ of the hourly energy. At
  `slot_duration_h = 1.0` this is byte-identical to vanilla. Needed because the
  evening-peak arbitrage decision is sub-hourly.

---

## Integration note (handled by the DVhub host, not in this repo)
The **fixed import price** for a fixed-tariff operator is supplied by pushing a
flat `elecprice_marketprice_wh` series (the gross Endkundenpreis) instead of
spot, while feed-in stays spot via `FeedInTariffImport`. EOS itself needs no
change for that — it's the elecprice/feedintariff *separation* (already vanilla)
that makes self-consumption-vs-export economics correct once the import series
carries the real price.

## Suggested upstream path
Promote the four env gates (sections 1, 3, 4 + the spot_factor of 2) to proper
EOS Pydantic config settings under `devices`/`optimization`, keeping env as a
fallback. Sections 2 and 5 are arguably useful to everyone, not just DV.
