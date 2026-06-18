"""Characterization tests for the DV-EOS overnight self-consumption reserve.

`_compute_overnight_reserve` (genetic.py) is a pure helper introduced by the DV
fork. It computes, per slot, the delivered-AC energy the battery must keep back
so the pack can ride from each evening to the next morning on self-consumption
instead of being sold empty at the evening peak and re-bought overnight. The
reserve is then translated into a per-slot SoC floor for the Case-2/Case-1 grid
EXPORT branch in inverter.process_energy (export_reserve_ac_wh).

The function and its gate (EOS_OVERNIGHT_RESERVE / _OVERNIGHT_RESERVE_ENABLED)
had zero test coverage. These tests pin the documented behaviour:

    reserve[h] = margin * Σ max(load[j] - pv[j], 0)

for j from h+1 up to (but excluding) the next slot where PV covers load, walked
backwards so each evening reserves exactly the night-ahead shortfall.
"""

import numpy as np
import pytest

import akkudoktoreos.optimization.genetic.genetic as genetic_mod
from akkudoktoreos.optimization.genetic.genetic import _compute_overnight_reserve


@pytest.fixture(autouse=True)
def _reserve_gate_on(monkeypatch):
    """Pin the gate ON regardless of the ambient EOS_OVERNIGHT_RESERVE env."""
    monkeypatch.setattr(genetic_mod, "_OVERNIGHT_RESERVE_ENABLED", True)


def test_gate_disabled_returns_all_zeros(monkeypatch):
    """With the gate off, the reserve is identically zero (vanilla fallback)."""
    monkeypatch.setattr(genetic_mod, "_OVERNIGHT_RESERVE_ENABLED", False)
    load = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    pv = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    reserve = _compute_overnight_reserve(load, pv, 0, 5, 1.0)
    assert reserve.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_reserve_accumulates_backwards_to_next_morning():
    """Each evening reserves the running night shortfall; a PV-covered slot
    (the next morning) resets the accumulator."""
    #          slot: 0     1     2     3      4
    load = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    pv = np.array([2000.0, 0.0, 0.0, 2000.0, 0.0])  # PV covers load at 0 and 3
    reserve = _compute_overnight_reserve(load, pv, 0, 5, 1.0)
    # h=4: next slot is the horizon end -> no reserve.
    # h=3: slot 4 uncovered (1000) -> 1000.
    # h=2: slot 3 PV-covered (morning) -> accumulator reset to 0.
    # h=1: slot 2 uncovered (1000) -> 1000.
    # h=0: slot 1 uncovered (+1000) on top of slot 2 -> 2000.
    assert reserve.tolist() == [2000.0, 1000.0, 0.0, 1000.0, 0.0]


def test_margin_scales_linearly():
    """The reserve is linear in margin: halving margin halves every slot."""
    load = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    pv = np.array([2000.0, 0.0, 0.0, 2000.0, 0.0])
    full = _compute_overnight_reserve(load, pv, 0, 5, 1.0)
    half = _compute_overnight_reserve(load, pv, 0, 5, 0.5)
    assert half.tolist() == [pytest.approx(v * 0.5) for v in full.tolist()]
    assert half.tolist() == [1000.0, 500.0, 0.0, 500.0, 0.0]


def test_pv_always_covers_load_no_reserve():
    """When PV covers load in every slot, nothing needs reserving."""
    load = np.array([1000.0, 1000.0, 1000.0])
    pv = np.array([2000.0, 2000.0, 2000.0])
    reserve = _compute_overnight_reserve(load, pv, 0, 3, 1.0)
    assert reserve.tolist() == [0.0, 0.0, 0.0]


def test_reserve_uses_shortfall_not_full_load():
    """Partial PV only reserves the uncovered remainder (load - pv), not load."""
    load = np.array([1000.0, 1000.0, 1000.0])
    pv = np.array([0.0, 300.0, 0.0])  # slot 1 is partially covered
    reserve = _compute_overnight_reserve(load, pv, 0, 3, 1.0)
    # h=2: horizon end -> 0.
    # h=1: slot 2 uncovered (1000) -> 1000.
    # h=0: slot 1 shortfall max(1000-300,0)=700 on top -> 1700.
    assert reserve.tolist() == [1700.0, 1000.0, 0.0]


def test_last_slot_has_no_reserve():
    """The final slot in the window never reserves (no slot beyond it)."""
    load = np.array([1000.0, 1000.0, 1000.0])
    pv = np.array([0.0, 0.0, 0.0])
    reserve = _compute_overnight_reserve(load, pv, 0, 3, 1.0)
    assert reserve[-1] == 0.0


def test_respects_start_and_end_window():
    """Slots outside [start_hour, end_hour) are left at zero."""
    load = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    pv = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    reserve = _compute_overnight_reserve(load, pv, 1, 4, 1.0)
    # Only indices 1..3 are written; 0 and 4 stay zero.
    assert reserve[0] == 0.0
    assert reserve[4] == 0.0
    # Within the window the same backward accumulation applies (end_hour=4):
    # h=3: nxt=4 == end_hour -> 0; h=2: slot 3 uncovered -> 1000; h=1: +1000 -> 2000.
    assert reserve[1] == 2000.0
    assert reserve[2] == 1000.0
    assert reserve[3] == 0.0


class TestPriceAwareReserveRelease:
    """Price-aware reserve release (EOS_RESERVE_PRICE_AWARE), the Finding-2 fix.

    When enabled, each slot's reserve is released down to a hard safety floor
    whenever the best still-reachable export price beats the highest avoided
    night-import price by more than the spread — i.e. selling into the evening
    peak and re-buying cheaper overnight beats holding the energy. The fix is
    tariff-agnostic: avoided_import is whatever per-slot import price is supplied
    (the DVhub bridge feeds the resolved fixed/dynamic/§14a price into it).
    """

    @pytest.fixture(autouse=True)
    def _price_aware_on(self, monkeypatch):
        # Gate ON + deterministic thresholds so the safety floor is visibly below
        # the energy-balance reserve in these scenarios.
        monkeypatch.setattr(genetic_mod, "_OVERNIGHT_RESERVE_ENABLED", True)
        monkeypatch.setattr(genetic_mod, "_RESERVE_PRICE_AWARE_ENABLED", True)
        monkeypatch.setattr(genetic_mod, "_RESERVE_RELEASE_SPREAD", 0.00005)  # 5 ct/kWh
        monkeypatch.setattr(genetic_mod, "_RESERVE_MIN_SAFETY_WH", 500.0)

    # Shared scenario: 3 slots, 1 kWh load each, no PV (pure night).
    LOAD = np.array([1000.0, 1000.0, 1000.0])
    PV = np.array([0.0, 0.0, 0.0])
    # Energy-balance reference (price-blind): [2000, 1000, 0].
    ENERGY_BALANCE = [2000.0, 1000.0, 0.0]

    def test_gate_off_with_price_args_equals_energy_balance(self, monkeypatch):
        """With the gate OFF the price/revenue args are ignored — byte-identical
        to the price-blind energy-balance reserve (the regression guard)."""
        monkeypatch.setattr(genetic_mod, "_RESERVE_PRICE_AWARE_ENABLED", False)
        price = np.array([0.00013, 0.00013, 0.00013])
        revenue = np.array([0.0005, 0.0001, 0.0001])
        reserve = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0, price, revenue)
        assert reserve.tolist() == self.ENERGY_BALANCE

    def test_release_when_peak_beats_avoided_import(self):
        """A 50 ct export peak reachable from slot 0 vs flat 10 ct night import:
        slot 0 releases to the safety floor (the peak is worth selling into)."""
        price = np.array([0.0001, 0.0001, 0.0001])    # 10 ct night import, flat
        revenue = np.array([0.0005, 0.0001, 0.0001])  # 50 ct peak at slot 0 only
        reserve = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0, price, revenue)
        # slot 0: peak 50 ct >> avoided 10 ct -> release to min(full=2000, safety=500)=500.
        # slot 1: best reachable export is now 10 ct == avoided -> not > spread -> hold full 1000.
        # slot 2: last slot, full=0.
        assert reserve.tolist() == [500.0, 1000.0, 0.0]

    def test_hold_when_no_peak_beats_avoided(self):
        """Flat price (export == import everywhere): nothing is worth selling,
        so the full energy-balance reserve is kept unchanged."""
        price = np.array([0.0001, 0.0001, 0.0001])
        revenue = np.array([0.0001, 0.0001, 0.0001])
        reserve = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0, price, revenue)
        assert reserve.tolist() == self.ENERGY_BALANCE

    def test_release_never_below_safety_and_never_above_full(self):
        """Release clamps to min(full_reserve, safety): never below the safety
        floor where full exceeds it, and never above the energy-balance amount."""
        price = np.array([0.0001, 0.0001, 0.0001])
        revenue = np.array([0.0009, 0.0009, 0.0009])  # huge peak everywhere -> release every slot
        reserve = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0, price, revenue)
        energy_balance = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0)
        for r, full in zip(reserve.tolist(), energy_balance.tolist()):
            assert r <= full + 1e-9                       # never reserve MORE than energy-balance
            assert r == pytest.approx(min(full, 500.0))   # released to the safety floor (or full if smaller)

    def test_tariff_agnostic_fixed_holds_where_spot_releases(self):
        """Same 20 ct peak, same code: a fixed 26.9 ct import tariff HOLDS the
        reserve (peak < tariff) while a 13 ct spot night RELEASES it (peak > spot).
        Demonstrates the fix carries no tariff logic — it reads whatever
        per-slot import price the bridge supplies."""
        revenue = np.array([0.0002, 0.0001, 0.0001])     # 20 ct peak at slot 0
        price_fixed = np.array([0.000269, 0.000269, 0.000269])  # 26.9 ct fixed tariff
        price_spot = np.array([0.00013, 0.00013, 0.00013])      # 13 ct spot night
        r_fixed = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0, price_fixed, revenue)
        r_spot = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0, price_spot, revenue)
        assert r_fixed.tolist() == self.ENERGY_BALANCE   # 20 ct < 26.9 ct -> hold full
        assert r_spot.tolist() == [500.0, 1000.0, 0.0]   # 20 ct > 13 ct + spread -> release slot 0

    def test_release_is_deterministic(self):
        """Pure function of the forecast arrays — identical inputs, identical output."""
        price = np.array([0.00013, 0.00013, 0.00013])
        revenue = np.array([0.0005, 0.0001, 0.0001])
        a = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0, price, revenue)
        b = _compute_overnight_reserve(self.LOAD, self.PV, 0, 3, 1.0, price, revenue)
        assert np.array_equal(a, b)

    def test_multi_day_horizon_scopes_release_to_its_own_night(self):
        """The release decision is scoped to a slot's OWN night window, not the
        whole 48h+ horizon. A later, pricier night must not suppress tonight's
        release (the windowing bug a naive max-to-end would hit, found by R22
        refute — every other test here is single-night and would miss it)."""
        # Two nights; PV covers load only at slot 2 (the morning between them).
        load = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        pv = np.array([0.0, 0.0, 2000.0, 0.0, 0.0])
        # Night 1 (slot 1) cheap re-buy 10 ct; night 2 (slots 3-4) expensive 40 ct.
        price = np.array([0.0001, 0.0001, 0.0001, 0.0004, 0.0004])
        # Evening-1 export peak 30 ct at slot 0.
        revenue = np.array([0.0003, 0.0001, 0.0001, 0.0001, 0.0001])
        reserve = _compute_overnight_reserve(load, pv, 0, 5, 1.0, price, revenue)
        # Slot 0: evening peak 30 ct > night-1 re-buy 10 ct -> RELEASE to the floor.
        # A max-to-horizon window would compare against night-2's 40 ct and wrongly hold.
        assert reserve[0] == 500.0
        # Slot 2: holds full for the EXPENSIVE night 2 (reachable export 10 ct <
        # re-buy 40 ct -> not worth selling before that night).
        assert reserve[2] == 2000.0
        assert reserve[3] == 1000.0
