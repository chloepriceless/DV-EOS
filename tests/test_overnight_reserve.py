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
