"""EOS_RESERVE_EFF_ADJUST: eta-adjustierte Release-Schwelle (Christin 2026-07-21).

Break-even-Asymmetrie: 1 Akku-kWh verdraengt nachts nur import x eta_night an
Netzbezug (Teillast-WR), verdient aber spot x eta_sell im Bulk-Verkauf. Das Gate
skaliert avoided_import mit eta_night/eta_sell (~0.90) — die faire Schwelle
sinkt von import x 1.2 auf ~import x 1.08 (bei MARGIN 0.20).
"""
import importlib

import numpy as np
import pytest


def _reload_genetic(monkeypatch, adjust=None, curve=None):
    for k, v in (("EOS_RESERVE_EFF_ADJUST", adjust), ("EOS_INVERTER_EFF_CURVE", curve)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    # Reserve-Gates aktiv wie prod
    monkeypatch.setenv("EOS_RESERVE_PRICE_AWARE", "1")
    monkeypatch.setenv("EOS_RESERVE_RELEASE_MARGIN", "0.20")
    import akkudoktoreos.devices.genetic.inverter as inv_mod
    import akkudoktoreos.optimization.genetic.genetic as gen_mod
    importlib.reload(inv_mod)
    importlib.reload(gen_mod)
    return gen_mod


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    import os
    for k in ("EOS_RESERVE_EFF_ADJUST", "EOS_INVERTER_EFF_CURVE",
              "EOS_RESERVE_PRICE_AWARE", "EOS_RESERVE_RELEASE_MARGIN"):
        os.environ.pop(k, None)
    import akkudoktoreos.devices.genetic.inverter as inv_mod
    import akkudoktoreos.optimization.genetic.genetic as gen_mod
    importlib.reload(inv_mod)
    importlib.reload(gen_mod)


def _mk_arrays():
    """4 Slots: Slot 0 = Abend-Verkaufskandidat, danach Nachtlast ohne PV."""
    load = np.array([500.0, 1000.0, 1000.0, 1000.0])
    pv = np.array([0.0, 0.0, 0.0, 0.0])
    price = np.full(4, 0.000269)          # 26.9 ct/kWh Import
    revenue = np.array([0.000300, 0.0, 0.0, 0.0])  # Slot 0: 30 ct Verkaufserloes
    return load, pv, price, revenue


def test_gate_off_30ct_haelt_reserve(monkeypatch):
    """Ohne EFF_ADJUST: Schwelle 26.9x1.2=32.3ct -> 30ct released NICHT."""
    gen = _reload_genetic(monkeypatch, adjust=None)
    load, pv, price, revenue = _mk_arrays()
    res = gen._compute_overnight_reserve(load, pv, 0, 4, 1.0, price, revenue)
    full = gen._compute_overnight_reserve(load, pv, 0, 4, 1.0)  # ohne Preise = volle Reserve
    assert res[0] == pytest.approx(full[0]), "30 ct < 32.3 ct: Reserve bleibt voll"


def test_gate_eff_adjust_30ct_released(monkeypatch):
    """Mit EFF_ADJUST + ratio 0.9: Schwelle 26.9x0.9x1.2=29.1ct -> 30ct released."""
    gen = _reload_genetic(monkeypatch, adjust="1")
    load, pv, price, revenue = _mk_arrays()
    full = gen._compute_overnight_reserve(load, pv, 0, 4, 1.0)
    res = gen._compute_overnight_reserve(load, pv, 0, 4, 1.0, price, revenue,
                                          eff_ratio=0.9)
    assert res[0] < full[0], "30 ct > 29.1 ct: Reserve released (auf Safety-Floor)"


def test_ratio_none_ist_neutral(monkeypatch):
    """eff_ratio=None (kein ADJUST/keine Kurve) == exakt altes Verhalten."""
    gen = _reload_genetic(monkeypatch, adjust="1")
    load, pv, price, revenue = _mk_arrays()
    a = gen._compute_overnight_reserve(load, pv, 0, 4, 1.0, price, revenue)
    b = gen._compute_overnight_reserve(load, pv, 0, 4, 1.0, price, revenue,
                                        eff_ratio=None)
    assert np.array_equal(a, b)


def test_reserve_eff_ratio_aus_kurve(monkeypatch):
    """_reserve_eff_ratio: mit Kurve ratio<1; ohne Kurve 1.0; OFF -> None."""
    gen = _reload_genetic(monkeypatch, adjust="1",
                          curve="0.05:0.80,0.20:0.90,0.60:0.95,1.0:0.93")
    import akkudoktoreos.devices.genetic.inverter as inv_mod
    from unittest.mock import Mock, patch
    pred = Mock(); pred.calculate_self_consumption.return_value = 1.0
    with patch("akkudoktoreos.devices.genetic.inverter.get_eos_load_interpolator",
               return_value=pred):
        iv = inv_mod.Inverter(inv_mod.InverterParameters(
            device_id="iv1", max_power_wh=10000.0, battery_id=None))
    ratio = gen._reserve_eff_ratio(iv, 6000.0)   # sell bei 60 % Pnenn
    # eta_night = eta(0.06 x 10k = 600Wh -> frac 0.06) ~ 0.8067; eta_sell = eta(0.6) = 0.95
    assert ratio == pytest.approx(0.8067 / 0.95, abs=0.01)
    gen_off = _reload_genetic(monkeypatch, adjust=None, curve=None)
    assert gen_off._reserve_eff_ratio(iv, 6000.0) is None
