"""T-EFF-CURVE (DVhub fork 2026-07-20): load-dependent inverter efficiency.

EOS_INVERTER_EFF_CURVE="frac:eta,..." replaces the constant dc_to_ac_efficiency
for real DC->AC conversions in process_energy(). Unset -> byte-identical legacy.
The curve is normalised over ac_wh/max_power_wh (P/Pnenn), so one measured
curve (e.g. MultiPlus 5000) scales to a bigger unit (MultiPlus 10000) verbatim.
"""
from unittest.mock import Mock, patch

import pytest


CURVE = "0.05:0.80,0.20:0.90,0.50:0.95,1.0:0.93"


def _reload_inverter(monkeypatch, curve):
    """Kurve fuer EINEN Test setzen.

    Frueher per importlib.reload(): seit EOS 0.4 steht InverterParameters (ein
    Pydantic-Modell) im selben Modul, und ein Reload erzeugt eine NEUE Klasse,
    waehrend andere Module die alte halten — nachfolgende Tests scheiterten an
    der Typpruefung. Deshalb nur die geparste Kurve tauschen; monkeypatch stellt
    sie nach dem Test wieder her. Geparst wird mit demselben Code wie beim
    Import, der Env-Pfad bleibt so abgedeckt.
    """
    import akkudoktoreos.devices.genetic.inverter as inv_mod
    monkeypatch.setattr(inv_mod, "_EFF_CURVE", inv_mod._parse_eff_curve(curve or ""))
    return inv_mod


def _make_inverter(inv_mod, max_power_wh=1000.0, battery=None):
    pred = Mock()
    pred.calculate_self_consumption.return_value = 1.0
    with patch(
        "akkudoktoreos.devices.genetic.inverter.get_eos_load_interpolator",
        return_value=pred,
    ):
        return inv_mod.Inverter(
            inv_mod.InverterParameters(
                device_id="iv1", max_power_wh=max_power_wh,
                battery_id=battery.parameters.device_id if battery else None,
            ),
            battery=battery,
        )


def test_curve_off_is_constant(monkeypatch):
    inv_mod = _reload_inverter(monkeypatch, None)
    iv = _make_inverter(inv_mod)
    assert iv._dc_ac_eff(10.0) == iv.dc_to_ac_efficiency
    assert iv._dc_ac_eff(900.0) == iv.dc_to_ac_efficiency


def test_curve_interpolation_and_clamps(monkeypatch):
    inv_mod = _reload_inverter(monkeypatch, CURVE)
    iv = _make_inverter(inv_mod, max_power_wh=1000.0)
    # Exact points
    assert iv._dc_ac_eff(50.0) == pytest.approx(0.80)    # frac 0.05
    assert iv._dc_ac_eff(200.0) == pytest.approx(0.90)   # frac 0.20
    assert iv._dc_ac_eff(1000.0) == pytest.approx(0.93)  # frac 1.0
    # Midpoint 0.05..0.20 -> frac 0.125 -> 0.85
    assert iv._dc_ac_eff(125.0) == pytest.approx(0.85)
    # Clamp below first point (fixed losses region held at first eta)
    assert iv._dc_ac_eff(10.0) == pytest.approx(0.80)
    # Clamp above last point (frac capped at 1.0)
    assert iv._dc_ac_eff(5000.0) == pytest.approx(0.93)


def test_curve_scales_with_device_size(monkeypatch):
    """Kern der 5000->10000-Frage: gleiche relative Last -> gleiches eta."""
    inv_mod = _reload_inverter(monkeypatch, CURVE)
    mp5000 = _make_inverter(inv_mod, max_power_wh=5000.0)
    mp10000 = _make_inverter(inv_mod, max_power_wh=10000.0)
    # 20% Auslastung: 1000 Wh auf dem 5000er == 2000 Wh auf dem 10000er
    assert mp5000._dc_ac_eff(1000.0) == pytest.approx(mp10000._dc_ac_eff(2000.0))
    assert mp5000._dc_ac_eff(1000.0) == pytest.approx(0.90)


def test_invalid_curve_falls_back_to_constant(monkeypatch):
    for bad in ("garbage", "0.5:1.5", "0.5", "0.2:0.9"):  # out-of-range / unparsable / <2 points
        inv_mod = _reload_inverter(monkeypatch, bad)
        iv = _make_inverter(inv_mod)
        assert iv._dc_ac_eff(500.0) == iv.dc_to_ac_efficiency, f"curve {bad!r} must be OFF"


def test_case2_partial_load_draws_more_dc(monkeypatch):
    """Teillast-Ehrlichkeit: 100 Wh Restlast bei eta 0.80 braucht 125 Wh DC.

    Case 2 (generation < consumption), Selbstverbrauchs-Deckung aus dem Akku.
    Mit Kurve muss der DC-Request um die Teillast-Effizienz vergroessert sein;
    die Wechselrichter-Verluste (DC-AC) tauchen in losses auf.
    """
    inv_mod = _reload_inverter(monkeypatch, CURVE)
    battery = Mock()
    battery.parameters.device_id = "battery1"
    battery.discharge_array = [0] * 48          # kein Export-Gen -> nur Selbstverbrauch
    battery.discharging_efficiency = 1.0
    battery.charge_energy = Mock(return_value=(0.0, 0.0))
    # Battery liefert exakt den angeforderten DC (kein Batterie-Verlust)
    battery.discharge_energy = Mock(side_effect=lambda dc, hour, **kw: (dc, 0.0))
    iv = _make_inverter(inv_mod, max_power_wh=1000.0, battery=battery)

    # generation 0, consumption 100 -> shortfall 100 Wh, frac 0.10.
    # Kurve 0.05:0.80 -> 0.20:0.90: eta(0.10) = 0.80 + (0.05/0.15)*0.10 = 0.8333
    grid_export, grid_import, losses, self_consumption = iv.process_energy(0.0, 100.0, 12)

    requested_dc = battery.discharge_energy.call_args[0][0]
    assert requested_dc == pytest.approx(100.0 / (0.80 + (0.05 / 0.15) * 0.10))
    assert grid_import == pytest.approx(0.0, abs=1e-9)
    assert self_consumption == pytest.approx(100.0)
    assert losses == pytest.approx(requested_dc - 100.0)  # reine WR-Verluste


def test_case2_off_matches_legacy(monkeypatch):
    """Gate OFF: identisches Verhalten wie vor dem Feature (Byte-Identitaet)."""
    inv_mod = _reload_inverter(monkeypatch, None)
    battery = Mock()
    battery.parameters.device_id = "battery1"
    battery.discharge_array = [0] * 48
    battery.discharging_efficiency = 1.0
    battery.charge_energy = Mock(return_value=(0.0, 0.0))
    battery.discharge_energy = Mock(side_effect=lambda dc, hour, **kw: (dc, 0.0))
    iv = _make_inverter(inv_mod, max_power_wh=1000.0, battery=battery)

    grid_export, grid_import, losses, self_consumption = iv.process_energy(0.0, 100.0, 12)
    requested_dc = battery.discharge_energy.call_args[0][0]
    # dc_to_ac_efficiency default aus InverterParameters (typisch 1.0 im Fork-Setup)
    assert requested_dc == pytest.approx(100.0 / iv.dc_to_ac_efficiency)
    assert grid_import == pytest.approx(0.0, abs=1e-9)


def test_ref_eff_night_operating_point(monkeypatch):
    """v2: Reserve-/Kapazitaets-Uebersetzungen nutzen eta am Nacht-Arbeitspunkt.

    NIGHT_FRAC default 0.06: auf 1000-Wh-Cap => 60 Wh => frac 0.06 auf Kurve
    0.05:0.80 -> 0.20:0.90: eta = 0.80 + (0.01/0.15)*0.10 = 0.8067.
    """
    inv_mod = _reload_inverter(monkeypatch, CURVE)
    iv = _make_inverter(inv_mod, max_power_wh=1000.0)
    assert iv._ref_eff() == pytest.approx(0.80 + (0.01 / 0.15) * 0.10)
    # Ohne Kurve: Konstante
    inv_mod2 = _reload_inverter(monkeypatch, None)
    iv2 = _make_inverter(inv_mod2)
    assert iv2._ref_eff() == iv2.dc_to_ac_efficiency


@pytest.mark.skip(
    reason="Braucht die Ueberhang-Freigabe (export_reserve_ac_wh in process_energy). "
    "Die ist beim Port auf Andreas' Zweig 2026-08-07 noch nicht uebernommen — "
    "Test bleibt stehen, damit er beim Nachziehen der Freigabe sofort greift."
)
def test_reserve_translation_holds_more_soc_with_curve(monkeypatch):
    """v2-Kern: gleiche AC-Reserve => MEHR zurueckgehaltene SoC-Wh bei eta<1.

    Case 2, Export erlaubt (Gen an), export_reserve_ac_wh=200. Mit Kurve wird
    die Reserve durch eta_night*disch_eff geteilt => groesserer SoC-Abzug =>
    weniger exportierbar als mit eta=1 (Legacy).
    """
    def run(curve):
        inv_mod = _reload_inverter(monkeypatch, curve)
        battery = Mock()
        battery.parameters.device_id = "battery1"
        battery.discharge_array = [1] * 48         # Export-Gen AN
        battery.discharging_efficiency = 1.0
        battery.soc_wh = 1000.0
        battery.min_soc_wh = 0.0
        battery.charge_energy = Mock(return_value=(0.0, 0.0))
        battery.discharge_energy = Mock(side_effect=lambda dc, hour, **kw: (dc, 0.0))
        iv = _make_inverter(inv_mod, max_power_wh=10000.0, battery=battery)
        # gen 0 < cons 100: shortfall 100, Rest der Akku-Energie oberhalb der
        # Reserve darf exportiert werden.
        grid_export, grid_import, losses, sc = iv.process_energy(
            0.0, 100.0, 12, export_reserve_ac_wh=200.0
        )
        return grid_export

    export_legacy = run(None)     # eta konstant (1.0) -> Reserve-Abzug exakt 200 Wh
    export_curve = run(CURVE)     # eta_night ~0.80 -> Reserve-Abzug 200/0.80 = 250 Wh
    assert export_curve < export_legacy, (
        f"Mit Kurve muss mehr SoC fuer die Nacht gehalten werden "
        f"(export {export_curve:.1f} !< {export_legacy:.1f})"
    )


# DVhub-Portierung auf v0.4.0rc1 (2026-09-22): 0.4 bewertet gespeicherte Energie
# an neuen Stellen (Endwert der Ladung, Tail-Kurve, Netzlade-Break-even). Mit
# der Konstanten allein (DVhub liefert 1.0) rechneten sie bei scharfer Kurve so,
# als gaebe es keinen Wechselrichterverlust. Sie nutzen jetzt das Referenz-η.
def test_reference_efficiency_follows_curve_at_night_point(monkeypatch):
    inv_mod = _reload_inverter(monkeypatch, CURVE)
    inv = _make_inverter(inv_mod, max_power_wh=1000.0)
    # Nacht-Arbeitspunkt 6 % → zwischen 0.05:0.80 und 0.20:0.90
    expected = 0.80 + (0.06 - 0.05) / (0.20 - 0.05) * (0.90 - 0.80)
    assert inv.reference_dc_to_ac_efficiency() == pytest.approx(expected)
    assert inv_mod.reference_dc_to_ac_efficiency(inv) == pytest.approx(expected)


def test_reference_efficiency_without_curve_is_constant(monkeypatch):
    inv_mod = _reload_inverter(monkeypatch, None)
    inv = _make_inverter(inv_mod, max_power_wh=1000.0)
    assert inv.reference_dc_to_ac_efficiency() == inv.dc_to_ac_efficiency


def test_reference_efficiency_accepts_plain_objects(monkeypatch):
    # Attrappen / Fremd-Implementierungen kennen nur die Konstante.
    from types import SimpleNamespace
    inv_mod = _reload_inverter(monkeypatch, CURVE)
    assert inv_mod.reference_dc_to_ac_efficiency(SimpleNamespace(dc_to_ac_efficiency=0.9)) == 0.9
