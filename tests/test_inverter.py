from unittest.mock import Mock, patch

import pytest

from akkudoktoreos.devices.genetic.battery import Battery, SolarPanelBatteryParameters
from akkudoktoreos.devices.genetic.inverter import Inverter, InverterParameters


@pytest.fixture
def mock_battery() -> Mock:
    mock_battery = Mock()
    mock_battery.charge_energy = Mock(return_value=(0.0, 0.0))
    mock_battery.discharge_energy = Mock(return_value=(0.0, 0.0))
    mock_battery.parameters.device_id = "battery1"
    # DVhub fork: the inverter reads battery.discharge_array[hour] (the genetic
    # discharge gene) to gate battery→grid co-export. The upstream mock only
    # stubbed the charge/discharge methods. An all-zero gene array keeps the DV
    # co-export branches inert, so these upstream tests still assert the VANILLA
    # numeric behaviour (the regression net) — the unit-level analogue of
    # test_byte_identity. 48 entries cover hourly and 15-min slot indices.
    mock_battery.discharge_array = [0] * 48
    # The inverter re-grosses the delivered DC by the discharge efficiency to
    # track the per-slot discharge power budget; a real Battery always exposes
    # this. 1.0 keeps the existing numeric assertions unchanged.
    mock_battery.discharging_efficiency = 1.0
    return mock_battery


@pytest.fixture
def inverter(mock_battery) -> Inverter:
    mock_self_consumption_predictor = Mock()
    mock_self_consumption_predictor.calculate_self_consumption.return_value = 1.0
    with patch(
        "akkudoktoreos.devices.genetic.inverter.get_eos_load_interpolator",
        return_value=mock_self_consumption_predictor,
    ):
        iv = Inverter(
            InverterParameters(
                device_id="iv1", max_power_wh=500.0, battery_id=mock_battery.parameters.device_id
            ),
            battery = mock_battery
        )
        return iv


def test_process_energy_excess_generation(inverter, mock_battery):
    # Battery charges 100 Wh with 10 Wh loss
    mock_battery.charge_energy.return_value = (100.0, 10.0)
    generation = 600.0
    consumption = 200.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == pytest.approx(290.0, rel=1e-2)  # 290 Wh feed-in after battery charges
    assert grid_import == 0.0  # No grid draw
    assert losses == 10.0  # Battery charging losses
    assert self_consumption == 200.0  # All consumption is met
    mock_battery.charge_energy.assert_called_once_with(400.0, hour)
    mock_battery.discharge_energy.assert_not_called()
    inverter.self_consumption_predictor.calculate_self_consumption.assert_called_once_with(
        consumption, generation
    )


def test_process_energy_excess_generation_interpolator(inverter, mock_battery):
    # Battery charges 100 Wh with 10 Wh loss
    mock_battery.charge_energy.return_value = (100.0, 10.0)
    mock_battery.discharge_energy.return_value = (20.0, 2.0)
    inverter.self_consumption_predictor.calculate_self_consumption.return_value = 0.95

    generation = 600.0
    consumption = 200.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == pytest.approx(
        270.0, rel=1e-2
    )  # 290 Wh feed-in - 5% of generation-consumption self consumption after battery charges
    assert grid_import == pytest.approx(0.0, rel=1e-2)  # No grid draw
    assert losses == 12.0  # Battery charging losses
    assert self_consumption == 220.0  # All consumption is met
    mock_battery.charge_energy.assert_called_once_with(pytest.approx(380.0, rel=1e-2), hour)
    mock_battery.discharge_energy.assert_called_once_with(pytest.approx(20.0, rel=1e-2), hour)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_called_once_with(
        consumption, generation
    )


def test_process_energy_generation_equals_consumption(inverter, mock_battery):
    generation = 300.0
    consumption = 300.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in as generation equals consumption
    assert grid_import == 0.0  # No grid draw
    assert losses == 0.0  # No losses
    assert self_consumption == 300.0  # All consumption is met with generation

    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_not_called()
    inverter.self_consumption_predictor.calculate_self_consumption.assert_called_once_with(
        consumption, generation
    )


def test_process_energy_battery_discharges(inverter, mock_battery):
    # Battery discharges 100 Wh with 10 Wh loss already accounted for in the discharge
    mock_battery.discharge_energy.return_value = (100.0, 10.0)
    generation = 100.0
    consumption = 250.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in as generation is insufficient
    assert grid_import == pytest.approx(
        50.0, rel=1e-2
    )  # Grid supplies remaining shortfall after battery discharge
    assert losses == 10.0  # Discharge losses
    assert self_consumption == 200.0  # Generation + battery discharge
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_called_once_with(150.0, hour, ignore_gate=True)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_not_called()


def test_process_energy_battery_empty(inverter, mock_battery):
    # Battery is empty, so no energy can be discharged
    mock_battery.discharge_energy.return_value = (0.0, 0.0)
    generation = 100.0
    consumption = 300.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in as generation is insufficient
    assert grid_import == pytest.approx(200.0, rel=1e-2)  # Grid has to cover the full shortfall
    assert losses == 0.0  # No losses as the battery didn't discharge
    assert self_consumption == 100.0  # Only generation is consumed
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_called_once_with(200.0, hour, ignore_gate=True)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_not_called()


def test_process_energy_battery_full_at_start(inverter, mock_battery):
    # Battery is full, so no charging happens
    mock_battery.charge_energy.return_value = (0.0, 0.0)
    generation = 500.0
    consumption = 200.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == pytest.approx(
        300.0, rel=1e-2
    )  # All excess energy should be fed into the grid
    assert grid_import == 0.0  # No grid draw
    assert losses == 0.0  # No losses
    assert self_consumption == 200.0  # Only consumption is met
    mock_battery.charge_energy.assert_called_once_with(300.0, hour)
    mock_battery.discharge_energy.assert_not_called()
    inverter.self_consumption_predictor.calculate_self_consumption.assert_called_once_with(
        consumption, generation
    )


def test_process_energy_insufficient_generation_no_battery(inverter, mock_battery):
    # Insufficient generation and no battery discharge
    mock_battery.discharge_energy.return_value = (0.0, 0.0)
    generation = 100.0
    consumption = 500.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in as generation is insufficient
    assert grid_import == pytest.approx(400.0, rel=1e-2)  # Grid supplies the shortfall
    assert losses == 0.0  # No losses
    assert self_consumption == 100.0  # Only generation is consumed
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_called_once_with(400.0, hour, ignore_gate=True)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_not_called()


def test_process_energy_insufficient_generation_battery_assists(inverter, mock_battery):
    # Battery assists with some discharge to cover the shortfall
    mock_battery.discharge_energy.return_value = (
        50.0,
        5.0,
    )  # Battery discharges 50 Wh with 5 Wh loss
    generation = 200.0
    consumption = 400.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in as generation is insufficient
    assert grid_import == pytest.approx(
        150.0, rel=1e-2
    )  # Grid supplies the remaining shortfall after battery discharge
    assert losses == 5.0  # Discharge losses
    assert self_consumption == 250.0  # Generation + battery discharge
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_called_once_with(200.0, hour, ignore_gate=True)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_not_called()


def test_process_energy_zero_generation(inverter, mock_battery):
    # Zero generation, full reliance on battery and grid
    mock_battery.discharge_energy.return_value = (
        100.0,
        5.0,
    )  # Battery discharges 100 Wh with 5 Wh loss
    generation = 0.0
    consumption = 300.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in as there is zero generation
    assert grid_import == pytest.approx(200.0, rel=1e-2)  # Grid supplies the remaining shortfall
    assert losses == 5.0  # Discharge losses
    assert self_consumption == 100.0  # Only battery discharge is consumed
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_called_once_with(300.0, hour, ignore_gate=True)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_not_called()


def test_process_energy_zero_consumption(inverter, mock_battery):
    # Generation exceeds consumption, but consumption is zero
    mock_battery.charge_energy.return_value = (100.0, 10.0)
    generation = 500.0
    consumption = 0.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == pytest.approx(390.0, rel=1e-2)  # Excess energy after battery charges
    assert grid_import == 0.0  # No grid draw as no consumption
    assert losses == 10.0  # Charging losses
    assert self_consumption == 0.0  # Zero consumption
    mock_battery.charge_energy.assert_called_once_with(500.0, hour)
    mock_battery.discharge_energy.assert_not_called()
    inverter.self_consumption_predictor.calculate_self_consumption.assert_called_once_with(
        consumption, generation
    )


def test_process_energy_zero_generation_zero_consumption(inverter, mock_battery):
    generation = 0.0
    consumption = 0.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in
    assert grid_import == 0.0  # No grid draw
    assert losses == 0.0  # No losses
    assert self_consumption == 0.0  # No consumption
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_not_called()
    inverter.self_consumption_predictor.calculate_self_consumption.assert_called_once_with(
        consumption, generation
    )


def test_process_energy_partial_battery_discharge(inverter, mock_battery):
    mock_battery.discharge_energy.return_value = (50.0, 5.0)
    generation = 200.0
    consumption = 400.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in due to insufficient generation
    assert grid_import == pytest.approx(
        150.0, rel=1e-2
    )  # Grid supplies the shortfall after battery assist
    assert losses == 5.0  # Discharge losses
    assert self_consumption == 250.0  # Generation + battery discharge
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_called_once_with(200.0, 12, ignore_gate=True)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_not_called()


def test_process_energy_consumption_exceeds_max_no_battery(inverter, mock_battery):
    # Battery is empty, and consumption is much higher than the inverter's max power
    mock_battery.discharge_energy.return_value = (0.0, 0.0)
    generation = 100.0
    consumption = 1000.0  # Exceeds the inverter's max power
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in
    assert grid_import == pytest.approx(900.0, rel=1e-2)  # Grid covers the remaining shortfall
    assert losses == 0.0  # No losses as the battery didn’t assist
    assert self_consumption == 100.0  # Only the generation is consumed, maxing out the inverter
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_called_once_with(400.0, hour, ignore_gate=True)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_not_called()


def test_process_energy_zero_generation_full_battery_high_consumption(inverter, mock_battery):
    # Full battery, no generation, and high consumption
    mock_battery.discharge_energy.return_value = (500.0, 10.0)
    generation = 0.0
    consumption = 600.0
    hour = 12

    grid_export, grid_import, losses, self_consumption = inverter.process_energy(
        generation, consumption, hour
    )

    assert grid_export == 0.0  # No feed-in due to zero generation
    assert grid_import == pytest.approx(
        100.0, rel=1e-2
    )  # Grid covers remaining shortfall after battery discharge
    assert losses == 10.0  # Battery discharge losses
    assert self_consumption == 500.0  # Battery fully discharges to meet consumption
    mock_battery.charge_energy.assert_not_called()
    mock_battery.discharge_energy.assert_called_once_with(500.0, hour, ignore_gate=True)
    inverter.self_consumption_predictor.calculate_self_consumption.assert_not_called()


def test_case1_coexport_respects_battery_discharge_power_cap():
    """DVhub fork (release-review #3): a Case-1 slot that discharges the battery
    for BOTH the self-consumption residual load (process_energy `:125`) and
    battery->grid co-export (`:203`) must not exceed the battery's per-slot
    discharge power cap (max_charge_power_w x slot_duration_h). The cap is
    enforced independently inside each discharge_energy() call, so before the
    fix two calls could draw up to 2x the limit in one slot.

    Uses a REAL Battery (not a mock) so discharge_energy's internal power cap is
    actually exercised.
    """
    params = SolarPanelBatteryParameters(
        device_id="battery1",
        capacity_wh=10000,
        initial_soc_percentage=80,   # 8000 Wh available, SoC will NOT be the binding limit
        charging_efficiency=1.0,
        discharging_efficiency=1.0,  # clean arithmetic: raw == delivered
        min_soc_percentage=0,
        max_soc_percentage=100,
        max_charge_power_w=1000.0,   # => 1000 Wh per 1h slot discharge power cap
        hours=48,
    )
    battery = Battery(params, prediction_hours=48, slot_duration_h=1.0)
    battery.reset()
    hour = 12
    battery.discharge_array[hour] = 1  # discharge gene on -> co-export branch active

    # Record raw Wh withdrawn across ALL discharge_energy calls in this slot.
    real_discharge = battery.discharge_energy
    raw_withdrawals: list[float] = []

    def recording_discharge(wh, h, ignore_gate=False):
        delivered, losses = real_discharge(wh, h, ignore_gate=ignore_gate)
        raw_withdrawals.append(delivered / battery.discharging_efficiency)
        return delivered, losses

    battery.discharge_energy = recording_discharge

    # scr=0.7 -> with surplus 2000: residual-load discharge 600 Wh (`:125`,
    # under the 1000 cap) AND co-export wants the rest of the (huge) SoC. Before
    # the fix the co-export would draw a further full 1000 Wh -> 1600 total.
    mock_pred = Mock()
    mock_pred.calculate_self_consumption.return_value = 0.7
    with patch(
        "akkudoktoreos.devices.genetic.inverter.get_eos_load_interpolator",
        return_value=mock_pred,
    ):
        iv = Inverter(
            InverterParameters(
                device_id="iv1", max_power_wh=100000.0, battery_id="battery1"
            ),
            battery=battery,
            slot_duration_h=1.0,
        )
    iv.process_energy(generation=3000.0, consumption=1000.0, hour=hour)

    max_raw_slot_wh = battery.max_charge_power_w * battery.slot_duration_h  # 1000.0
    total_raw = sum(raw_withdrawals)
    # Non-vacuous: BOTH discharge paths must have fired (else the cap is trivial).
    assert len(raw_withdrawals) == 2, (
        f"expected both self-consumption and co-export discharges, got "
        f"{len(raw_withdrawals)} calls: {raw_withdrawals}"
    )
    assert total_raw <= max_raw_slot_wh + 1e-6, (
        f"battery discharged {total_raw} Wh raw in one slot, exceeding the "
        f"{max_raw_slot_wh} Wh per-slot power cap (release-review #3 regression)"
    )


# ---------------------------------------------------------------------------
# DVhub fork: Case-1 battery->grid CO-EXPORT feature characterization
# (gate EOS_BATTERY_GRID_EXPORT, inverter.py:166-219).
#
# The power-cap edge of this path is covered above
# (test_case1_coexport_respects_battery_discharge_power_cap); these tests
# characterize the FEATURE itself — that the battery actually co-exports in
# Case 1, the two gates (env + discharge gene), and the SoC / reserve / inverter
# headroom limits — which had no dedicated coverage (the documented coverage gap).
#
# Isolation trick: generation == consumption is the Case-1 boundary, so the PV
# surplus is exactly zero -> no battery charge (inverter.py:143) and no
# self-consumption discharge (inverter.py:120). The returned grid_export is then
# the pure battery co-export, with clean unit efficiencies (1.0) so the oracle is
# hand-computable. The per-slot power cap is set non-binding (20 kW) so it never
# masks the SoC/reserve/headroom limits under test.
# ---------------------------------------------------------------------------


def _coexport_setup(
    *,
    soc_pct: float,
    max_power_wh: float,
    min_soc_pct: float = 10.0,
    cap_wh: float = 10000.0,
    max_charge_power_w: float = 20000.0,
    disch_eff: float = 1.0,
    dc_to_ac: float = 1.0,
):
    """Build a REAL Battery + Inverter wired for the Case-1 co-export path.

    Efficiencies default to 1.0 (raw == delivered == AC) for clean arithmetic;
    pass disch_eff / dc_to_ac to characterize the conversion wiring.
    """
    params = SolarPanelBatteryParameters(
        device_id="battery1",
        capacity_wh=cap_wh,
        initial_soc_percentage=soc_pct,
        charging_efficiency=1.0,
        discharging_efficiency=disch_eff,
        min_soc_percentage=min_soc_pct,
        max_soc_percentage=100,
        max_charge_power_w=max_charge_power_w,  # per-slot discharge power cap (non-binding here)
        hours=48,
    )
    battery = Battery(params, prediction_hours=48, slot_duration_h=1.0)
    battery.reset()
    mock_pred = Mock()
    mock_pred.calculate_self_consumption.return_value = 1.0
    with patch(
        "akkudoktoreos.devices.genetic.inverter.get_eos_load_interpolator",
        return_value=mock_pred,
    ):
        iv = Inverter(
            InverterParameters(
                device_id="iv1",
                max_power_wh=max_power_wh,
                battery_id="battery1",
                dc_to_ac_efficiency=dc_to_ac,
            ),
            battery=battery,
            slot_duration_h=1.0,
        )
    return iv, battery


def test_case1_coexport_sells_battery_bounded_by_inverter_headroom():
    """Gate + discharge gene on: the battery co-exports in Case 1, capped by the
    inverter AC headroom (max_power_wh - load) when SoC is plentiful."""
    iv, battery = _coexport_setup(soc_pct=80, max_power_wh=5000.0)  # 8000 Wh, headroom 4000
    hour = 12
    battery.discharge_array[hour] = 1  # discharge gene on
    soc_before = battery.soc_wh

    grid_export, grid_import, losses, self_consumption = iv.process_energy(
        generation=1000.0, consumption=1000.0, hour=hour
    )

    # headroom = max_power_wh - load = 5000 - 1000 = 4000; SoC-unreserved = 7000 -> headroom binds
    assert grid_export == pytest.approx(4000.0)
    assert grid_import == pytest.approx(0.0)
    assert self_consumption == pytest.approx(1000.0)
    assert battery.soc_wh == pytest.approx(soc_before - 4000.0)


def test_case1_coexport_drains_only_to_min_soc():
    """With ample inverter headroom the co-export is bounded by the usable SoC
    above the hard floor and never drains the battery below min_soc."""
    iv, battery = _coexport_setup(soc_pct=80, max_power_wh=100000.0)  # 8000 Wh, headroom huge
    hour = 12
    battery.discharge_array[hour] = 1

    grid_export, _, _, _ = iv.process_energy(
        generation=1000.0, consumption=1000.0, hour=hour
    )

    # usable = soc - min_soc = 8000 - 1000 = 7000; headroom non-binding -> sells 7000
    assert grid_export == pytest.approx(7000.0)
    assert battery.soc_wh == pytest.approx(battery.min_soc_wh)  # exactly the floor, not below
    assert battery.soc_wh == pytest.approx(1000.0)


def test_case1_coexport_off_when_gate_disabled(monkeypatch):
    """Negative control: with EOS_BATTERY_GRID_EXPORT off the battery never
    co-exports in Case 1 (vanilla / Option-A fallback) — grid_export stays 0 and
    the SoC is untouched, even though the discharge gene is on."""
    monkeypatch.setattr(
        "akkudoktoreos.devices.genetic.inverter._BATTERY_GRID_EXPORT_ENABLED", False
    )
    iv, battery = _coexport_setup(soc_pct=80, max_power_wh=100000.0)
    hour = 12
    battery.discharge_array[hour] = 1  # gene on, but gate off -> still no co-export
    soc_before = battery.soc_wh

    grid_export, _, _, _ = iv.process_energy(
        generation=1000.0, consumption=1000.0, hour=hour
    )

    assert grid_export == pytest.approx(0.0)
    assert battery.soc_wh == pytest.approx(soc_before)  # battery untouched


def test_case1_coexport_off_when_discharge_gene_zero():
    """Second gate: even with EOS_BATTERY_GRID_EXPORT on, a slot whose discharge
    gene is 0 does not co-export (the GA controls WHEN to sell)."""
    iv, battery = _coexport_setup(soc_pct=80, max_power_wh=100000.0)
    hour = 12
    # discharge_array[hour] stays 0 from reset() -> gene off
    soc_before = battery.soc_wh

    grid_export, _, _, _ = iv.process_energy(
        generation=1000.0, consumption=1000.0, hour=hour
    )

    assert grid_export == pytest.approx(0.0)
    assert battery.soc_wh == pytest.approx(soc_before)


def test_case1_coexport_holds_overnight_reserve():
    """export_reserve_ac_wh is a SoC floor for the EXPORT branch: the co-export
    sells only the surplus ABOVE (min_soc + reserve), so the pack keeps the
    overnight self-consumption reserve instead of being sold empty at the peak."""
    iv, battery = _coexport_setup(soc_pct=80, max_power_wh=100000.0)  # 8000 Wh, headroom huge
    hour = 12
    battery.discharge_array[hour] = 1

    grid_export, _, _, _ = iv.process_energy(
        generation=1000.0, consumption=1000.0, hour=hour, export_reserve_ac_wh=3000.0
    )

    # usable above floor+reserve = 8000 - 1000 - 3000 = 4000 -> sells 4000
    assert grid_export == pytest.approx(4000.0)
    # ends at min_soc + reserve = 1000 + 3000 = 4000 Wh (reserve held)
    assert battery.soc_wh == pytest.approx(4000.0)


def test_case1_coexport_fully_suppressed_when_reserve_covers_usable_soc():
    """Boundary: a reserve >= the usable SoC above the hard floor suppresses
    co-export entirely — the battery holds everything for overnight self-
    consumption rather than selling any of it."""
    iv, battery = _coexport_setup(soc_pct=80, max_power_wh=100000.0)  # usable above floor = 7000
    hour = 12
    battery.discharge_array[hour] = 1
    soc_before = battery.soc_wh

    grid_export, _, _, _ = iv.process_energy(
        generation=1000.0, consumption=1000.0, hour=hour, export_reserve_ac_wh=8000.0
    )

    assert grid_export == pytest.approx(0.0)
    assert battery.soc_wh == pytest.approx(soc_before)  # nothing sold, full reserve held


def test_case1_coexport_applies_discharge_and_inverter_efficiencies():
    """Co-export converts pack DC to grid AC through BOTH the battery discharge
    efficiency AND the inverter DC->AC efficiency; the SoC drawdown is the raw
    (re-grossed) energy. The other co-export tests use unity efficiencies, so this
    is the one that pins down the conversion wiring (inverter.py:210/221/224)."""
    iv, battery = _coexport_setup(
        soc_pct=80, max_power_wh=100000.0, disch_eff=0.9, dc_to_ac=0.95
    )  # usable raw above floor = 7000, inverter headroom non-binding
    hour = 12
    battery.discharge_array[hour] = 1

    grid_export, _, _, _ = iv.process_energy(
        generation=1000.0, consumption=1000.0, hour=hour
    )

    # ac delivered = usable_raw * disch_eff * dc_to_ac = 7000 * 0.9 * 0.95 = 5985 (SoC-bound)
    assert grid_export == pytest.approx(5985.0)
    # the full usable raw (7000 Wh) leaves the pack -> ends exactly at the hard floor
    assert battery.soc_wh == pytest.approx(battery.min_soc_wh)
    assert battery.soc_wh == pytest.approx(1000.0)
