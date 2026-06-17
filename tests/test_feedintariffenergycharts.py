"""Tests for the DVhub fork FeedInTariffEnergyCharts provider.

The provider mirrors the ElecPriceEnergyCharts spot series into feed_in_tariff_wh
multiplied by an operator factor. When the operator surfaces real Bezugskosten on
the BUY side (elecprice.charges_kwh > 0, with VAT), the elec series carries those
charges + VAT, so the feed-in (which must stay pure SPOT) is reverse-engineered:

    spot_wh = price_wh / vat_rate - charges_wh      (charges > 0)
    spot_wh = price_wh                              (charges == 0, no-op)

This non-trivial unwind was previously untested (release-review 2026-06-14 #7).

These tests stub get_prediction() (so no prediction container is needed) and
record update_value() calls (so the per-slot data store is not exercised),
isolating exactly the unwind arithmetic, the factor, and the missing-provider path.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from akkudoktoreos.prediction.feedintariffenergycharts import FeedInTariffEnergyCharts
from akkudoktoreos.utils.datetimeutil import to_datetime


class _FakeElec:
    """Minimal stand-in for the ElecPriceEnergyCharts provider: iterable of
    per-slot records, with a no-op update_data()."""

    def __init__(self, records):
        self._records = records

    def update_data(self, force_update=False):
        pass

    def __iter__(self):
        return iter(self._records)


def _record(price_wh):
    return SimpleNamespace(
        date_time=to_datetime("2026-06-17 12:00:00"),
        elecprice_marketprice_wh=price_wh,
    )


@pytest.fixture
def provider(config_eos):
    settings = {
        "feedintariff": {
            "provider": "FeedInTariffEnergyCharts",
            "provider_settings": {
                "FeedInTariffEnergyCharts": {"spot_factor": 1.0},
            },
        }
    }
    config_eos.merge_settings_from_dict(settings)
    assert config_eos.feedintariff.provider == "FeedInTariffEnergyCharts"
    return FeedInTariffEnergyCharts()


def _patch_elec(monkeypatch, fake_elec):
    """Make the provider's late `get_prediction()` return a container whose
    provider_by_id('ElecPriceEnergyCharts') yields `fake_elec` (or None)."""
    fake_pred = Mock()
    fake_pred.provider_by_id = Mock(return_value=fake_elec)
    monkeypatch.setattr(
        "akkudoktoreos.core.coreabc.get_prediction", lambda: fake_pred
    )


def _capture_update_values(monkeypatch, provider):
    # The provider is a pydantic model, so per-instance attribute assignment is
    # blocked; patch update_value on the class (reverted by monkeypatch).
    captured: list[tuple] = []

    def _recorder(self, date_time, key, value):
        captured.append((key, value))

    monkeypatch.setattr(type(provider), "update_value", _recorder)
    return captured


def test_singleton_instance(provider):
    assert provider is FeedInTariffEnergyCharts()


def test_charges_zero_is_pure_spot_mirror(provider, config_eos, monkeypatch):
    """charges_kwh == 0 -> feed_in == spot * factor, no unwind."""
    config_eos.merge_settings_from_dict({"elecprice": {"charges_kwh": 0.0}})
    _patch_elec(monkeypatch, _FakeElec([_record(0.0006)]))
    captured = _capture_update_values(monkeypatch, provider)

    provider._update_data()

    assert captured == [("feed_in_tariff_wh", pytest.approx(0.0006))]


def test_charges_positive_unwinds_vat_and_charges(provider, config_eos, monkeypatch):
    """charges_kwh > 0 -> spot = price/vat - charges_wh (then * factor)."""
    config_eos.merge_settings_from_dict(
        {"elecprice": {"charges_kwh": 0.2, "vat_rate": 1.2}}
    )
    _patch_elec(monkeypatch, _FakeElec([_record(0.0006)]))
    captured = _capture_update_values(monkeypatch, provider)

    provider._update_data()

    # 0.0006 / 1.2 = 0.0005 ; charges_wh = 0.2/1000 = 0.0002 ; 0.0005 - 0.0002 = 0.0003
    assert captured == [("feed_in_tariff_wh", pytest.approx(0.0003))]


def test_spot_factor_is_applied(provider, config_eos, monkeypatch):
    """The operator spot_factor scales the mirrored value."""
    config_eos.merge_settings_from_dict(
        {
            "elecprice": {"charges_kwh": 0.0},
            "feedintariff": {
                "provider": "FeedInTariffEnergyCharts",
                "provider_settings": {
                    "FeedInTariffEnergyCharts": {"spot_factor": 0.95}
                },
            },
        }
    )
    _patch_elec(monkeypatch, _FakeElec([_record(0.0006)]))
    captured = _capture_update_values(monkeypatch, provider)

    provider._update_data()

    assert captured == [("feed_in_tariff_wh", pytest.approx(0.0006 * 0.95))]


def test_missing_elec_provider_returns_cleanly(provider, config_eos, monkeypatch):
    """No ElecPriceEnergyCharts provider registered -> early return, no writes."""
    config_eos.merge_settings_from_dict({"elecprice": {"charges_kwh": 0.0}})
    _patch_elec(monkeypatch, None)  # provider_by_id -> None
    captured = _capture_update_values(monkeypatch, provider)

    provider._update_data()

    assert captured == []
