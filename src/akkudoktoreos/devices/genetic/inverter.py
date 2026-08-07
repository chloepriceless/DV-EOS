import os
from typing import Optional

from loguru import logger

from akkudoktoreos.devices.genetic.battery import Battery
from akkudoktoreos.optimization.genetic.geneticdevices import InverterParameters
from akkudoktoreos.prediction.interpolator import get_eos_load_interpolator


# DVhub-Portierung 2026-08-07 (ursprünglich 2026-07-20/21, Christin) —
# lastabhängige Wechselrichter-Wirkungsgradkurve.
#
# Das Serienmodell rechnet JEDE Wandlung mit EINER konstanten
# dc_to_ac_efficiency: eine 200-W-Trickle-Entladung gilt als genauso effizient
# wie ein 4-kW-Block, obwohl ein realer MultiPlus bei kleiner Teillast deutlich
# schlechter ist (feste Verluste). Die Kurve ist NORMIERT über die
# Slot-AC-Auslastung (ac_wh / max_power_wh = P/Pnenn) angegeben, damit dieselbe
# Kurve über Gerätegrößen skaliert: Parallelschaltung erhält η(P/Pnenn) exakt.
#
# Format: EOS_INVERTER_EFF_CURVE="frac:eta,frac:eta,..."
#   z. B. "0.02:0.75,0.05:0.86,0.10:0.92,0.20:0.945,0.35:0.95,0.60:0.945,1.0:0.93"
# Nicht gesetzt/leer/unparsbar → AUS, byte-identisches Verhalten mit der
# Konstanten. Ist sie AN, ERSETZT die Kurve dc_to_ac_efficiency für die echten
# DC→AC-Wandlungen; sie multipliziert NICHT obendrauf (DVhub liefert
# dc_to_ac_efficiency=1.0, die Umlaufverluste stecken heute in den
# Batteriewirkungsgraden — siehe dvhub eos-config-sync.js buildEosInverters.
# Geht die Kurve scharf, muss optimizer.roundTripEfficiency auf den reinen
# Batteriewert gesenkt werden, sonst werden Verluste doppelt gezählt).
def _parse_eff_curve(raw: str):
    """'frac:eta,...' → sortierte [(frac, eta), ...] mit ≥2 Punkten, sonst None."""
    if not raw:
        return None
    points = []
    try:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            frac_s, eta_s = token.split(":")
            frac, eta = float(frac_s), float(eta_s)
            if not (0.0 <= frac <= 1.0) or not (0.0 < eta <= 1.0):
                logger.warning(f"EOS_INVERTER_EFF_CURVE: point out of range, curve OFF: {token}")
                return None
            points.append((frac, eta))
    except ValueError:
        logger.warning(f"EOS_INVERTER_EFF_CURVE: unparsable, curve OFF: {raw!r}")
        return None
    points.sort()
    if len(points) < 2:
        logger.warning("EOS_INVERTER_EFF_CURVE: needs >=2 points, curve OFF")
        return None
    return points


_EFF_CURVE = _parse_eff_curve(os.environ.get("EOS_INVERTER_EFF_CURVE", ""))

# v2 (Christin 2026-07-21): Reserve-/Kapazitäts-Übersetzungen aggregieren VIELE
# künftige Nacht-Slots — es gibt dort keine einzelne Wandlungsgröße, an der man
# η auswerten könnte. Sie nutzen deshalb η an einem definierten
# NACHT-ARBEITSPUNKT (NIGHT_FRAC × Pnenn, Vorgabe 6 % ≈ 1,4 kW auf 24 kW —
# typische nächtliche Hauslast). v1 behielt dort die Konstante und dimensionierte
# die Reserve zu klein, sobald die Kurve an war (Replay 16.07.: Nacht-Netzbezug
# 0,04 → 0,88 kWh). Mit η_night < 1 bindet dieselbe AC-Reserve MEHR SoC-Wh.
_EFF_CURVE_NIGHT_FRAC = 0.06
try:
    _EFF_CURVE_NIGHT_FRAC = float(os.environ.get("EOS_INVERTER_EFF_CURVE_NIGHT_FRAC", "0.06"))
except ValueError:
    logger.warning("EOS_INVERTER_EFF_CURVE_NIGHT_FRAC unparsable, using 0.06")


class Inverter:
    def __init__(
        self,
        parameters: InverterParameters,
        battery: Optional[Battery] = None,
        slot_duration_h: float = 1.0,
    ):
        self.parameters: InverterParameters = parameters
        self.battery: Optional[Battery] = battery
        self.slot_duration_h: float = slot_duration_h
        self._setup()

    def _setup(self) -> None:
        if self.battery and self.parameters.battery_id != self.battery.parameters.device_id:
            error_msg = f"Battery ID mismatch - {self.parameters.battery_id} is configured; got {self.battery.parameters.device_id}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.self_consumption_predictor = get_eos_load_interpolator()
        # max_power_wh is supplied as power [W] but used as the maximum energy
        # the inverter can move during one optimization slot.
        self.max_power_wh = self.parameters.max_power_wh * self.slot_duration_h
        self.dc_to_ac_efficiency = self.parameters.dc_to_ac_efficiency
        self.ac_to_dc_efficiency = self.parameters.ac_to_dc_efficiency
        # This value remains a power [W]. GeneticSimulation converts it into a
        # slot-independent charge-factor limit.
        self.max_ac_charge_power_w = self.parameters.max_ac_charge_power_w

    def _dc_ac_eff(self, ac_wh: float) -> float:
        """DC→AC-Wirkungsgrad für EINE Wandlung, die in diesem Slot ac_wh liefert.

        Mit gesetzter EOS_INVERTER_EFF_CURVE: lineare Interpolation von η über
        die Slot-AC-Auslastung frac = ac_wh / max_power_wh (∈[0,1], also
        P/Pnenn — max_power_wh ist bereits slot-skaliert, das Verhältnis damit
        dimensionslos). Außerhalb auf die Randpunkte geklemmt. Ohne Kurve: die
        Konstante (Legacy, byte-identisch).
        """
        if not _EFF_CURVE:
            return self.dc_to_ac_efficiency
        cap = self.max_power_wh
        frac = 0.0 if cap <= 0 else min(max(ac_wh / cap, 0.0), 1.0)
        points = _EFF_CURVE
        if frac <= points[0][0]:
            return points[0][1]
        if frac >= points[-1][0]:
            return points[-1][1]
        for i in range(1, len(points)):
            f1, e1 = points[i]
            if frac <= f1:
                f0, e0 = points[i - 1]
                t = (frac - f0) / (f1 - f0) if f1 > f0 else 0.0
                return e0 + t * (e1 - e0)
        return points[-1][1]

    def _ref_eff(self) -> float:
        """Referenz-η für Reserve-/Kapazitäts-Übersetzungen (v2).

        Ohne Kurve: die Konstante (byte-identisches Legacy-Verhalten).
        """
        if not _EFF_CURVE:
            return self.dc_to_ac_efficiency
        return self._dc_ac_eff(_EFF_CURVE_NIGHT_FRAC * self.max_power_wh)

    def _discharge_battery_to_ac(self, requested_ac_wh: float, hour: int) -> tuple[float, float]:
        """Discharge battery energy and convert it to AC energy."""
        if not self.battery or requested_ac_wh <= 0.0:
            return 0.0, 0.0

        # DVhub-Portierung: η aus der Kurve, ausgewertet an der ANGEFRAGTEN
        # AC-Größe. Ohne Kurve liefert _dc_ac_eff() die Konstante zurück, der
        # Pfad bleibt dann byte-identisch zum Original.
        eta = self._dc_ac_eff(requested_ac_wh)
        dc_request = requested_ac_wh / eta
        battery_discharge_dc, discharge_losses = self.battery.discharge_energy(dc_request, hour)
        battery_discharge_ac = battery_discharge_dc * eta
        inverter_discharge_losses = battery_discharge_dc - battery_discharge_ac
        return battery_discharge_ac, discharge_losses + inverter_discharge_losses

    def process_energy(
        self,
        generation: float,
        consumption: float,
        hour: int,
        allow_battery_grid_export: bool = False,
        export_reserve_ac_wh: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Process one slot using probabilistic direct PV-to-load overlap.

        ``generation`` and ``consumption`` are interval energies. The load
        probability table is evaluated in watts and yields the expected direct
        PV-to-load power. The remaining load and PV surplus are then handled
        independently, because both can occur during different sub-intervals of
        the same hourly or 15-minute slot.

        ``export_reserve_ac_wh`` (DVhub-Portierung 2026-08-07): der
        Eigenverbrauch (Last − PV), den der Akku von NACH diesem Slot bis zur
        nächsten PV-Deckung noch tragen muss — als gelieferte AC-Energie. Der
        Akku→Netz-EXPORT darf den Speicher nicht unter diese Reserve ziehen,
        sonst wird er am Abendpeak leer verkauft und die Nacht anschließend aus
        dem Netz zurückgekauft. Betrifft NUR den Export; die Deckung der lokalen
        Last bleibt unangetastet. 0.0 = aus, Pfad dann identisch zum Original.
        """
        losses = 0.0
        grid_export = 0.0
        generation = max(float(generation), 0.0)
        consumption = max(float(consumption), 0.0)

        # Convert interval energy [Wh] to mean power [W] for the probability
        # lookup, then convert its expected direct power back to slot energy.
        if generation > 0.0 and consumption > 0.0:
            expected_direct_power_w = (
                self.self_consumption_predictor.calculate_expected_direct_consumption(
                    consumption / self.slot_duration_h,
                    generation / self.slot_duration_h,
                )
            )
            direct_pv_energy = expected_direct_power_w * self.slot_duration_h
        else:
            direct_pv_energy = 0.0

        # Direct PV is bounded by both input energies and by the AC energy the
        # inverter can move during this slot.
        direct_pv_energy = min(
            max(direct_pv_energy, 0.0),
            generation,
            consumption,
            self.max_power_wh,
        )
        remaining_load = max(consumption - direct_pv_energy, 0.0)
        pv_surplus = max(generation - direct_pv_energy, 0.0)
        remaining_inverter_ac_capacity = max(self.max_power_wh - direct_pv_energy, 0.0)

        # Load gaps and PV surplus may both occur within the same coarse slot.
        # Cover the load gap first; this preserves the existing chronological
        # approximation and can create headroom for later PV charging.
        battery_discharge_ac = 0.0
        if remaining_load > 0.0 and self.battery and remaining_inverter_ac_capacity > 0.0:
            requested_ac_wh = min(remaining_load, remaining_inverter_ac_capacity)
            battery_discharge_ac, battery_discharge_losses = self._discharge_battery_to_ac(
                requested_ac_wh, hour
            )
            remaining_load = max(remaining_load - battery_discharge_ac, 0.0)
            remaining_inverter_ac_capacity = max(
                remaining_inverter_ac_capacity - battery_discharge_ac, 0.0
            )
            losses += battery_discharge_losses

        grid_import = remaining_load

        # Charge from the probabilistic PV surplus on the DC path. Stored energy
        # plus charge losses equals the PV energy accepted by the battery.
        remaining_surplus = pv_surplus
        if remaining_surplus > 0.0 and self.battery:
            charged_energy, charge_losses = self.battery.charge_energy(remaining_surplus, hour)
            remaining_surplus = max(remaining_surplus - charged_energy - charge_losses, 0.0)
            losses += charge_losses

        pv_grid_export = min(remaining_surplus, remaining_inverter_ac_capacity)
        grid_export += pv_grid_export
        remaining_inverter_ac_capacity = max(remaining_inverter_ac_capacity - pv_grid_export, 0.0)
        # PV which can neither charge the battery nor pass through the inverter
        # is curtailed and reported as a loss.
        losses += max(remaining_surplus - pv_grid_export, 0.0)

        if allow_battery_grid_export and self.battery and remaining_inverter_ac_capacity > 0.0:
            remaining_battery_ac = (
                # DVhub-Portierung: Kapazitäts-Vorschätzung, keine echte
                # Wandlung → Referenz-η am Nacht-Arbeitspunkt (v2).
                self.battery.remaining_discharge_energy_wh(hour) * self._ref_eff()
            )
            # DVhub-Portierung: die Nacht-Reserve wird VOM EXPORTIERBAREN
            # Anteil abgezogen, nicht vom Akku selbst. Was die lokale Last
            # deckt, ist oben bereits passiert und bleibt unberührt — die
            # Reserve verhindert nur den Verkauf des Nachtvorrats.
            exportable_battery_ac = max(
                remaining_battery_ac - max(float(export_reserve_ac_wh), 0.0), 0.0
            )
            export_capacity = min(remaining_inverter_ac_capacity, exportable_battery_ac)
            battery_export_ac, battery_export_losses = self._discharge_battery_to_ac(
                export_capacity, hour
            )
            grid_export += battery_export_ac
            losses += battery_export_losses

        self_consumption = direct_pv_energy + battery_discharge_ac
        return grid_export, grid_import, losses, self_consumption
