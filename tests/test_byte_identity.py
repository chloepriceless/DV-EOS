"""Guard: with all DV-EOS feature gates OFF, the genetic simulation must
reproduce upstream Akkudoktor-EOS v0.3.0 exactly (byte-identical).

The DV-EOS fork adds battery->grid arbitrage behind env gates that default ON.
The fork keeps a documented "revert to vanilla" capability: setting the gates
to 0 must restore upstream behaviour. That property is operationally unused
(production always runs with the gates ON, the changes are our default) but is
maintained so the changes stay proposable upstream and so a clean fall-back to
the original EOS behaviour exists for the case upstream EOS evolves.

It silently regressed once (vanilla gene-based load coverage was folded into the
export gate, so with all gates off the battery stopped covering the house load).
Because we never exercise the gates-off path in production, only a guard like
this surfaces such a regression. The gates are read at module import time, so
this runs the canonical vanilla-balance tests in a subprocess with the gates
forced off.
"""
import os
import subprocess
import sys

# Upstream v0.3.0 reference tests that assert exact Euro balances / per-slot
# grid values. With the DV gates off these must pass unchanged.
_VANILLA_BALANCE_TESTS = [
    "tests/test_geneticsimulation.py::test_simulation",
    "tests/test_geneticsimulation2.py::test_simulation",
]

_GATES_OFF = {
    "EOS_BATTERY_GRID_EXPORT": "0",
    "EOS_OVERNIGHT_RESERVE": "0",
    "EOS_SELF_CONSUMPTION_PRIORITY": "0",
}


def test_all_gates_off_is_byte_identical_to_vanilla():
    """All DV gates off => upstream v0.3.0 balances reproduce exactly."""
    env = {**os.environ, **_GATES_OFF}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *_VANILLA_BALANCE_TESTS, "-q", "--no-header"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "With all DV-EOS gates off the simulation must stay byte-identical to "
        "upstream Akkudoktor-EOS v0.3.0, but the vanilla-balance tests failed.\n"
        "A DV change has leaked into the gates-off path (the revert-to-vanilla "
        "capability is broken).\n"
        f"--- pytest stdout ---\n{result.stdout}\n--- pytest stderr ---\n{result.stderr}"
    )
