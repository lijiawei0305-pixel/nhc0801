"""Parent-Level Protocol P01 frozen constants for NHC0801.

Matches HANDOFF_PHASE9B_V004 §4.4. Do NOT fall back to production
B3LYP-D3(BJ)/def2-SVP (two_endpoint.LOCKED_PROTOCOL).
"""

from __future__ import annotations

from typing import Final

PROTOCOL_ID: Final = "Parent-Level Protocol P01"
PROTOCOL_SHA256: Final = (
    "227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a"
)

FUNCTIONAL: Final = "wb97m-d3bj"
BASIS: Final = "def2-TZVPP"
GRID_LEVEL: Final = 4
SCF_CONV_TOL: Final = 1.0e-9
VV10: Final = False
DISPERSION: Final = "explicit_two_body_d3bj"
ATM: Final = False
PHASE: Final = "gas"
SPIN_RESTRICTED: Final = True  # closed-shell RKS

# Label (electronic energy only; not Gibbs)
HARTREE_TO_KCAL: Final = 627.509474
PROTON_CONSTANT_KCAL: Final = 6.28
LOWER_IS_BETTER: Final = True

# Endpoint definitions
CATION_CHARGE: Final = 1
CATION_MULTIPLICITY: Final = 1
NEUTRAL_CHARGE: Final = 0
NEUTRAL_MULTIPLICITY: Final = 1


def deprotonation_electronic_kcal(e_neutral_hartree: float, e_cation_hartree: float) -> float:
    electronic = (e_neutral_hartree - e_cation_hartree) * HARTREE_TO_KCAL
    return electronic - PROTON_CONSTANT_KCAL
