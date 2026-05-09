import json
from pathlib import Path

import numpy as np

from drivetrain import NREL_5MW


REFERENCE_PATH = Path(__file__).parent / "data" / "nrel_5mw_matlab_gear_geometry.json"

ARRAY_FIELDS = [
    "d",
    "d_a",
    "d_b",
    "d_f",
    "d_m",
    "mass",
    "J_x",
    "J_y",
    "J_z",
]

SCALAR_FIELDS = [
    "u",
    "eps_alpha",
    "eps_beta",
    "eps_gamma",
    "cprime_th",
    "cprime",
    "c_gamma_alpha",
    "c_gamma_beta",
    "c_gamma",
    "k_mesh",
]


def test_nrel_5mw_gear_geometry_matches_matlab_reference():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    tolerances = reference["metadata"]["tolerances"]
    rtol = tolerances["relative"]
    atol = tolerances["absolute"]

    drivetrain = NREL_5MW()

    assert len(reference["stages"]) == drivetrain.N_st

    for expected, stage in zip(reference["stages"], drivetrain.stage):
        assert expected["configuration"] == stage.configuration
        assert expected["N_p"] == stage.N_p

        for field in ARRAY_FIELDS:
            np.testing.assert_allclose(
                getattr(stage, field),
                np.array(expected[field]),
                rtol=rtol,
                atol=atol,
                err_msg=f"Mismatch for stage {expected['stage']} field {field}",
            )

        for field in SCALAR_FIELDS:
            np.testing.assert_allclose(
                getattr(stage, field),
                expected[field],
                rtol=rtol,
                atol=atol,
                err_msg=f"Mismatch for stage {expected['stage']} field {field}",
            )
