import numpy as np

from Drivetrain import NREL_5MW


def test_nrel_5mw_stage_layout():
    model = NREL_5MW()

    assert model.N_st == 3
    assert [stage.configuration for stage in model.stage] == [
        "planetary",
        "planetary",
        "parallel",
    ]
    assert [stage.N_p for stage in model.stage] == [3, 3, 1]


def test_nrel_5mw_reference_geometry():
    model = NREL_5MW()

    assert [stage.m_n for stage in model.stage] == [45.0, 21.0, 14.0]
    assert [stage.a_w for stage in model.stage] == [863.0, 584.0, 861.0]
    assert [stage.z.tolist() for stage in model.stage] == [
        [19, 17, -56],
        [18, 36, -93],
        [24, 95],
    ]


def test_nrel_5mw_ratios_speeds_and_torques():
    model = NREL_5MW()

    np.testing.assert_allclose(
        [stage.u for stage in model.stage],
        [3.9473684210526314, 6.166666666666667, 3.9583333333333335],
    )
    np.testing.assert_allclose(
        model.u,
        [3.9473684210526314, 24.342105263157894, 96.35416666666666],
    )
    np.testing.assert_allclose(
        model.n_out,
        [47.763157894736835, 294.5394736842105, 1165.8854166666665],
    )
    np.testing.assert_allclose(
        model.T_out,
        [999650.88223009, 162105.54846974, 40952.98066604],
        rtol=1e-10,
    )


def test_nrel_5mw_default_modal_result_is_basic_2dof():
    model = NREL_5MW()

    assert model.f_n.size == 2
    np.testing.assert_allclose(model.f_n, [0.0, 2.27549207], rtol=1e-8)
    assert model.mode_shape.shape == (2, 2)
