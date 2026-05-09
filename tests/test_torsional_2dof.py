import numpy as np

from drivetrain.dynamic_formulation import torsional_2DOF


class _DummyShaft:
    def __init__(self, torsional_stiffness):
        self._torsional_stiffness = torsional_stiffness

    def stiffness(self, option):
        if option != "torsional":
            raise ValueError(option)
        return self._torsional_stiffness


class _DummyStage:
    def __init__(self, output_shaft):
        self.output_shaft = output_shaft


class _DummyDrivetrain:
    def __init__(self, rotor_inertia, generator_inertia, ratio, main_stiffness, high_speed_stiffness):
        self.J_Rotor = rotor_inertia
        self.J_Gen = generator_inertia
        self.u = np.array([ratio])
        self.main_shaft = _DummyShaft(main_stiffness)
        self.stage = [_DummyStage(_DummyShaft(high_speed_stiffness))]


def test_torsional_2dof_matches_analytical_solution():
    rotor_inertia = 10.0
    generator_inertia = 2.0
    ratio = 3.0
    main_stiffness = 1.2e6
    high_speed_stiffness = 8.0e5

    drivetrain = _DummyDrivetrain(
        rotor_inertia=rotor_inertia,
        generator_inertia=generator_inertia,
        ratio=ratio,
        main_stiffness=main_stiffness,
        high_speed_stiffness=high_speed_stiffness,
    )

    model = torsional_2DOF(drivetrain)

    reflected_generator_inertia = generator_inertia * ratio**2
    equivalent_stiffness = (
        main_stiffness * high_speed_stiffness * ratio**2
    ) / (main_stiffness + high_speed_stiffness * ratio**2)
    expected_elastic_frequency = (
        np.sqrt(equivalent_stiffness * (1.0 / rotor_inertia + 1.0 / reflected_generator_inertia))
        / (2.0 * np.pi)
    )

    np.testing.assert_allclose(model.f_n[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(model.f_n[1], expected_elastic_frequency, rtol=1e-12)
    np.testing.assert_allclose(model.mode_shape[:, 0], [1.0, 1.0], atol=1e-8)
