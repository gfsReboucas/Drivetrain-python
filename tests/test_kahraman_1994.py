import numpy as np

from drivetrain.dynamics import Kahraman_94
from drivetrain.models import NREL_5MW


class _ZeroShaft:
    def inertia_matrix(self, option):
        assert option == "torsional"
        return np.zeros((2, 2))

    def stiffness_matrix(self, option):
        assert option == "torsional"
        return np.zeros((2, 2))

    def damping_matrix(self, option):
        assert option == "torsional"
        return np.zeros((2, 2))


class _Carrier:
    mass = 4.0


class _Mesh:
    def __init__(self, k_mesh):
        self.k_mesh = k_mesh


class _PlanetaryStage:
    configuration = "planetary"
    N_p = 3
    a_w = 1000.0
    d = np.array([1000.0, 1000.0])
    mass = np.array([2.0, 3.0])
    carrier = _Carrier()
    output_shaft = _ZeroShaft()

    def sub_set(self, option):
        if option == "planet-ring":
            return _Mesh(11.0)
        if option == "sun-planet":
            return _Mesh(7.0)
        raise ValueError(option)


def test_kahraman_stage_helpers_document_reduced_dof_ordering():
    stage = _PlanetaryStage()

    inertia = Kahraman_94.stage_inertia_matrix(stage)
    stiffness = Kahraman_94.stage_stiffness_matrix(stage)
    damping = Kahraman_94.stage_damping_matrix(stage)

    assert inertia.shape == (6, 6)
    assert stiffness.shape == (6, 6)
    assert damping.shape == (6, 6)
    np.testing.assert_allclose(np.diag(inertia)[:5], [4.0, 0.75, 0.75, 0.75, 0.5])
    np.testing.assert_allclose(inertia[-1], np.zeros(6))
    np.testing.assert_allclose(inertia[:, -1], np.zeros(6))
    np.testing.assert_allclose(stiffness, stiffness.T)
    np.testing.assert_allclose(damping, damping.T)


def test_kahraman_fixed_ring_planetary_stage_matches_analytical_modes():
    stage = _PlanetaryStage()
    inertia = Kahraman_94.stage_inertia_matrix(stage)[:-1, :-1]
    stiffness = Kahraman_94.stage_stiffness_matrix(stage)[:-1, :-1]

    eigenvalues = np.linalg.eigvals(np.linalg.solve(inertia, stiffness)).real
    frequencies = np.sort(np.sqrt(np.clip(eigenvalues, 0.0, None))/(2.0*np.pi))

    expected = Kahraman_94.fixed_ring_planetary_frequencies(stage)

    np.testing.assert_allclose(frequencies, expected, atol=1e-12)


def test_kahraman_fixed_ring_analytical_frequencies_are_stage_parameter_based():
    stage = _PlanetaryStage()
    expected = np.array(
        [0.0, 0.389848400616838, 0.389848400616838, 0.508728970094716, 0.707886793423852]
    )

    np.testing.assert_allclose(Kahraman_94.fixed_ring_planetary_frequencies(stage), expected)


def test_kahraman_dof_descriptions_and_load_vectors_for_nrel_5mw():
    drivetrain = NREL_5MW(dynamic_model=Kahraman_94)
    dynamic_model = drivetrain.dynamic_model(drivetrain)

    assert len(dynamic_model.dof_description) == 2*dynamic_model.n_DOF[-1]
    assert dynamic_model.dof_description[0] == ("Rotor angular displacement, [rad]", "theta_R")
    assert dynamic_model.dof_description[dynamic_model.n_DOF[-1] - 1] == (
        "Generator angular displacement, [rad]",
        "theta_G",
    )
    assert dynamic_model.b.shape == (dynamic_model.n_DOF[-1], 2)
    np.testing.assert_allclose(dynamic_model.b[0], [1.0, 0.0])
    np.testing.assert_allclose(dynamic_model.b[-1], [0.0, 1.0])
    np.testing.assert_allclose(dynamic_model.c, np.zeros(dynamic_model.n_DOF[-1]))
