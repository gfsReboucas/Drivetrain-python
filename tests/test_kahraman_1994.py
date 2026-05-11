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

    n_planets = stage.N_p
    k_ring = stage.sub_set("planet-ring").k_mesh
    k_sun = stage.sub_set("sun-planet").k_mesh
    m_sun = stage.mass[0]
    m_planet = stage.mass[1]
    m_carrier = stage.carrier.mass

    lambda_1 = m_planet*m_carrier*m_sun
    lambda_2 = -(
        n_planets*k_sun*m_planet*m_carrier
        + (k_ring + k_sun)*m_carrier*m_sun
        + n_planets*(k_ring + k_sun)*m_planet*m_sun
    )
    lambda_3 = n_planets*k_ring*k_sun*(n_planets*m_planet + m_carrier + 4.0*m_sun)
    eig_1 = (-lambda_2 - np.sqrt(lambda_2**2 - 4.0*lambda_1*lambda_3))/(2.0*lambda_1)
    eig_2 = (-lambda_2 + np.sqrt(lambda_2**2 - 4.0*lambda_1*lambda_3))/(2.0*lambda_1)

    expected = np.sort(
        np.array(
            [
                0.0,
                *([np.sqrt((k_ring + k_sun)/m_planet)/(2.0*np.pi)]*(n_planets - 1)),
                np.sqrt(eig_1)/(2.0*np.pi),
                np.sqrt(eig_2)/(2.0*np.pi),
            ]
        )
    )

    np.testing.assert_allclose(frequencies, expected, atol=1e-12)


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
