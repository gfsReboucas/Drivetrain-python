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


class _ParallelStage:
    configuration = "parallel"
    N_p = 1
    d = np.array([1000.0, 2000.0])
    mass = np.array([2.0, 3.0])
    k_mesh = 5.0
    output_shaft = _ZeroShaft()


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


def test_kahraman_fault_stiffness_helpers_are_symmetric_and_component_based():
    stage = _PlanetaryStage()
    sun = Kahraman_94.sun_fault_stiffness_matrix(stage)
    planet = Kahraman_94.planet_fault_stiffness_matrix(stage, planet_index=1)
    ring = Kahraman_94.ring_fault_stiffness_matrix(stage)
    nominal = Kahraman_94.stage_stiffness_matrix(stage)

    np.testing.assert_allclose(sun, sun.T)
    np.testing.assert_allclose(planet, planet.T)
    np.testing.assert_allclose(ring, ring.T)
    np.testing.assert_allclose(sun + ring, nominal)
    assert np.count_nonzero(planet[2]) > 0
    assert np.count_nonzero(planet[1]) == 0
    assert np.count_nonzero(ring[-2]) == 0


def test_kahraman_parallel_fault_uses_full_parallel_mesh_stiffness():
    stage = _ParallelStage()

    np.testing.assert_allclose(
        Kahraman_94.fault_stiffness_matrix(stage, "parallel"),
        Kahraman_94.stage_stiffness_matrix(stage),
    )


def test_kahraman_faulty_inertia_matrix_reduces_selected_component():
    planetary = _PlanetaryStage()
    parallel = _ParallelStage()

    planetary_nominal = Kahraman_94.stage_inertia_matrix(planetary)
    planetary_faulty = Kahraman_94.stage_faulty_inertia_matrix(
        planetary,
        fault_val=0.25,
        planet_index=2,
    )
    parallel_nominal = Kahraman_94.stage_inertia_matrix(parallel)
    parallel_faulty = Kahraman_94.stage_faulty_inertia_matrix(parallel, fault_val=0.25)

    np.testing.assert_allclose(planetary_faulty[3, 3], 0.75*planetary_nominal[3, 3])
    np.testing.assert_allclose(planetary_faulty[1, 1], planetary_nominal[1, 1])
    np.testing.assert_allclose(parallel_faulty[0, 0], 0.75*parallel_nominal[0, 0])
    np.testing.assert_allclose(parallel_faulty[1, 1], parallel_nominal[1, 1])


def test_kahraman_model_level_fault_reduces_selected_stage_matrix():
    drivetrain = NREL_5MW()
    stage = drivetrain.stage[0]

    nominal = Kahraman_94(drivetrain)
    stiffness_fault = Kahraman_94(
        drivetrain,
        fault_type="sun",
        fault_stage=0,
        fault_val=0.1,
    )
    mass_fault = Kahraman_94(
        drivetrain,
        fault_type="mass",
        fault_stage=0,
        fault_val=0.2,
        fault_planet=1,
    )

    stage_slice = slice(nominal.n_DOF[1] - 1, nominal.n_DOF[2])
    expected_stiffness = 0.1*Kahraman_94.sun_fault_stiffness_matrix(stage)
    np.testing.assert_allclose(
        nominal.K[stage_slice, stage_slice] - stiffness_fault.K[stage_slice, stage_slice],
        expected_stiffness,
    )
    np.testing.assert_allclose(
        mass_fault.M[stage_slice.start + 2, stage_slice.start + 2],
        0.8*nominal.M[stage_slice.start + 2, stage_slice.start + 2],
    )


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
