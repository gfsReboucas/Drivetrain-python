from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la

from drivetrain.components import Bearing
from drivetrain.dynamics import Lin_Parker_99
from drivetrain.models import NREL_5MW


def _bearing_array(entries):
    stiffness = np.array(
        [[0.0, entry["k_y"], entry["k_z"], entry["k_alpha"], 0.0, 0.0] for entry in entries]
    ).T
    damping = np.zeros_like(stiffness)
    return Bearing(
        stiffness=stiffness,
        damping=damping,
        name=[entry["name"] for entry in entries],
    )


class _ZeroLP99Shaft:
    """Shaft double that keeps hand-calculated stage fixtures local to one stage."""

    def inertia_matrix(self, option):
        assert option == "Lin_Parker_99"
        return np.zeros((6, 6))

    def stiffness_matrix(self, option):
        assert option == "Lin_Parker_99"
        return np.zeros((6, 6))


def _lp99_hand_parallel_stage():
    """Two-gear fixture with alpha=0 and base radii 1/2 for direct inspection."""

    return SimpleNamespace(
        configuration="parallel",
        N_p=1,
        alpha_n=0.0,
        mass=np.array([3.0, 5.0]),
        J_x=np.array([6.0, 20.0]),
        d_b=np.array([2000.0, 4000.0]),
        k_mesh=7.0,
        bearing=_bearing_array(
            [
                {"k_y": 2.0, "k_z": 3.0, "k_alpha": 32.0, "name": "wheel-a"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "wheel-b"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "wheel-c"},
                {"k_y": 11.0, "k_z": 13.0, "k_alpha": 17.0, "name": "pinion-a"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "pinion-b"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "pinion-c"},
            ]
        ),
        output_shaft=_ZeroLP99Shaft(),
    )


def _lp99_hand_planetary_stage():
    """One-planet fixture with alpha=0 and unit base radii for compact matrices."""

    mesh = {
        "sun-planet": SimpleNamespace(k_mesh=2.0),
        "planet-ring": SimpleNamespace(k_mesh=3.0),
    }
    return SimpleNamespace(
        configuration="planetary",
        N_p=1,
        alpha_n=0.0,
        mass=np.array([2.0, 3.0, 5.0]),
        J_x=np.array([11.0, 13.0, 17.0]),
        d_b=np.array([2000.0, 2000.0, 2000.0]),
        a_w=1000.0,
        carrier=SimpleNamespace(mass=7.0, J_x=19.0),
        bearing=_bearing_array(
            [
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "planet-a"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "planet-b"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "carrier-a"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "carrier-b"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "sun"},
                {"k_y": 0.0, "k_z": 0.0, "k_alpha": 0.0, "name": "ring"},
            ]
        ),
        sub_set=lambda name: mesh[name],
        output_shaft=_ZeroLP99Shaft(),
    )


def _reference_stage_frequencies(stage):
    inertia = Lin_Parker_99.raw_stage_inertia_matrix(stage)
    stiffness_parts = Lin_Parker_99.raw_stage_stiffness_matrix(stage)
    stiffness = stiffness_parts["K_b"] + stiffness_parts["K_m"]
    eigenvalues = la.eigvals(stiffness, inertia)
    eigenvalues = np.real_if_close(eigenvalues)
    eigenvalues = np.real(eigenvalues[np.isreal(eigenvalues)])
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    eigenvalues[eigenvalues < 0.0] = 0.0
    frequencies = np.sqrt(eigenvalues)/(2.0*np.pi)
    frequencies[frequencies < 1.0e-4] = 0.0
    return np.sort(frequencies)


def _lin_parker_1999_validation_stage(num_planet):
    # Lin and Parker 1999 Table 1 / Lin thesis Table C.1.
    m_s = 0.4
    m_r = 2.35
    m_c = 5.43
    m_p = 0.66
    d_s = 77.4
    d_r = 275.0
    d_c = 176.8
    d_p = 100.3

    r_s = d_s*1.0e-3/2
    r_r = d_r*1.0e-3/2
    r_c = d_c*1.0e-3/2
    r_p = d_p*1.0e-3/2

    k_sp = 5.0e8
    k_s = 1.0e8
    k_r = k_s
    k_c = k_s
    k_p = k_s
    k_su = 0.0
    k_ru = 1.0e9
    k_cu = k_su
    k_pu = k_p

    mesh = SimpleNamespace(k_mesh=k_sp)
    return SimpleNamespace(
        configuration="planetary",
        N_p=num_planet,
        alpha_n=24.6,
        mass=np.array([m_s, m_p, m_r]),
        d_b=np.array([d_s, d_p, d_r]),
        a_w=d_c/2,
        J_x=np.array([0.39*r_s**2, 0.61*r_p**2, 3.0*r_r**2]),
        carrier=SimpleNamespace(mass=m_c, J_x=6.29*r_c**2),
        bearing=_bearing_array(
            [
                {"k_y": 0.5*k_p, "k_z": 0.5*k_p, "k_alpha": 0.5*k_pu*r_p**2, "name": "planet-a"},
                {"k_y": 0.5*k_p, "k_z": 0.5*k_p, "k_alpha": 0.5*k_pu*r_p**2, "name": "planet-b"},
                {"k_y": 0.5*k_c, "k_z": 0.5*k_c, "k_alpha": 0.5*k_cu*r_c**2, "name": "carrier-a"},
                {"k_y": 0.5*k_c, "k_z": 0.5*k_c, "k_alpha": 0.5*k_cu*r_c**2, "name": "carrier-b"},
                {"k_y": k_s, "k_z": k_s, "k_alpha": k_su*r_s**2, "name": "sun"},
                {"k_y": k_r, "k_z": k_r, "k_alpha": k_ru*r_r**2, "name": "ring"},
            ]
        ),
        sub_set=lambda name: mesh,
    )


def _lin_parker_1999_reference_frequencies(num_planet):
    if num_planet == 3:
        frequencies = [
            0.0,
            1475.7,
            1930.3,
            2658.3,
            7462.8,
            11775.3,
            *([743.2, 1102.4, 1896.0, 2276.4, 6986.3, 9647.9]*2),
        ]
    elif num_planet == 4:
        frequencies = [
            0.0,
            1536.6,
            1970.6,
            2625.7,
            7773.6,
            13071.1,
            *([727.0, 1091.0, 1892.8, 2342.5, 7189.9, 10437.6]*2),
            1808.2,
            5963.8,
            6981.7,
        ]
    elif num_planet == 5:
        frequencies = [
            0.0,
            1567.4,
            2006.1,
            2614.8,
            8065.4,
            14253.1,
            *([710.0, 1072.0, 1888.1, 2425.3, 7382.4, 11172.3]*2),
            *([1808.2, 5963.8, 6981.7]*2),
        ]
    else:
        raise ValueError("num_planet must be 3, 4, or 5")

    return np.sort(frequencies)


def _cooley_parker_2012_validation_stage(num_planet):
    # Cooley and Parker 2012 Table 1, evaluated at Omega_c = 0.
    m_s = 3.0
    m_r = 7.64
    m_c = 12.0
    m_p = 1.86
    d_s = 55.75
    d_r = 109.7
    d_c = 88.6
    d_p = 27.0

    r_s = d_s*1.0e-3/2
    r_r = d_r*1.0e-3/2
    r_c = d_c*1.0e-3/2
    r_p = d_p*1.0e-3/2

    k_sp = 100.0e6
    k_s = 20.0e6
    k_r = 100.0e6
    k_c = 50.0e6
    k_p = 10.0e6
    k_su = 20.0e6
    k_ru = 100.0e6
    k_cu = k_ru
    k_pu = k_p

    mesh = SimpleNamespace(k_mesh=k_sp)
    return SimpleNamespace(
        configuration="planetary",
        N_p=num_planet,
        alpha_n=20.0,
        mass=np.array([m_s, m_p, m_r]),
        d_b=np.array([d_s, d_p, d_r]),
        a_w=d_c/2,
        J_x=np.array([1.75*r_s**2, 1.25*r_p**2, 8.09*r_r**2]),
        carrier=SimpleNamespace(mass=m_c, J_x=6.80*r_c**2),
        bearing=_bearing_array(
            [
                {"k_y": 0.5*k_p, "k_z": 0.5*k_p, "k_alpha": 0.5*k_pu*r_p**2, "name": "planet-a"},
                {"k_y": 0.5*k_p, "k_z": 0.5*k_p, "k_alpha": 0.5*k_pu*r_p**2, "name": "planet-b"},
                {"k_y": 0.5*k_c, "k_z": 0.5*k_c, "k_alpha": 0.5*k_cu*r_c**2, "name": "carrier-a"},
                {"k_y": 0.5*k_c, "k_z": 0.5*k_c, "k_alpha": 0.5*k_cu*r_c**2, "name": "carrier-b"},
                {"k_y": k_s, "k_z": k_s, "k_alpha": k_su*r_s**2, "name": "sun"},
                {"k_y": k_r, "k_z": k_r, "k_alpha": k_ru*r_r**2, "name": "ring"},
            ]
        ),
        sub_set=lambda name: mesh,
    )


def _cooley_parker_2012_reference_frequencies(num_planet):
    if num_planet == 3:
        nondimensional = [
            0.9296,
            1.090,
            1.358,
            1.931,
            5.566,
            7.741,
            *([0.7432, 1.018, 1.265, 1.487, 4.976, 6.282]*2),
        ]
    elif num_planet == 4:
        nondimensional = [
            0.9077,
            1.046,
            1.317,
            2.013,
            5.809,
            8.383,
            *([0.7199, 1.001, 1.317, 1.481, 5.153, 6.488]*2),
            0.9617,
            4.320,
            5.672,
        ]
    elif num_planet == 5:
        nondimensional = [
            0.8802,
            1.019,
            1.281,
            2.090,
            6.030,
            8.986,
            *([0.6992, 0.9863, 1.344, 1.496, 5.317, 6.693]*2),
            *([0.9617, 4.320, 5.672]*2),
        ]
    else:
        raise ValueError("num_planet must be 3, 4, or 5")

    k_p = 10.0e6
    m_p = 1.86
    frequency_scale = np.sqrt(k_p/m_p)/(2.0*np.pi)
    return np.sort(np.array(nondimensional)*frequency_scale)


def _bearing_matrix(x_stiffness, y_stiffness, torsional_stiffness=0.0):
    return np.diag([x_stiffness, y_stiffness, torsional_stiffness])


def _body_inertia_matrix(mass, polar_inertia, base_radius):
    return np.diag([mass, mass, polar_inertia/(base_radius**2)])


def _body_gyroscopic_matrix(mass):
    return np.array(
        [
            [0.0, -2.0*mass, 0.0],
            [2.0*mass, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )


def _clean_numerical_noise(matrix):
    matrix = matrix.copy()
    matrix[abs(matrix) <= 1.0e-4] = 0.0
    return matrix


def _stage_transform_to_assembled_coordinates(stage, matrix):
    transform = Lin_Parker_99.stage_coordinate_change(stage)
    return transform.T @ matrix @ transform


def _with_output_interface(matrix):
    return la.block_diag(matrix, np.zeros((3, 3)))


def _assert_positive_semidefinite(matrix, atol=1.0e-5):
    """Assert matrix has no physically meaningful negative eigenvalues."""
    symmetric = (matrix + matrix.T)/2
    eigenvalues = np.linalg.eigvalsh(symmetric)

    assert eigenvalues[0] >= -atol


def _planet_block_rotation_matrix(stage):
    """Return a permutation that cyclically reindexes equal planet DOF blocks."""
    matrix_size = 9 + 3*stage.N_p
    permutation = np.eye(matrix_size)
    planet_start = 9
    planet_size = 3*stage.N_p
    permutation[planet_start:, planet_start:] = np.roll(
        np.eye(planet_size),
        shift=3,
        axis=1,
    )
    return permutation


def _paper_stage_inertia_matrix(stage):
    if stage.configuration == "parallel":
        pinion_radius = stage.d_b[0]*1.0e-3/2
        wheel_radius = stage.d_b[1]*1.0e-3/2
        raw = la.block_diag(
            _body_inertia_matrix(stage.mass[0], stage.J_x[0], pinion_radius),
            _body_inertia_matrix(stage.mass[1], stage.J_x[1], wheel_radius),
        )
    elif stage.configuration == "planetary":
        carrier_radius = stage.a_w*1.0e-3
        ring_radius = stage.d_b[2]*1.0e-3/2
        sun_radius = stage.d_b[0]*1.0e-3/2
        planet_radius = stage.d_b[1]*1.0e-3/2
        raw = la.block_diag(
            _body_inertia_matrix(stage.carrier.mass, stage.carrier.J_x, carrier_radius),
            _body_inertia_matrix(stage.mass[2], stage.J_x[2], ring_radius),
            _body_inertia_matrix(stage.mass[0], stage.J_x[0], sun_radius),
            *[
                _body_inertia_matrix(stage.mass[1], stage.J_x[1], planet_radius)
                for _ in range(stage.N_p)
            ],
        )
    else:
        raise ValueError(stage.configuration)

    return _with_output_interface(_stage_transform_to_assembled_coordinates(stage, raw))


def _paper_stage_gyroscopic_matrix(stage):
    if stage.configuration == "parallel":
        raw = la.block_diag(
            _body_gyroscopic_matrix(stage.mass[0]),
            _body_gyroscopic_matrix(stage.mass[1]),
        )
    elif stage.configuration == "planetary":
        raw = la.block_diag(
            _body_gyroscopic_matrix(stage.carrier.mass),
            _body_gyroscopic_matrix(stage.mass[2]),
            _body_gyroscopic_matrix(stage.mass[0]),
            *[_body_gyroscopic_matrix(stage.mass[1]) for _ in range(stage.N_p)],
        )
    else:
        raise ValueError(stage.configuration)

    return _with_output_interface(_stage_transform_to_assembled_coordinates(stage, raw))


def _mesh_blocks(stage):
    alpha_n = np.radians(stage.alpha_n)

    def psi(index):
        return (index - 1)*(2*np.pi/stage.N_p)

    def psi_s(index):
        return psi(index) - alpha_n

    def psi_r(index):
        return psi(index) + alpha_n

    def ring_body_vector(index):
        return np.array([np.sin(psi_r(index)), -np.cos(psi_r(index)), -1.0])

    def sun_body_vector(index):
        return np.array([np.sin(psi_s(index)), -np.cos(psi_s(index)), -1.0])

    def planet_sun_vector():
        return np.array([np.sin(alpha_n), np.cos(alpha_n), -1.0])

    def planet_ring_vector():
        return np.array([np.sin(alpha_n), -np.cos(alpha_n), -1.0])

    def ring_ring(k_mesh, index):
        return k_mesh*np.outer(ring_body_vector(index), ring_body_vector(index))

    def ring_planet(k_mesh, index):
        return -k_mesh*np.outer(ring_body_vector(index), planet_ring_vector())

    def sun_sun(k_mesh, index):
        return k_mesh*np.outer(sun_body_vector(index), sun_body_vector(index))

    def sun_planet(k_mesh, index):
        return k_mesh*np.outer(sun_body_vector(index), planet_sun_vector())

    def planet_sun(k_mesh):
        return k_mesh*np.outer(planet_sun_vector(), planet_sun_vector())

    def planet_ring(k_mesh):
        return k_mesh*np.outer(planet_ring_vector(), planet_ring_vector())

    def carrier_planet(k_mesh, index):
        return k_mesh*np.array(
            [
                [-np.cos(psi(index)), np.sin(psi(index)), 0.0],
                [-np.sin(psi(index)), -np.cos(psi(index)), 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

    return {
        "ring_ring": ring_ring,
        "ring_planet": ring_planet,
        "sun_sun": sun_sun,
        "sun_planet": sun_planet,
        "planet_sun": planet_sun,
        "planet_ring": planet_ring,
        "carrier_planet": carrier_planet,
    }


def _paper_parallel_stage_stiffness_matrices(stage):
    pinion_radius = stage.d_b[0]*1.0e-3/2
    wheel_radius = stage.d_b[1]*1.0e-3/2
    mesh = _mesh_blocks(stage)

    pinion_bearing = stage.bearing[3:].parallel_association()
    pinion_bearing_matrix = _bearing_matrix(
        pinion_bearing.k_y,
        pinion_bearing.k_z,
        pinion_bearing.k_alpha/(pinion_radius**2),
    )
    raw_bearing = la.block_diag(np.zeros((3, 3)), pinion_bearing_matrix)

    wheel_bearing = stage.bearing[:3].parallel_association()
    wheel_support_matrix = _bearing_matrix(
        wheel_bearing.k_y,
        wheel_bearing.k_z,
        wheel_bearing.k_alpha/(wheel_radius**2),
    )
    mesh_coupling = mesh["sun_planet"](stage.k_mesh, 1)
    raw_mesh = np.block(
        [
            [mesh["planet_sun"](stage.k_mesh) + wheel_support_matrix, mesh_coupling.T],
            [mesh_coupling, mesh["sun_sun"](stage.k_mesh, 1)],
        ]
    )
    raw_centripetal = la.block_diag(
        _bearing_matrix(stage.mass[0], stage.mass[0]),
        _bearing_matrix(stage.mass[1], stage.mass[1]),
    )

    return {
        "K_b": raw_bearing,
        "K_m": raw_mesh,
        "K_Omega": raw_centripetal,
    }


def _paper_planetary_stage_stiffness_matrices(stage):
    carrier_radius = stage.a_w*1.0e-3
    planet_radius = stage.d_b[1]*1.0e-3/2
    mesh = _mesh_blocks(stage)
    n_planet_dof = 3*stage.N_p

    bearing_count = len(stage.bearing.k_y)
    if bearing_count >= 5:
        sun_radius = stage.d_b[0]*1.0e-3/2
        sun_bearing = stage.bearing[4]
        sun_bearing_matrix = _bearing_matrix(
            sun_bearing.k_y,
            sun_bearing.k_z,
            sun_bearing.k_alpha/(sun_radius**2),
        )
    else:
        sun_bearing_matrix = np.zeros((3, 3))
    if bearing_count >= 6:
        ring_radius = stage.d_b[2]*1.0e-3/2
        ring_bearing = stage.bearing[5]
        ring_bearing_matrix = _bearing_matrix(
            ring_bearing.k_y,
            ring_bearing.k_z,
            ring_bearing.k_alpha/(ring_radius**2),
        )
    else:
        ring_bearing_matrix = np.zeros((3, 3))

    carrier_bearing = stage.bearing[2:4].parallel_association()
    carrier_bearing_matrix = _bearing_matrix(
        carrier_bearing.k_y,
        carrier_bearing.k_z,
        carrier_bearing.k_alpha/(carrier_radius**2),
    )
    raw_bearing = la.block_diag(
        carrier_bearing_matrix,
        ring_bearing_matrix,
        sun_bearing_matrix,
        np.zeros((n_planet_dof, n_planet_dof)),
    )

    planet_bearing = stage.bearing[:2].parallel_association()
    planet_bearing_matrix = _bearing_matrix(
        planet_bearing.k_y,
        planet_bearing.k_z,
        planet_bearing.k_alpha/(planet_radius**2),
    )
    k_sp = stage.sub_set("sun-planet").k_mesh
    k_pr = stage.sub_set("planet-ring").k_mesh

    carrier_planet_row = np.zeros((3, n_planet_dof))
    ring_planet_row = np.zeros_like(carrier_planet_row)
    sun_planet_row = np.zeros_like(carrier_planet_row)
    carrier_sum = np.zeros((3, 3))

    for planet_index in range(stage.N_p):
        index = planet_index + 1
        planet_dofs = slice(3*planet_index, 3*(planet_index + 1))
        carrier_planet_coupling = mesh["carrier_planet"](1.0, index)

        carrier_planet_row[:, planet_dofs] = (
            carrier_planet_coupling @ planet_bearing_matrix
        )
        ring_planet_row[:, planet_dofs] = mesh["ring_planet"](k_pr, index)
        sun_planet_row[:, planet_dofs] = mesh["sun_planet"](k_sp, index)
        carrier_sum += (
            carrier_planet_coupling
            @ planet_bearing_matrix
            @ carrier_planet_coupling.T
        )

    ring_sum = sum(mesh["ring_ring"](k_pr, index + 1) for index in range(stage.N_p))
    sun_sum = sum(mesh["sun_sun"](k_sp, index + 1) for index in range(stage.N_p))
    body_diagonal = la.block_diag(carrier_sum, ring_sum, sun_sum)
    body_planet_coupling = np.vstack(
        (carrier_planet_row, ring_planet_row, sun_planet_row)
    )
    planet_diagonal = (
        _bearing_matrix(planet_bearing.k_y, planet_bearing.k_z)
        + mesh["planet_ring"](k_pr)
        + mesh["planet_sun"](k_sp)
    )
    planet_block = la.block_diag(*[planet_diagonal for _ in range(stage.N_p)])

    raw_mesh = np.block(
        [
            [body_diagonal, body_planet_coupling],
            [body_planet_coupling.T, planet_block],
        ]
    )
    raw_centripetal = la.block_diag(
        _bearing_matrix(stage.carrier.mass, stage.carrier.mass),
        _bearing_matrix(stage.mass[2], stage.mass[2]),
        _bearing_matrix(stage.mass[0], stage.mass[0]),
        *[_bearing_matrix(stage.mass[1], stage.mass[1]) for _ in range(stage.N_p)],
    )

    return {
        "K_b": raw_bearing,
        "K_m": raw_mesh,
        "K_Omega": raw_centripetal,
    }


def _paper_stage_stiffness_matrices(stage):
    if stage.configuration == "parallel":
        raw_stiffness = _paper_parallel_stage_stiffness_matrices(stage)
    elif stage.configuration == "planetary":
        raw_stiffness = _paper_planetary_stage_stiffness_matrices(stage)
    else:
        raise ValueError(stage.configuration)

    return {
        key: _clean_numerical_noise(
            _with_output_interface(
                _stage_transform_to_assembled_coordinates(stage, matrix)
            )
        )
        for key, matrix in raw_stiffness.items()
    }


def test_lin_parker_99_public_class_instantiates_for_nrel_5mw():
    drivetrain = NREL_5MW(dynamic_model=Lin_Parker_99)

    assert drivetrain.f_n.shape == (42,)
    assert drivetrain.mode_shape.shape == (42, 42)
    assert np.isfinite(drivetrain.f_n).sum() >= 39


def test_lin_parker_99_stage_matrix_dimensions_for_nrel_stages():
    planetary = NREL_5MW.gear_set(0)
    parallel = NREL_5MW.gear_set(2)

    planetary_inertia = Lin_Parker_99.stage_inertia_matrix(planetary)
    planetary_gyroscopic = Lin_Parker_99.stage_gyroscopic_matrix(planetary)
    planetary_stiffness = Lin_Parker_99.stage_stiffness_matrix(planetary)
    parallel_inertia = Lin_Parker_99.stage_inertia_matrix(parallel)
    parallel_gyroscopic = Lin_Parker_99.stage_gyroscopic_matrix(parallel)
    parallel_stiffness = Lin_Parker_99.stage_stiffness_matrix(parallel)

    assert planetary_inertia.shape == (18, 18)
    assert planetary_gyroscopic.shape == (18, 18)
    assert planetary_stiffness["K_b"].shape == (18, 18)
    assert planetary_stiffness["K_m"].shape == (18, 18)
    assert planetary_stiffness["K_Omega"].shape == (18, 18)
    assert parallel_inertia.shape == (9, 9)
    assert parallel_gyroscopic.shape == (9, 9)
    assert parallel_stiffness["K_b"].shape == (9, 9)
    assert parallel_stiffness["K_m"].shape == (9, 9)
    assert parallel_stiffness["K_Omega"].shape == (9, 9)


def test_lin_parker_99_stage_inertia_and_gyroscopic_matrices_match_lp99_forms():
    # Lin and Parker write each body block as diag(m, m, J/r^2) and each
    # gyroscopic block as [[0, -2m, 0], [2m, 0, 0], [0, 0, 0]] before the
    # stage coordinate transformation.
    for stage_index in range(3):
        stage = NREL_5MW.gear_set(stage_index)

        np.testing.assert_allclose(
            Lin_Parker_99.stage_inertia_matrix(stage),
            _paper_stage_inertia_matrix(stage),
        )
        np.testing.assert_allclose(
            Lin_Parker_99.stage_gyroscopic_matrix(stage),
            _paper_stage_gyroscopic_matrix(stage),
        )


def test_lin_parker_99_stage_stiffness_matrices_match_lp99_assembly():
    # The expected matrices are assembled in the paper's raw body coordinates,
    # then mapped through stage_coordinate_change to match the implementation.
    for stage_index in range(3):
        stage = NREL_5MW.gear_set(stage_index)
        actual = Lin_Parker_99.stage_stiffness_matrix(stage)
        expected = _paper_stage_stiffness_matrices(stage)

        for key in ("K_b", "K_m", "K_Omega"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1.0e-4)


def test_lin_parker_99_hand_calculated_parallel_fixture_raw_matrices():
    """Check raw LP99 matrices against a small fixture evaluated by hand."""

    stage = _lp99_hand_parallel_stage()

    expected_inertia = np.diag([3.0, 3.0, 6.0, 5.0, 5.0, 5.0])
    expected_gyro = la.block_diag(
        np.array([[0.0, -6.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, -10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    expected_bearing = la.block_diag(np.zeros((3, 3)), np.diag([11.0, 13.0, 17.0]))
    expected_mesh = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, -7.0, 0.0, -7.0, -7.0],
            [0.0, -7.0, 15.0, 0.0, 7.0, 7.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -7.0, 7.0, 0.0, 7.0, 7.0],
            [0.0, -7.0, 7.0, 0.0, 7.0, 7.0],
        ]
    )
    expected_centripetal = np.diag([3.0, 3.0, 0.0, 5.0, 5.0, 0.0])

    stiffness = Lin_Parker_99.raw_stage_stiffness_matrix(stage)

    np.testing.assert_allclose(Lin_Parker_99.raw_stage_inertia_matrix(stage), expected_inertia)
    np.testing.assert_allclose(Lin_Parker_99.raw_stage_gyroscopic_matrix(stage), expected_gyro)
    np.testing.assert_allclose(stiffness["K_b"], expected_bearing)
    np.testing.assert_allclose(stiffness["K_m"], expected_mesh)
    np.testing.assert_allclose(stiffness["K_Omega"], expected_centripetal)


def test_lin_parker_99_hand_calculated_one_planet_fixture_raw_matrices():
    """Check planetary LP99 matrices against a compact one-planet hand fixture."""

    stage = _lp99_hand_planetary_stage()

    expected_inertia = np.diag([7.0, 7.0, 19.0, 5.0, 5.0, 17.0, 2.0, 2.0, 11.0, 3.0, 3.0, 13.0])
    expected_gyro = la.block_diag(
        np.array([[0.0, -14.0, 0.0], [14.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, -10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, -4.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, -6.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    expected_bearing = np.zeros((12, 12))
    expected_mesh = np.zeros((12, 12))
    expected_mesh[3:6, 3:6] = 3.0 * np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    expected_mesh[6:9, 6:9] = 2.0 * np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    expected_mesh[3:6, 9:12] = 3.0 * np.array([[0.0, 0.0, 0.0], [0.0, -1.0, -1.0], [0.0, -1.0, -1.0]])
    expected_mesh[6:9, 9:12] = 2.0 * np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, -1.0, 1.0]])
    expected_mesh[9:12, 3:6] = expected_mesh[3:6, 9:12].T
    expected_mesh[9:12, 6:9] = expected_mesh[6:9, 9:12].T
    expected_mesh[9:12, 9:12] = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 1.0], [0.0, 1.0, 5.0]])
    expected_centripetal = np.diag([7.0, 7.0, 0.0, 5.0, 5.0, 0.0, 2.0, 2.0, 0.0, 3.0, 3.0, 0.0])

    stiffness = Lin_Parker_99.raw_stage_stiffness_matrix(stage)

    np.testing.assert_allclose(Lin_Parker_99.raw_stage_inertia_matrix(stage), expected_inertia)
    np.testing.assert_allclose(Lin_Parker_99.raw_stage_gyroscopic_matrix(stage), expected_gyro)
    np.testing.assert_allclose(stiffness["K_b"], expected_bearing)
    np.testing.assert_allclose(stiffness["K_m"], expected_mesh)
    np.testing.assert_allclose(stiffness["K_Omega"], expected_centripetal)


@pytest.mark.parametrize("num_planet", [3, 4, 5])
def test_lin_parker_99_reference_case_01_matches_published_frequencies(num_planet):
    stage = _lin_parker_1999_validation_stage(num_planet)
    actual = _reference_stage_frequencies(stage)
    expected = _lin_parker_1999_reference_frequencies(num_planet)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1.0e-2, atol=1.0e-8)


@pytest.mark.parametrize("num_planet", [3, 4, 5])
def test_lin_parker_99_reference_case_02_matches_published_frequencies(num_planet):
    stage = _cooley_parker_2012_validation_stage(num_planet)
    actual = _reference_stage_frequencies(stage)
    expected = _cooley_parker_2012_reference_frequencies(num_planet)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1.0e-2, atol=1.0e-8)


def test_lin_parker_99_stage_coordinate_change_dimensions():
    planetary = NREL_5MW.gear_set(0)
    parallel = NREL_5MW.gear_set(2)

    planetary_transform = Lin_Parker_99.stage_coordinate_change(planetary)
    parallel_transform = Lin_Parker_99.stage_coordinate_change(parallel)

    assert planetary_transform.shape == (18, 15)
    assert parallel_transform.shape == (6, 6)


def test_lin_parker_99_stage_output_shaft_inclusion_is_explicit():
    stage = NREL_5MW.gear_set(2)

    inertia_without_shaft = Lin_Parker_99.stage_inertia_matrix(stage)
    inertia_with_shaft = Lin_Parker_99.stage_inertia_matrix(stage, include_output_shaft=True)
    stiffness_without_shaft = Lin_Parker_99.stage_stiffness_matrix(stage)
    stiffness_with_shaft = Lin_Parker_99.stage_stiffness_matrix(stage, include_output_shaft=True)

    assert np.count_nonzero(inertia_without_shaft[-6:, -6:]) < np.count_nonzero(
        inertia_with_shaft[-6:, -6:]
    )
    assert np.count_nonzero(stiffness_without_shaft["K_b"][-6:, -6:]) < np.count_nonzero(
        stiffness_with_shaft["K_b"][-6:, -6:]
    )


def test_lin_parker_99_full_model_shaft_inclusion_is_explicit():
    drivetrain = NREL_5MW()

    with_shafts = Lin_Parker_99(drivetrain)
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        without_shafts = Lin_Parker_99(drivetrain, include_shafts=False)

    assert np.count_nonzero(without_shafts.M) < np.count_nonzero(with_shafts.M)
    assert np.count_nonzero(without_shafts.K_b) < np.count_nonzero(with_shafts.K_b)


def test_lin_parker_99_gyroscopic_matrices_are_skew_symmetric():
    drivetrain = NREL_5MW()
    dynamic_model = Lin_Parker_99(drivetrain)

    np.testing.assert_allclose(dynamic_model.G, -dynamic_model.G.T)
    assert dynamic_model.G.shape == dynamic_model.M.shape

    for stage_index in range(3):
        gyroscopic = Lin_Parker_99.stage_gyroscopic_matrix(NREL_5MW.gear_set(stage_index))
        np.testing.assert_allclose(gyroscopic, -gyroscopic.T)


def test_lin_parker_99_stage_stiffness_matrices_are_symmetric():
    for stage_index in range(3):
        stiffness = Lin_Parker_99.stage_stiffness_matrix(NREL_5MW.gear_set(stage_index))

        np.testing.assert_allclose(stiffness["K_b"], stiffness["K_b"].T)
        np.testing.assert_allclose(stiffness["K_m"], stiffness["K_m"].T)
        np.testing.assert_allclose(stiffness["K_Omega"], stiffness["K_Omega"].T)


def test_lin_parker_99_inertia_matrices_are_positive_semidefinite():
    drivetrain = NREL_5MW()
    dynamic_model = Lin_Parker_99(drivetrain)

    _assert_positive_semidefinite(dynamic_model.M)
    for stage_index in range(3):
        stage = NREL_5MW.gear_set(stage_index)
        raw_inertia = Lin_Parker_99.raw_stage_inertia_matrix(stage)
        stage_inertia = Lin_Parker_99.stage_inertia_matrix(stage)

        assert np.linalg.eigvalsh(raw_inertia)[0] > 0.0
        _assert_positive_semidefinite(stage_inertia)


def test_lin_parker_99_stiffness_components_are_positive_semidefinite():
    drivetrain = NREL_5MW()
    dynamic_model = Lin_Parker_99(drivetrain)

    for matrix in (dynamic_model.K_b, dynamic_model.K_m, dynamic_model.K_Omega):
        _assert_positive_semidefinite(matrix, atol=1.0e-4)

    for stage_index in range(3):
        stage = NREL_5MW.gear_set(stage_index)
        raw_stiffness = Lin_Parker_99.raw_stage_stiffness_matrix(stage)
        stage_stiffness = Lin_Parker_99.stage_stiffness_matrix(stage)

        for key in ("K_b", "K_m", "K_Omega"):
            _assert_positive_semidefinite(raw_stiffness[key], atol=1.0e-4)
            _assert_positive_semidefinite(stage_stiffness[key], atol=1.0e-4)


@pytest.mark.parametrize("num_planet", [3, 4, 5])
def test_lin_parker_99_planet_block_rotation_preserves_reference_stage_spectra(num_planet):
    stage = _lin_parker_1999_validation_stage(num_planet)
    permutation = _planet_block_rotation_matrix(stage)
    raw_stiffness = Lin_Parker_99.raw_stage_stiffness_matrix(stage)

    for matrix in (
        Lin_Parker_99.raw_stage_inertia_matrix(stage),
        Lin_Parker_99.raw_stage_gyroscopic_matrix(stage),
        raw_stiffness["K_b"],
        raw_stiffness["K_m"],
        raw_stiffness["K_Omega"],
    ):
        rotated = permutation.T @ matrix @ permutation
        np.testing.assert_allclose(
            np.sort_complex(np.linalg.eigvals(rotated)),
            np.sort_complex(np.linalg.eigvals(matrix)),
            atol=1.0e-6,
        )


def test_lin_parker_99_stage_total_stiffness_has_no_significant_negative_eigenvalues():
    for stage_index in range(3):
        stiffness = Lin_Parker_99.stage_stiffness_matrix(NREL_5MW.gear_set(stage_index))
        stage_total = stiffness["K_b"] + stiffness["K_m"] + stiffness["K_Omega"]
        eigenvalues = np.linalg.eigvalsh((stage_total + stage_total.T)/2)

        assert eigenvalues[0] >= -1.0e-5
