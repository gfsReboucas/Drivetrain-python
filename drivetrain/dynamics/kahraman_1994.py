"""Kahraman 1994 drivetrain dynamic formulation.

Reference:
    A. Kahraman, "Natural Modes of Planetary Gear Trains", Journal of Sound
    and Vibration, vol. 173, no. 1, pp. 125-130, 1994.
    https://doi.org/10.1006/jsvi.1994.1222.

The reduced stage DOF ordering is:

- parallel stage: wheel, pinion, output shaft
- planetary stage: carrier, planet 1..N, sun, output shaft

When assembled into a drivetrain model, the first DOF is the rotor and the
last DOF is the generator. Adjacent stage/output-shaft coordinates overlap at
stage interfaces.
"""

import numpy as np

from .base import model


class Kahraman_94(model):
    def __init__(self, dtrain, fault_type="", fault_stage=None, fault_val=0.0, fault_planet=0):
        super().__init__(dtrain)
        self.fault_type = self._normalize_fault_type(fault_type)
        self.fault_stage = fault_stage
        self.fault_val = fault_val
        self.fault_planet = fault_planet
        self.has_fault = self.fault_type != ""
        
        # number of DOFs for each stage:
        self.n_DOF = self.__calc_NDOF()
        
        self.M = self.__inertia_matrix()
        self.K = self.__stiffness_matrix()
        self.K_m = self.K
        self.K_b = np.zeros_like(self.K)
        self.K_Omega = np.zeros_like(self.K)
        self.D = self.damping_matrix()
        self.c = self.centripetal_force_vector()
        self.b = self.external_load_vector()
        self.A = self.state_matrix()
        self.dof_description = self.explain_DOF()
        
        modA = self.modal_analysis()
        self.f_n = modA['f_n']
        self.mode_shape = modA['mode_shape']

    def explain_DOF(self):
        """Return displacement and speed descriptions for each Kahraman DOF."""
        n = self.n_DOF[-1]
        displacement = [None]*n
        displacement[0] = ("Rotor angular displacement, [rad]", "theta_R")
        displacement[-1] = ("Generator angular displacement, [rad]", "theta_G")

        cursor = 1
        for stage_index, stage in enumerate(self.drivetrain.stage, start=1):
            if stage.configuration == "parallel":
                displacement[cursor] = (
                    f"Stage {stage_index}: Wheel angular displacement, [rad]",
                    f"theta_W{stage_index}",
                )
                cursor += 1
                displacement[cursor] = (
                    f"Stage {stage_index}: Pinion angular displacement, [rad]",
                    f"theta_P{stage_index}",
                )
                cursor += 1
            elif stage.configuration == "planetary":
                displacement[cursor] = (
                    f"Stage {stage_index}: Carrier angular displacement, [rad]",
                    f"theta_c{stage_index}",
                )
                cursor += 1
                for planet_index in range(1, stage.N_p + 1):
                    displacement[cursor] = (
                        f"Stage {stage_index}: Planet {planet_index} angular displacement, [rad]",
                        f"theta_p{stage_index}{planet_index}",
                    )
                    cursor += 1
                displacement[cursor] = (
                    f"Stage {stage_index}: Sun angular displacement, [rad]",
                    f"theta_s{stage_index}",
                )
                cursor += 1

        speed = [
            (
                description.replace("displacement, [rad]", "speed, [rad/s]"),
                symbol.replace("theta", "omega"),
            )
            for description, symbol in displacement
        ]
        return displacement + speed
    
    def __calc_NDOF(self):
        stage = self.drivetrain.stage
        Np = [0, 2]
        
        for i in range(len(stage)):
            Np.append(Np[-1] + sum([stage[i].N_p + 1 if(stage[i].configuration == 'parallel')
                                    else stage[i].N_p + 2]))
        
        return Np
        
    def __inertia_matrix(self):
        
        DT = self.drivetrain
        
        N = self.n_DOF

        M = np.zeros((N[-1], N[-1]))

        M[0 , 0 ] = DT.J_Rotor # [kg-m^2], Rotor inertia
        M[-1, -1] = DT.J_Gen   # [kg-m^2], Generator inertia
        
        i = 0
        sub_range = slice(N[i], N[i + 1])
        M[sub_range, 
          sub_range] += DT.main_shaft.inertia_matrix('torsional')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 1, N[i + 2])
            if self._stage_has_fault(i) and self.fault_type == "mass":
                stage_inertia = Kahraman_94.stage_faulty_inertia_matrix(
                    DT.stage[i],
                    self._fault_value_for_stage(i),
                    planet_index=self.fault_planet,
                )
            else:
                stage_inertia = Kahraman_94.stage_inertia_matrix(DT.stage[i])
            M[sub_range, 
              sub_range] += stage_inertia
    
        return M

    @staticmethod
    def fixed_ring_planetary_frequencies(stage):
        """Return the analytical fixed-ring planetary stage frequencies.

        This is Kahraman's reduced torsional planetary model in the same
        coordinates used by ``stage_inertia_matrix`` and
        ``stage_stiffness_matrix``: carrier, planet 1..N, sun.
        """
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
        lambda_3 = n_planets*k_ring*k_sun*(
            n_planets*m_planet + m_carrier + 4.0*m_sun
        )
        discriminant = lambda_2**2 - 4.0*lambda_1*lambda_3
        eig_1 = (-lambda_2 - np.sqrt(discriminant))/(2.0*lambda_1)
        eig_2 = (-lambda_2 + np.sqrt(discriminant))/(2.0*lambda_1)

        repeated_frequency = np.sqrt((k_ring + k_sun)/m_planet)/(2.0*np.pi)
        frequencies = np.array(
            [
                0.0,
                *([repeated_frequency]*(n_planets - 1)),
                np.sqrt(eig_1)/(2.0*np.pi),
                np.sqrt(eig_2)/(2.0*np.pi),
            ]
        )
        return np.sort(frequencies)

    @staticmethod
    def stage_faulty_inertia_matrix(stage, fault_val, planet_index=0):
        """Return a stage inertia matrix with one inertial component reduced.

        ``fault_val`` is a fraction. A value of ``0.2`` removes 20 percent of
        the selected inertial term. For parallel stages the wheel inertia is
        reduced, matching the MATLAB helper. For planetary stages one planet is
        selected explicitly with zero-based ``planet_index``.
        """
        if not 0.0 <= fault_val <= 1.0:
            raise ValueError("fault_val must be a fraction between 0 and 1")

        inertia = Kahraman_94.stage_inertia_matrix(stage)
        if stage.configuration == "parallel":
            inertia[0, 0] *= 1.0 - fault_val
        elif stage.configuration == "planetary":
            if not 0 <= planet_index < stage.N_p:
                raise ValueError("planet_index is out of range for the stage")
            inertia[1 + planet_index, 1 + planet_index] *= 1.0 - fault_val
        else:
            raise ValueError(f"Unsupported stage configuration: {stage.configuration}")
        return inertia

    @staticmethod
    def fault_stiffness_matrix(stage, fault_type, planet_index=0):
        """Return the unscaled stiffness contribution removed by a fault."""
        fault_type = Kahraman_94._normalize_fault_type(fault_type)
        if fault_type == "sun":
            return Kahraman_94.sun_fault_stiffness_matrix(stage)
        if fault_type == "planet":
            return Kahraman_94.planet_fault_stiffness_matrix(stage, planet_index)
        if fault_type == "ring":
            return Kahraman_94.ring_fault_stiffness_matrix(stage)
        if fault_type == "parallel":
            if stage.configuration != "parallel":
                raise ValueError("parallel fault requires a parallel stage")
            return Kahraman_94.stage_stiffness_matrix(stage)
        if fault_type == "mass":
            raise ValueError("mass faults affect inertia, not stiffness")
        raise ValueError("empty fault_type has no stiffness contribution")

    @staticmethod
    def sun_fault_stiffness_matrix(stage):
        """Return the sun-planet mesh stiffness contribution for all planets."""
        Kahraman_94._require_planetary_stage(stage)
        n = stage.N_p + 3
        stiffness = np.zeros((n, n))

        k_sun = stage.sub_set("sun-planet").k_mesh
        r_c = stage.a_w*1.0e-3
        r_s = stage.d[0]*1.0e-3/2.0
        r_p = stage.d[1]*1.0e-3/2.0
        sun = n - 2

        stiffness[0, 0] = stage.N_p*k_sun*r_c**2
        stiffness[0, sun] = -stage.N_p*r_s*k_sun*r_c
        stiffness[sun, 0] = stiffness[0, sun]
        stiffness[sun, sun] = stage.N_p*k_sun*r_s**2

        for planet in range(1, sun):
            stiffness[0, planet] = -r_c*r_p*k_sun
            stiffness[planet, 0] = stiffness[0, planet]
            stiffness[planet, planet] = k_sun*r_p**2
            stiffness[sun, planet] = r_s*r_p*k_sun
            stiffness[planet, sun] = stiffness[sun, planet]
        return stiffness

    @staticmethod
    def planet_fault_stiffness_matrix(stage, planet_index=0):
        """Return one planet's sun-planet mesh stiffness contribution."""
        Kahraman_94._require_planetary_stage(stage)
        if not 0 <= planet_index < stage.N_p:
            raise ValueError("planet_index is out of range for the stage")
        n = stage.N_p + 3
        stiffness = np.zeros((n, n))

        k_sun = stage.sub_set("sun-planet").k_mesh
        r_c = stage.a_w*1.0e-3
        r_s = stage.d[0]*1.0e-3/2.0
        r_p = stage.d[1]*1.0e-3/2.0
        planet = 1 + planet_index
        sun = n - 2

        stiffness[0, 0] = k_sun*r_c**2
        stiffness[0, planet] = -r_c*r_p*k_sun
        stiffness[planet, 0] = stiffness[0, planet]
        stiffness[0, sun] = -r_s*k_sun*r_c
        stiffness[sun, 0] = stiffness[0, sun]
        stiffness[planet, planet] = k_sun*r_p**2
        stiffness[sun, planet] = r_s*r_p*k_sun
        stiffness[planet, sun] = stiffness[sun, planet]
        stiffness[sun, sun] = k_sun*r_s**2
        return stiffness

    @staticmethod
    def ring_fault_stiffness_matrix(stage):
        """Return the ring-planet mesh stiffness contribution for all planets."""
        Kahraman_94._require_planetary_stage(stage)
        n = stage.N_p + 3
        stiffness = np.zeros((n, n))

        k_ring = stage.sub_set("planet-ring").k_mesh
        r_c = stage.a_w*1.0e-3
        r_p = stage.d[1]*1.0e-3/2.0
        sun = n - 2

        stiffness[0, 0] = stage.N_p*k_ring*r_c**2
        for planet in range(1, sun):
            stiffness[0, planet] = r_c*r_p*k_ring
            stiffness[planet, 0] = stiffness[0, planet]
            stiffness[planet, planet] = k_ring*r_p**2
        return stiffness

    @staticmethod
    def _require_planetary_stage(stage):
        """Fail early when a planetary-only fault helper receives another stage."""
        if stage.configuration != "planetary":
            raise ValueError("fault helper requires a planetary stage")

    @staticmethod
    def stage_inertia_matrix(stage):
        """Return the reduced Kahraman stage inertia matrix.

        DOF order follows the module documentation. Planet and gear inertias are
        represented as mass times pitch-radius squared, matching the formulation
        used for fixed-ring analytical validation.
        """
        
        if(stage.configuration == 'parallel'):
            r_p = stage.d[0]*1.0e-3/2.0
            r_w = stage.d[1]*1.0e-3/2.0
            J_p = stage.mass[0]*r_p**2
            J_w = stage.mass[1]*r_w**2
            
            M = np.diag([J_w, J_p, 0.0])
            
        elif(stage.configuration == 'planetary'):
            r_c = stage.a_w*1.0e-3
            r_s = stage.d[0]*1.0e-3/2.0
            r_p = stage.d[1]*1.0e-3/2.0
            J_c = stage.carrier.mass*r_c**2
            J_s = stage.mass[0]*r_s**2
            J_p = stage.mass[1]*r_p**2
            
            d = [J_c]
            [d.append(J_p) for i in range(stage.N_p)]
            d.append(J_s)
            d.append(0.0)
            
            M = np.diag(d)
        
        M[-2:, -2:] += stage.output_shaft.inertia_matrix('torsional')
        
        return M

    __stage_inertia_matrix = stage_inertia_matrix

    def __stiffness_matrix(self):
        
        DT = self.drivetrain
        N = self.n_DOF
                
        K = np.zeros((N[-1], N[-1]))

        i = 0
        sub_range = slice(N[i], N[i + 1])
        K[sub_range, 
          sub_range] += DT.main_shaft.stiffness_matrix('torsional')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 1, N[i + 2])
            stage_stiffness = Kahraman_94.stage_stiffness_matrix(DT.stage[i])
            if self._stage_has_fault(i) and self.fault_type != "mass":
                stage_stiffness = stage_stiffness - self._fault_value_for_stage(
                    i
                )*Kahraman_94.fault_stiffness_matrix(
                    DT.stage[i],
                    self.fault_type,
                    planet_index=self.fault_planet,
                )
            K[sub_range, 
              sub_range] += stage_stiffness

        return K

    def damping_matrix(self):
        DT = self.drivetrain
        N = self.n_DOF
        D = np.zeros((N[-1], N[-1]))

        sub_range = slice(N[0], N[1])
        D[sub_range, sub_range] += DT.main_shaft.damping_matrix("torsional")

        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 1, N[i + 2])
            D[sub_range, sub_range] += Kahraman_94.stage_damping_matrix(DT.stage[i])

        return D

    def centripetal_force_vector(self):
        return np.zeros(self.n_DOF[-1])

    def external_load_vector(self):
        load = np.zeros((self.n_DOF[-1], 2))
        load[0, 0] = 1.0
        load[-1, 1] = 1.0
        return load

    @staticmethod
    def _normalize_fault_type(fault_type):
        fault_type = "" if fault_type is None else str(fault_type).lower()
        aliases = {
            "": "",
            "none": "",
            "m": "mass",
            "mass": "mass",
            "inertia": "mass",
            "sun": "sun",
            "planet": "planet",
            "ring": "ring",
            "parallel": "parallel",
        }
        if fault_type not in aliases:
            raise ValueError(
                "fault_type must be one of '', 'mass', 'sun', 'planet', 'ring', or 'parallel'"
            )
        return aliases[fault_type]

    def _stage_has_fault(self, stage_index):
        if not self.has_fault:
            return False
        if self.fault_stage is None:
            return stage_index == 0
        return stage_index == self.fault_stage

    def _fault_value_for_stage(self, stage_index):
        values = np.atleast_1d(np.asarray(self.fault_val, dtype=float))
        if values.size == 1:
            return values[0]
        return values[stage_index]
    
    @staticmethod
    def stage_damping_matrix(stage, mesh_damping=500.0e6):
        """Return the reduced Kahraman stage damping matrix."""
        if stage.configuration == "parallel":
            N = 3
            D = np.zeros((N, N))
            r_p = stage.d[0]*1.0e-3/2.0
            r_w = stage.d[1]*1.0e-3/2.0
            D[0:2, 0:2] = mesh_damping*np.array(
                [[r_w**2, r_p*r_w], [r_p*r_w, r_p**2]]
            )
        elif stage.configuration == "planetary":
            N = stage.N_p + 3
            D = np.zeros((N, N))
            d_1 = mesh_damping
            d_2 = mesh_damping
            r_c = stage.a_w*1.0e-3
            r_s = stage.d[0]*1.0e-3/2.0
            r_p = stage.d[1]*1.0e-3/2.0

            D[0, 0] = stage.N_p*(d_1 + d_2)*r_c**2
            D[0, N - 2] = -stage.N_p*r_s*d_2*r_c
            D[N - 2, 0] = D[0, N - 2]
            D[N - 2, N - 2] = stage.N_p*d_2*r_s**2

            for i in range(1, N - 2):
                D[0, i] = r_c*r_p*(d_1 - d_2)
                D[i, 0] = D[0, i]
                D[i, i] = (d_1 + d_2)*r_p**2
                D[N - 2, i] = r_s*r_p*d_2
                D[i, N - 2] = D[N - 2, i]
        else:
            raise ValueError(f"Unsupported stage configuration: {stage.configuration}")

        D[-2:, -2:] += stage.output_shaft.damping_matrix("torsional")
        return D

    @staticmethod
    def stage_stiffness_matrix(stage):
        """Return the reduced Kahraman stage stiffness matrix."""
        if(stage.configuration == 'parallel'):
            N = 3
            K = np.zeros((N, N))

            r_p = stage.d[0]*1.0e-3/2.0
            r_w = stage.d[1]*1.0e-3/2.0
            
            k = stage.k_mesh
            
            K[0:2, 0:2] = k*np.array([[    r_w**2, r_p*r_w],
                                   [r_p*r_w   , r_p**2]])
            
        elif(stage.configuration == 'planetary'):
            N = stage.N_p + 3
            K = np.zeros((N, N))
            
            k_1 = stage.sub_set('planet-ring').k_mesh
            k_2 = stage.sub_set('sun-planet').k_mesh

            r_c = stage.a_w*1.0e-3
            r_s = stage.d[0]*1.0e-3/2.0
            r_p = stage.d[1]*1.0e-3/2.0
            
            d = [stage.N_p*r_c*(k_1 + k_2)]
            [d.append((k_1 + k_2)*r_p**2) for i in range(stage.N_p)]
            d.append(stage.N_p*k_2*r_s**2)
            d.append(0.0)

            pla_lin     = np.ones(stage.N_p + 1)*r_c*r_p*(k_1 - k_2)
            pla_lin[-1] = -3.0*k_2*r_s*r_c
            pla_col     = np.ones(stage.N_p    )*k_2*r_p*r_s
            
            i = stage.N_p + 1
            i1 = i + 1
            K[0, 1:i1] = pla_lin
            K[1:i, -2] = pla_col 
            K += K.T
            K += np.diag(d)

        K[-2:, -2:] += stage.output_shaft.stiffness_matrix('torsional')

        return K

    __stage_stiffness_matrix = stage_stiffness_matrix

###############################################################################
