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
    def __init__(self, dtrain):
        super().__init__(dtrain)
        
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
            M[sub_range, 
              sub_range] += Kahraman_94.stage_inertia_matrix(DT.stage[i])
    
        return M

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
            K[sub_range, 
              sub_range] += Kahraman_94.stage_stiffness_matrix(DT.stage[i])

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
