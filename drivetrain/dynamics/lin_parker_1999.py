"""Lin and Parker 1999 drivetrain dynamic formulations.

References:
    [1] J. Lin and R. G. Parker, "Analytical Characterization of the Unique
        Properties of Planetary Gear Free Vibration", Journal of Vibration and
        Acoustics, vol. 121, no. 3, pp. 316-321, 1999.
        https://doi.org/10.1115/1.2893982
    [2] J. Lin, "Analytical investigation of planetary gear dynamics", Ph.D.
        thesis, Ohio State University, 2000.
        http://rave.ohiolink.edu/etdc/view?acc_num=osu1488203552779634
    [3] C. G. Cooley and R. G. Parker, "Vibration Properties of High-Speed
        Planetary Gears With Gyroscopic Effects", Journal of Vibration and
        Acoustics, vol. 134, no. 6, Dec. 2012.
        https://doi.org/10.1115/1.4006646
"""

import numpy as np
import scipy.linalg as la

from .base import model


class Lin_Parker_99(model):
    def __init__(self, dtrain, include_shafts=True):
        super().__init__(dtrain)
        self.include_shafts = include_shafts
        
        # number of DOFs for each stage:
        self.n_DOF = self.__calc_NDOF()
        
        inertia = self.__inertia_matrix()
        self.M = inertia['M']
        self.G = inertia['G']

        stiff = self.__stiffness_matrix()

        self.K_b     = stiff['K_b']
        self.K_m     = stiff['K_m']
        self.K_Omega = stiff['K_Omega']

        self.K       = self.K_b + self.K_m

        modA = self.modal_analysis()
        self.f_n = modA['f_n']
        self.mode_shape = modA['mode_shape']

    def __calc_NDOF(self):
        stage = self.drivetrain.stage
        Np = [0, 6]
        
        for i in range(len(stage)):
            Np.append(Np[-1] + sum([(stage[i].N_p + 1)*3 if(stage[i].configuration == 'parallel')
                                    else (stage[i].N_p + 2)*3]))
        
        return Np

    def __inertia_matrix(self):
        DT = self.drivetrain
        
        m_R = DT.m_Rotor
        J_R = DT.J_Rotor

        m_G = DT.m_Gen
        J_G = DT.J_Gen

        N = self.n_DOF
        
        M = np.zeros((N[-1], N[-1]))
        G = np.zeros_like(M)
        M[:3,  :3 ] = np.diag([m_R, m_R, J_R]) # Rotor inertia matrix
        M[-3:, -3:] = np.diag([m_G, m_G, J_G]) # Generator inertia matrix

        i = 0
        sub_range = slice(N[i], N[i + 1])
        if self.include_shafts:
            M[sub_range, sub_range] += DT.main_shaft.inertia_matrix('Lin_Parker_99')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 3, N[i + 2])
            M[sub_range, sub_range] += Lin_Parker_99.stage_inertia_matrix(
                DT.stage[i],
                include_output_shaft=self.include_shafts,
            )
            G[sub_range, sub_range] += Lin_Parker_99.stage_gyroscopic_matrix(DT.stage[i])
    
        return {'M': M, 'G': G}
    
    @staticmethod
    def stage_inertia_matrix(stage, include_output_shaft=False):

        M_ = lambda m, J, r: np.diag([m, m, J/(r**2)])

        if(stage.configuration == 'parallel'):
            r_p = stage.d_b[0]*1.0e-3/2
            r_w = stage.d_b[1]*1.0e-3/2

            d = [M_(stage.mass[0], stage.J_x[0], r_p), # pinion
                 M_(stage.mass[1], stage.J_x[1], r_w)] # wheel

        elif(stage.configuration == 'planetary'):
            r_c = stage.a_w*1.0e-3
            r_r = stage.d_b[2]*1.0e-3/2
            r_s = stage.d_b[0]*1.0e-3/2
            r_p = stage.d_b[1]*1.0e-3/2

            m_p = stage.mass[1]
            J_p = stage.J_x[1]
            
            d = [M_(stage.carrier.mass, stage.carrier.J_x, r_c), # carrier
                 M_(stage.mass[2],      stage.J_x[2],      r_r), # ring
                 M_(stage.mass[0],      stage.J_x[0],      r_s)] # sun
            [d.append(M_(m_p, J_p, r_p)) for i in range(stage.N_p)] # planet

        M = la.block_diag(*d)
        R = Lin_Parker_99.stage_coordinate_change(stage)
        M = R.T @ M @ R
        M = la.block_diag(M, np.zeros((3, 3)))
        
        if include_output_shaft:
            M[-6:, -6:] += stage.output_shaft.inertia_matrix('Lin_Parker_99')
        
        return M

    __stage_inertia_matrix = stage_inertia_matrix

    @staticmethod
    def stage_gyroscopic_matrix(stage):
        G_ = lambda m: np.array([[0.0, -2.0*m, 0.0],
                              [2.0*m,  0.0  , 0.0],
                              [0.0  ,  0.0  , 0.0]])

        if(stage.configuration == 'parallel'):
            d = [G_(stage.mass[0]), # pinion
                 G_(stage.mass[1])] # wheel

        elif(stage.configuration == 'planetary'):
            d = [G_(stage.carrier.mass), # carrier
                 G_(stage.mass[2]),      # ring
                 G_(stage.mass[0])]      # sun
            [d.append(G_(stage.mass[1])) for i in range(stage.N_p)] # planet

        G = la.block_diag(*d)
        R = Lin_Parker_99.stage_coordinate_change(stage)
        G = R.T @ G @ R

        return la.block_diag(G, np.zeros((3, 3)))
        
    def __stiffness_matrix(self):
        DT = self.drivetrain
        
        N = self.n_DOF
        
        K_b     = np.zeros((N[-1], N[-1]))
        K_m     = np.zeros_like(K_b)
        K_Omega = np.zeros_like(K_b)

        i = 0
        sub_range = slice(N[i], N[i + 1])
        if self.include_shafts:
            K_b[sub_range, sub_range] += DT.main_shaft.stiffness_matrix('Lin_Parker_99')
        
        for i in range(DT.N_st):
            stiff = Lin_Parker_99.stage_stiffness_matrix(
                DT.stage[i],
                include_output_shaft=self.include_shafts,
            )

            sub_range = slice(N[i + 1] - 3, N[i + 2])
            K_b[    sub_range, sub_range] += stiff['K_b']
            K_m[    sub_range, sub_range] += stiff['K_m']
            K_Omega[sub_range, sub_range] += stiff['K_Omega']
            
        return {'K_b'    : K_b,
                'K_m'    : K_m,
                'K_Omega': K_Omega}

    @staticmethod
    def stage_stiffness_matrix(stage, include_output_shaft=False):

        # Bearing stiffness sub-matrix:
        K_b_ = lambda x, y, u=0: np.diag([x, y, u])
        
        alpha_n = np.radians(stage.alpha_n)

        psi   = lambda i: (i - 1)*(2*np.pi/stage.N_p)
        psi_s = lambda i: psi(i) - alpha_n
        psi_r = lambda i: psi(i) + alpha_n
        # ring-ring mesh-stiffness matrix:
        K_r1 = lambda k, i: k*np.array([[ np.sin(psi_r(i))**2, -np.sin(psi_r(i))*np.cos(psi_r(i)), -np.sin(psi_r(i))],
                                     [-np.sin(psi_r(i))*np.cos(psi_r(i)),  np.cos(psi_r(i))**2,  np.cos(psi_r(i))],
                                     [-np.sin(psi_r(i)),  np.cos(psi_r(i)), 1]])
        # ring-planet mesh-stiffness matrix:
        K_r2 = lambda k, i: k*np.array([[-np.sin(psi_r(i))*np.sin(alpha_n),  np.sin(psi_r(i))*np.cos(alpha_n),  np.sin(psi_r(i))],
                                     [ np.cos(psi_r(i))*np.sin(alpha_n), -np.cos(psi_r(i))*np.cos(alpha_n), -np.cos(psi_r(i))],
                                     [ np.sin(alpha_n), -np.cos(alpha_n), -1]])
        # sun-sun mesh-stiffness matrix:
        K_s1 = lambda k, i: k*np.array([[               np.sin(psi_s(i))**2, -np.cos(psi_s(i))*np.sin(psi_s(i)), -np.sin(psi_s(i))],
                                     [-np.cos(psi_s(i))*np.sin(psi_s(i))   ,  np.cos(psi_s(i))**2           ,  np.cos(psi_s(i))],
                                     [-              np.sin(psi_s(i))   ,  np.cos(psi_s(i))              ,         1     ]])
        # sun-planet mesh-stiffness matrix:
        K_s2 = lambda k, i: k*np.array([[ np.sin(psi_s(i))*np.sin(alpha_n),  np.sin(psi_s(i))*np.cos(alpha_n), -np.sin(psi_s(i))],
                                     [-np.cos(psi_s(i))*np.sin(alpha_n), -np.cos(psi_s(i))*np.cos(alpha_n),  np.cos(psi_s(i))],
                                     [-              np.sin(alpha_n), -              np.cos(alpha_n),         1     ]])
        # planet-planet [?] mesh-stiffness matrix:
        K_s3 = lambda k   : k*np.array([[ np.sin(alpha_n)**2          ,  np.sin(alpha_n)*np.cos(alpha_n), -np.sin(alpha_n)],
                                     [ np.sin(alpha_n)*np.cos(alpha_n),  np.cos(alpha_n)**2          , -np.cos(alpha_n)],
                                     [-np.sin(alpha_n)             , -np.cos(alpha_n)             ,        1     ]])
        # [?]
        K_r3 = lambda k   : k*np.array([[ np.sin(alpha_n)**2          , -np.sin(alpha_n)*np.cos(alpha_n), -np.sin(alpha_n)],
                                     [-np.sin(alpha_n)*np.cos(alpha_n),  np.cos(alpha_n)**2          ,  np.cos(alpha_n)],
                                     [-np.sin(alpha_n)             ,  np.cos(alpha_n)             ,        1     ]])
        # carrier-planet bearing stiffness matrix:
        K_c2 = lambda k, i: k*np.array([[-np.cos(psi(i)),  np.sin(psi(i)), 0],
                                     [-np.sin(psi(i)), -np.cos(psi(i)), 0],
                                     [ 0          , -1          , 0]])
        # [?]
        K_c3 = lambda x, y: K_b_(x, y)
        
        Z3 = np.zeros((3, 3))
        if(stage.configuration == 'parallel'):
            r_p = stage.d_b[0]*1.0e-3/2
            r_w = stage.d_b[1]*1.0e-3/2

            # Bearing component:
            b_p  = stage.bearing[3:]
            b_p  = b_p.parallel_association()
            k_px = b_p.k_y
            k_py = b_p.k_z
            k_pu = b_p.k_alpha/(r_p**2)

            K_b = la.block_diag(Z3,                # pinion
                             K_b_(k_px, k_py, k_pu)) # wheel

            # Mesh component:
            b_w  = stage.bearing[:3]
            b_w  = b_w.parallel_association()
            k_wx = b_w.k_y
            k_wy = b_w.k_z
            k_wu = b_w.k_alpha/(r_w**2)

            k = stage.k_mesh
            mesh_coupling = K_s2(k, 1)
            K_m = np.block([[K_s3(k) + K_b_(k_wx, k_wy, k_wu), mesh_coupling.T],
                         [mesh_coupling                , K_s1(k, 1)]])
            
            # Centripetal component:
            K_Omega = la.block_diag(K_b_(stage.mass[0], stage.mass[0]), # pinion
                                 K_b_(stage.mass[1], stage.mass[1])) # wheel

        elif(stage.configuration == 'planetary'):
            n_planet_dof = 3*stage.N_p
            r_c = stage.a_w*1.0e-3
            r_s = stage.d_b[0]*1.0e-3/2
            r_p = stage.d_b[1]*1.0e-3/2

            # Bearing component:
            b_c = stage.bearing[2:4]
            b_c = b_c.parallel_association()

            k_cx = b_c.k_y
            k_cy = b_c.k_z
            k_cu = b_c.k_alpha/(r_c**2)

            K_b = la.block_diag(K_b_(k_cx, k_cy, k_cu), # carrier
                             Z3,                     # ring
                             Z3,                     # sun
                             np.zeros((n_planet_dof, n_planet_dof))) # planet

            # Mesh component:
            k_sp = stage.sub_set('sun-planet').k_mesh
            k_pr = stage.sub_set('planet-ring').k_mesh

            b_p = stage.bearing[:2]
            b_p = b_p.parallel_association()

            k_px = b_p.k_y
            k_py = b_p.k_z
            k_pu = b_p.k_alpha/(r_p**2)

            planet_bearing = K_b_(k_px, k_py, k_pu)
            K_c_row = np.zeros((3, n_planet_dof))
            K_r_row = np.zeros_like(K_c_row)
            K_s_row = np.zeros_like(K_c_row)
            
            sum_Kc = np.zeros((3, 3))
            for i in range(stage.N_p):
                dofs = slice(3*i, 3*(i + 1))
                carrier_planet_coupling = K_c2(1, i + 1)
                # Use C @ D @ C.T so anisotropic planet bearing stiffness
                # remains positive semidefinite. This reduces to the MATLAB
                # shortcut for isotropic bearing stiffness.
                K_c_row[:, dofs] = carrier_planet_coupling @ planet_bearing
                K_r_row[:, dofs] = K_r2(k_pr, i + 1)
                K_s_row[:, dofs] = K_s2(k_sp, i + 1)
                sum_Kc += carrier_planet_coupling @ planet_bearing @ carrier_planet_coupling.T

            sum_Kr = sum(K_r1(k_pr, i + 1) for i in range(stage.N_p))
            sum_Ks = sum(K_s1(k_sp, i + 1) for i in range(stage.N_p))

            diag_01 = la.block_diag(sum_Kc, sum_Kr, sum_Ks)
            diag_up = np.vstack((K_c_row, K_r_row, K_s_row))

            K_pp = planet_bearing + K_r3(k_pr) + K_s3(k_sp)
            diag_02 = la.block_diag(*[K_pp for i in range(stage.N_p)])
            K_m = np.block([[diag_01,  diag_up],
                         [diag_up.T, diag_02]])

            # Centripetal component:
            d = [     K_b_(stage.carrier.mass, stage.carrier.mass)]                       # carrier
            d.append( K_b_(stage.mass[2],      stage.mass[2]))                            # ring
            d.append( K_b_(stage.mass[0],      stage.mass[0]))                            # sun
            [d.append(K_b_(stage.mass[1],      stage.mass[1])) for i in range(stage.N_p)] # planet

            K_Omega = la.block_diag(*d)

        R = Lin_Parker_99.stage_coordinate_change(stage)
        K_b     = R.T @ K_b     @ R
        K_m     = R.T @ K_m     @ R
        K_Omega = R.T @ K_Omega @ R

        K_b = la.block_diag(K_b, Z3)
        K_m = la.block_diag(K_m, Z3)
        K_Omega = la.block_diag(K_Omega, Z3)

        if include_output_shaft:
            K_b[-6:, -6:] += stage.output_shaft.stiffness_matrix('Lin_Parker_99')

        # removing spurious elements:
        K_b[    abs(K_b)     <= 1.0e-4] = 0.0
        K_m[    abs(K_m)     <= 1.0e-4] = 0.0
        K_Omega[abs(K_Omega) <= 1.0e-4] = 0.0

        return {'K_b'    : K_b,
                'K_m'    : K_m,
                'K_Omega': K_Omega}

    __stage_stiffness_matrix = stage_stiffness_matrix

    @staticmethod
    def stage_coordinate_change(stage):
        """Map assembled rotational stage DOFs to raw mesh coordinates."""
        R_ = lambda r: np.diag([1, 1, r])

        Z3 = np.zeros((3, 3))
        I3 = np.eye(3)

        if(stage.configuration == 'parallel'):
            r_p = stage.d_b[0]*1.0e-3/2
            r_w = stage.d_b[1]*1.0e-3/2

            return np.block([[Z3     , R_(r_p)],
                             [R_(r_w), Z3     ]])

        if(stage.configuration == 'planetary'):
            n_planet_dof = 3*stage.N_p
            raw_size = 9 + n_planet_dof
            assembled_size = 6 + n_planet_dof

            r_c = stage.a_w*1.0e-3
            r_s = stage.d_b[0]*1.0e-3/2
            r_p = stage.d_b[1]*1.0e-3/2

            R = np.zeros((raw_size, assembled_size))
            R[0:3, 0:3] = R_(r_c) # carrier
            R[6:9, -3:] = R_(r_s) # sun

            for i in range(stage.N_p):
                raw = slice(9 + 3*i, 9 + 3*(i + 1))
                assembled = slice(3 + 3*i, 3 + 3*(i + 1))
                R[raw, assembled] = R_(r_p)

            return R

        raise ValueError("Unsupported stage configuration: {}".format(stage.configuration))

    @staticmethod
    def testing():
        class dummy_bearing:
            def __init__(self):
                self.k_x     = 1.0e8
                self.k_y     = 1.0e8
                self.k_alpha = 1.0e9
            
            def parallel_association(self):
                return self

        class dummy_carrier:
            def __init__(self):
                self.mass = 5.43
                self.J_x  = 6.29

        class dummy_stage:
            def __init__(self):
                self.alpha_n = 24.6
                self.a_w  = 176.8/2
                self.mass = np.array([ 0.4,    0.66,   2.35])
                self.J_x  = np.array([ 0.39,   0.61,   3.0 ])
                self.d    = np.array([77.4,  100.3,  275.0 ])
                self.carrier = dummy_carrier()
            
            def sub_set(self, opt):
                val = dummy_stage()
                val.k_mesh = 5.0e8
                return val
        
        stage = dummy_stage()

        tmp = Lin_Parker_99(stage)


        print(stage.sub_set('tmp').k_mesh)
