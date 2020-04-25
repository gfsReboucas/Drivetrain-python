# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:04:56 2020

@author: geraldod
"""
from numpy import pi, sin, cos, argsort, iscomplex, real, nan
from numpy import array, diag, argsort, zeros, zeros_like, eye, ones 
from numpy import allclose, argmax, hstack, vstack, block, repeat, reshape
from numpy.lib.scimath import sqrt
from scipy.linalg import eig, cholesky, inv, block_diag, pinv
# import Drivetrain

class torsional_model:
    def __init__(self, dtrain):
        self.drivetrain = dtrain

        self.n_DOF = self.__calc_NDOF()
        self.M     = self.__inertia_matrix()
        self.K     = self.__stiffness_matrix()
        
        MA = self.modal_analysis()

        self.f_n        = MA['f_n']
        self.mode_shape = MA['mode_shape']
    
    def __calc_NDOF(self):
        return 2

    def __inertia_matrix(self):
        DT = self.drivetrain
        
        J_R = DT.J_Rotor # [kg-m^2], Rotor inertia
        J_G = DT.J_Gen   # [kg-m^2], Generator inertia

        U = DT.u[-1]
        
        M = diag([J_R, J_G*U**2])
        
        return M

    def __stiffness_matrix(self):
        DT = self.drivetrain
        U = DT.u[-1]
        
        k_LSS = DT.main_shaft.stiffness('torsional')
        k_HSS = DT.stage[-1].output_shaft.stiffness('torsional')
            
        k = (k_LSS*k_HSS*U**2)/(k_LSS + k_HSS*U**2)
        
        K = k*array([[ 1.0, -1.0],
                     [-1.0,  1.0]])
        
        return K

    def modal_analysis(self):
        
        eig_val, mode_shape = eig(self.K, self.M, right = True)

        if(not any(iscomplex(eig_val))):
            eig_val = real(eig_val)
        else:
            print('At least one complex eigenvalue detected during the calculation of the symmetric undamped eigenvalue problem.')
        
        # lambda to omega_n:
        omega_n = sqrt(eig_val)
        # omega_n to Hz:
        f_n = omega_n/(2.0*pi)
        
        idx = argsort(f_n)
        f_n = f_n[idx]
        mode_shape = mode_shape[:, idx]
        
        for i in range(len(f_n)):
            j = argmax(abs(mode_shape[:, i]))
            mode_shape[:, i] = mode_shape[:, i]/mode_shape[j, i]

        return {
                'f_n': f_n,
                'mode_shape': mode_shape
                }

    @staticmethod
    def testing():
        class dummy_shaft:
            def __init__(self, k):
                self.k = k
            def stiffness(self, opt):
                return self.k

        class dummy_stage:
            def __init__(self, k):
                self.output_shaft = dummy_shaft(k)

        class dummy:
            def __init__(self):
                self.J_Rotor    = 3
                self.J_Gen      = 5
                self.u          = [7]
                self.k_LSS      = 11
                self.k_HSS      = 13
                self.main_shaft =  dummy_shaft(self.k_LSS)
                self.stage      = [dummy_stage(self.k_HSS)]
        
        DT = dummy()
        test = torsional_model(DT)
        
        # Analytical results:
        k_LSS = DT.k_LSS
        k_HSS = DT.k_HSS
        U     = DT.u[-1]

        k = (k_LSS*k_HSS*U**2)/(k_LSS + k_HSS*U**2)
        f1 = sqrt(k/DT.J_Rotor + k/(DT.J_Gen*U**2))/(2*pi)

        f_ana = array([0.0, f1])
        f_test = sorted(test.f_n)
        if(not allclose(f_test, f_ana)):
            print('ERROR!!!!!!!!')

###############################################################################
class general_Kahraman_94(torsional_model):
    def __init__(self, stage):
        self.stage = stage
        self.n_DOF = self.__calc_NDOF()
        self.M     = self.__inertia_matrix()
        self.K     = self.__stiffness_matrix()
        
        MA = self.modal_analysis()

        self.f_n        = MA['f_n']
        self.mode_shape = MA['mode_shape']
    
    def __calc_NDOF(self):
        return self.stage.N_p + 3

    def __inertia_matrix(self):
        stage = self.stage

        if(stage.configuration == 'parallel'):
            m_p = stage.J_x[0]/(stage.d[0]*1.0e-3/2)**2
            m_w = stage.J_x[1]/(stage.d[1]*1.0e-3/2)**2
            d = array(m_w, m_p)

        elif(stage.configuration == 'planetary'):
            m_c = stage.carrier.J_x/(stage.a_w*1.0e-3)**2
            m_r = stage.J_x[2]/(stage.d[2]*1.0e-3/2)**2
            m_s = stage.J_x[0]/(stage.d[0]*1.0e-3/2)**2
            m_p = stage.J_x[1]/(stage.d[1]*1.0e-3/2)**2

            v1 = array([m_c, m_r, m_s])
            vp = ones(stage.N_p)*m_p
            d  = hstack((v1, vp))

        M = diag(d)

        if(stage.configuration == 'planetary'):
            N_eo = stage.N_p + 3 # number of elements (original)
            N_e  = N_eo - 1 # number of elements (new)

            W = eye(N_e, N_eo)
            for idx in range(N_e - 1):
                c1 = N_e - idx
                c2 = c1 - 1
                W[:, [c1, c2]] = W[:, [c2, c1]]
            
            W = pinv(W)
            M = W.T @ M @ W

        return M

    @staticmethod
    def __stage_inertia_matrix(stage):
        pass

    def __stiffness_matrix(self):
        return general_Kahraman_94.__stage_stiffness_matrix(self.drivetrain.stage)

    @staticmethod
    def __stage_stiffness_matrix(stage):
        if(stage.configuration == 'parallel'):
            pass
        elif(stage.configuration == 'planetary'):
            k_c = 0
            k_s = 0
            k_r = 0 # fixed ring

            n = stage.N_p
            k_1 = stage.sub_set('planet-ring').k_mesh
            k_2 = stage.sub_set('sun-planet').k_mesh

            v = array([ k_1 - k_2,
                       -k_1,
                              k_2])
            
            k_11 = zeros((3, 3))
            k_11[0, 1:] = -n*array([k_1, k_2])
            k_11 += k_11.T
            k_11 += diag([n*(k_1 + k_2) + k_c,
                          n* k_1 +        k_r,
                          n*       k_2  + k_s]) 
            
            k_12 = repeat(v, n).reshape(3, n)

            d = ones(n)*(k_1 + k_2)
            k_22 = diag(d)

            K = block([[k_11  , k_12],
                       [k_12.T, k_22]])
            
            N_eo = n + 3 # number of elements (original)
            N_e  = N_eo - 1 # number of elements (new)

            W = eye(N_e, N_eo)
            for idx in range(N_e - 1):
                c1 = N_e - idx
                c2 = c1 - 1
                W[:, [c1, c2]] = W[:, [c2, c1]]
            
            W = pinv(W)
            K = W.T @ K @ W
        
        return K

    @staticmethod
    def testing():
        class dummy_carrier:
            def __init__(self):
                self.J_x = 7

        class dummy_stage:
            def __init__(self):
                self.configuration = 'planetary'
                self.N_p = 3
                self.J_x = array([3, 5, 7])
                self.d   = array([71, 83, 97])
                self.carrier = dummy_carrier()
                self.a_w = 101

            def sub_set(self, opt):
                val = dummy_stage()
                val.k_mesh = 193
                return val                

        stage = dummy_stage()
        test = general_Kahraman_94(stage)

        # Analytical results for fixed ring:
        n = stage.N_p
        m_p = stage.J_x[1]/(stage.d[1]*1.0e-3/2)**2
        m_c = stage.carrier.J_x/(stage.a_w*1.0e-3)**2
        m_s = stage.J_x[0]/(stage.d[0]*1.0e-3/2)**2

        k_1 = stage.sub_set('planet-ring').k_mesh
        k_2 = stage.sub_set('sun-planet').k_mesh

        Lambda_1 = m_p*m_c*m_s
        Lambda_2 = -(n*k_2*m_p*m_c + (k_1 + k_2)*m_c*m_s + n*(k_1 + k_2)*m_p*m_s)
        Lambda_3 = n*k_1*k_2*(n*m_p + m_c + 4*m_s)

        omega_1 = 0
        omega_2 = sqrt((-Lambda_2 - sqrt(Lambda_2**2 - 4*Lambda_1*Lambda_3))/(2*Lambda_1))/(2*pi)
        omega_3 = sqrt((k_1 + k_2)/m_p)*ones(n - 1)/(2*pi)
        omega_4 = sqrt((-Lambda_2 + sqrt(Lambda_2**2 - 4*Lambda_1*Lambda_3))/(2*Lambda_1))/(2*pi)
        f_ana   = hstack((omega_3, array([omega_1,omega_2, omega_4])))

        f_test = sorted(test.f_n)
        f_ana  = sorted(f_ana)
        if(not allclose(f_test, f_ana)):
            print('ERROR!!!!!!!!')

###############################################################################
class Kahraman_94(torsional_model):
    def __init__(self, dtrain):
        super().__init__(dtrain)
        
        # number of DOFs for each stage:
        self.n_DOF = self.__calc_NDOF()
        
        self.M = self.__inertia_matrix()
        self.K = self.__stiffness_matrix()
        
        modA = self.modal_analysis()
        self.f_n = modA['f_n']
        self.mode_shape = modA['mode_shape']
    
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

        M = zeros((N[-1], N[-1]))

        M[0 , 0 ] = DT.J_Rotor # [kg-m^2], Rotor inertia
        M[-1, -1] = DT.J_Gen   # [kg-m^2], Generator inertia
        
        i = 0
        sub_range = slice(N[i], N[i + 1])
        M[sub_range, 
          sub_range] += DT.main_shaft.inertia_matrix('torsional')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 1, N[i + 2])
            M[sub_range, 
              sub_range] += Kahraman_94.__stage_inertia_matrix(DT.stage[i])
    
        return M

    @staticmethod
    def __stage_inertia_matrix(stage):
        
        if(stage.configuration == 'parallel'):
            J_p = stage.J_x[0]
            J_w = stage.J_x[1]
            
            M = diag([J_w, J_p, 0.0])
            
        elif(stage.configuration == 'planetary'):
            J_c = stage.carrier.J_x
            J_s = stage.J_x[0]
            J_p = stage.J_x[1]
            
            d = [J_c]
            [d.append(J_p) for i in range(stage.N_p)]
            d.append(J_s)
            d.append(0.0)
            
            M = diag(d)
        
        M[-2:, -2:] += stage.output_shaft.inertia_matrix('torsional')
        
        return M

    def __stiffness_matrix(self):
        
        DT = self.drivetrain
        N = self.n_DOF
                
        K = zeros((N[-1], N[-1]))

        i = 0
        sub_range = slice(N[i], N[i + 1])
        K[sub_range, 
          sub_range] += DT.main_shaft.stiffness_matrix('torsional')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 1, N[i + 2])
            K[sub_range, 
              sub_range] += Kahraman_94.__stage_stiffness_matrix(DT.stage[i])

        return K
    
    @staticmethod
    def __stage_stiffness_matrix(stage):
        if(stage.configuration == 'parallel'):
            N = 3
            K = zeros((N, N))

            r_p = stage.d[0]*1.0e-3/2.0
            r_w = stage.d[1]*1.0e-3/2.0
            
            k = stage.k_mesh
            
            K[0:2, 0:2] = k*array([[    r_w**2, r_p*r_w],
                                   [r_p*r_w   , r_p**2]])
            
        elif(stage.configuration == 'planetary'):
            N = stage.N_p + 3
            K = zeros((N, N))
            
            k_1 = stage.sub_set('planet-ring').k_mesh
            k_2 = stage.sub_set('sun-planet').k_mesh

            r_c = stage.a_w*1.0e-3
            r_s = stage.d[0]*1.0e-3/2.0
            r_p = stage.d[1]*1.0e-3/2.0
            
            d = [stage.N_p*r_c*(k_1 + k_2)]
            [d.append((k_1 + k_2)*r_p**2) for i in range(stage.N_p)]
            d.append(stage.N_p*k_2*r_s**2)
            d.append(0.0)

            pla_lin     = ones(stage.N_p + 1)*r_c*r_p*(k_1 - k_2)
            pla_lin[-1] = -3.0*k_2*r_s*r_c
            pla_col     = ones(stage.N_p    )*k_2*r_p*r_s
            
            i = stage.N_p + 1
            i1 = i + 1
            K[0, 1:i1] = pla_lin
            K[1:i, -2] = pla_col 
            K += K.T
            K += diag(d)

        K[-2:, -2:] += stage.output_shaft.stiffness_matrix('torsional')

        return K

###############################################################################
class Lin_Parker_99(torsional_model):
    def __init__(self, dtrain):
        super().__init__(dtrain)

        # self.n_DOF = self.__calc_


class Lin_Parker_99_mod(torsional_model):
    def __init__(self, dtrain):
        super().__init__(dtrain)
        
        # number of DOFs for each stage:
        self.n_DOF = self.__calc_NDOF()
        
        self.M = self.__inertia_matrix()

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
        
        M = zeros((N[-1], N[-1]))
        M[:3,  :3 ] = diag([m_R, m_R, J_R]) # Rotor inertia matrix
        M[-3:, -3:] = diag([m_G, m_G, J_G]) # Generator inertia matrix

        i = 0
        sub_range = slice(N[i], N[i + 1])
        M[sub_range, 
          sub_range] += DT.main_shaft.inertia_matrix('Lin_Parker_99')*0
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 3, N[i + 2])
            M[sub_range, 
              sub_range] += Lin_Parker_99.__stage_inertia_matrix(DT.stage[i])
    
        return M
    
    @staticmethod
    def __stage_inertia_matrix(stage):

        M_ = lambda m, J: diag([m, m, J])

        if(stage.configuration == 'parallel'):

            d = [M_(stage.mass[1], stage.J_x[1]), # wheel
                 M_(stage.mass[0], stage.J_x[0]), # pinion
                 M_(           0 ,           0 )] # output shaft

        elif(stage.configuration == 'planetary'):
            m_p = stage.mass[1]
            J_p = stage.J_x[1]
            
            d = [M_(stage.carrier.mass, stage.carrier.J_x)]    # carrier
            [d.append(M_(m_p, J_p)) for i in range(stage.N_p)] # planet
            d.append( M_(stage.mass[0], stage.J_x[0]))         # sun
            d.append( M_(  0,   0))                            # output shaft

        M = block_diag(*d)
        
        M[-6:, -6:] += stage.output_shaft.inertia_matrix('Lin_Parker_99')*0
        
        return M
        
    def __stiffness_matrix(self):
        DT = self.drivetrain
        
        N = self.n_DOF
        
        K_b     = zeros((N[-1], N[-1]))
        K_m     = zeros_like(K_b)
        K_Omega = zeros_like(K_b)

        i = 0
        sub_range = slice(N[i], N[i + 1])
        K_b[sub_range, 
            sub_range] += DT.main_shaft.stiffness_matrix('Lin_Parker_99')*0
        
        for i in range(DT.N_st):
            stiff = Lin_Parker_99.__stage_stiffness_matrix(DT.stage[i])

            sub_range = slice(N[i + 1] - 3, N[i + 2])
            K_b[    sub_range, sub_range] += stiff['K_b']
            K_m[    sub_range, sub_range] += stiff['K_m']
            K_Omega[sub_range, sub_range] += stiff['K_Omega']
            
        return {'K_b'    : K_b,
                'K_m'    : K_m,
                'K_Omega': K_Omega}

    @staticmethod
    def __stage_stiffness_matrix(stage):

        # Bearing stiffness sub-matrix:
        K_b_ = lambda x, y: diag([x, y, 0])
        
        alpha_n = stage.alpha_n

        psi   = lambda i: (i - 1)*(2*pi/stage.N_p)
        psi_s = lambda i: psi(i) - alpha_n
        # psi_r = lambda i: psi(i) + alpha_n
        # sun-sun mesh-stiffness matrix:
        K_s1 = lambda k, i: k*array([[               sin(psi_s(i))**2, -cos(psi_s(i))*sin(psi_s(i)), -sin(psi_s(i))],
                                     [-cos(psi_s(i))*sin(psi_s(i))   ,  cos(psi_s(i))**2           ,  cos(psi_s(i))],
                                     [-              sin(psi_s(i))   ,  cos(psi_s(i))              ,         1     ]])
        # sun-planet mesh-stiffness matrix:
        K_s2 = lambda k, i: k*array([[ sin(psi_s(i))*sin(alpha_n),  sin(psi_s(i))*cos(alpha_n), -sin(psi_s(i))],
                                     [-cos(psi_s(i))*sin(alpha_n), -cos(psi_s(i))*cos(alpha_n),  cos(psi_s(i))],
                                     [-              sin(alpha_n), -              cos(alpha_n),         1     ]])
        # planet-planet [?] mesh-stiffness matrix:
        K_s3 = lambda k   : k*array([[ sin(alpha_n)**2          ,  sin(alpha_n)*cos(alpha_n), -sin(alpha_n)],
                                     [ sin(alpha_n)*cos(alpha_n),  cos(alpha_n)**2          , -cos(alpha_n)],
                                     [-sin(alpha_n)             , -cos(alpha_n)             ,        1     ]])
        # [?]
        K_r3 = lambda k   : k*array([[ sin(alpha_n)**2          , -sin(alpha_n)*cos(alpha_n), -sin(alpha_n)],
                                     [-sin(alpha_n)*cos(alpha_n),  cos(alpha_n)**2          ,  cos(alpha_n)],
                                     [-sin(alpha_n)             ,  cos(alpha_n)             ,        1     ]])
        # carrier-carrier bearing stiffness matrix:
        K_c1 = lambda k, i: k*array([[ 1          , 0          , -sin(psi(i))],
                                     [ 0          , 1          ,  cos(psi(i))],
                                     [-sin(psi(i)), cos(psi(i)),        1    ]])
        # carrier-planet bearing stiffness matrix:
        K_c2 = lambda k, i: k*array([[-cos(psi(i)),  sin(psi(i)), 0],
                                     [-sin(psi(i)), -cos(psi(i)), 0],
                                     [ 0          , -1          , 0]])
        # [?]
        K_c3 = lambda x, y: K_b_(x, y)
        
        # From torsional to translational coordinates:
        R_ = lambda r: diag([1, 1, r])

        Z3 = zeros((3, 3))
        I3 = eye(3)
        if(stage.configuration == 'parallel'):
            # Bearing component:
            b_p  = stage.bearing[3:]
            b_p  = b_p.parallel_association()
            k_px = b_p.k_y
            k_py = b_p.k_z

            K_b = block_diag(Z3,               # wheel
                             K_b_(k_px, k_py), # pinion
                             Z3)               # shaft

            # Mesh component:
            b_w  = stage.bearing[:3]
            b_w  = b_w.parallel_association()
            k_wx = b_w.k_y
            k_wy = b_w.k_z

            k = stage.k_mesh
            K_m = block([[K_s3(k) + K_c3(k_wx, k_wy), K_s2(k, 1)],
                         [K_s2(k, 1)                , K_s1(k, 1)]])

            K_m = block_diag(K_m, Z3)
            
            # Centripetal component:
            K_Omega = block_diag(K_b_(stage.mass[1], stage.mass[1]), # wheel
                                 K_b_(stage.mass[0], stage.mass[0]), # pinion
                                 Z3)                                 # shaft

            # Torsional to translational:
            r_p = stage.d[1]*1.0e-3/2
            r_w = stage.d[1]*1.0e-3/2

            R = block_diag(R_(r_w), R_(r_p), I3)

        elif(stage.configuration == 'planetary'):
            # Bearing component:
            b_c = stage.bearing[2:]
            b_c = b_c.parallel_association()

            k_cx = b_c.k_y
            k_cy = b_c.k_z

            K_cb = K_b_(k_cx, k_cy)

            K_sb = Z3

            np = 3*stage.N_p
            K_b = block_diag(K_cb,            # carrier
                             zeros((np, np)), # planet
                             K_sb,            # sun
                             Z3)              # shaft

            # Mesh component:
            k_sp = stage.sub_set('sun-planet').k_mesh
            k_pr = stage.sub_set('planet-ring').k_mesh

            b_p = stage.bearing[:2]
            b_p = b_p.parallel_association()

            k_px = b_p.k_y
            k_py = b_p.k_z

            K_c = [K_c2(1, i + 1)*K_b_(k_px, k_py) for i in range(stage.N_p)]
            K_c = hstack(K_c)
            K_s = [K_s2(k_sp, i + 1)               for i in range(stage.N_p)]
            K_s = vstack(K_s)

            K_m = zeros_like(K_b)
            
            K_pp = K_c3(k_cx, k_cy) + K_r3(k_pr) + K_s3(k_sp)

            sum_Kc = 0
            sum_Ks = 0
            for i in range(stage.N_p):
                sum_Kc += K_c1( 1  , i + 1)
                sum_Ks += K_s1(k_sp, i + 1)

            sum_Kc = sum_Kc @ K_b_(k_px, k_py)

            d = [sum_Kc]
            [d.append(K_pp) for i in range(stage.N_p)]
            d.append(sum_Ks)
            d.append(Z3)

            K_m[ :3     ,  3:np + 3] = K_c
            K_m[3:np + 3, -3:      ] = K_s
            K_m += K_m.T
            K_m += block_diag(*d)

            # Centripetal component:
            d = [     K_b_(stage.carrier.mass, stage.carrier.mass)]                       # carrier
            [d.append(K_b_(stage.mass[1],      stage.mass[1])) for i in range(stage.N_p)] # planet
            d.append( K_b_(stage.mass[0],      stage.mass[0]))                            # sun
            d.append(Z3)                                                                  # shaft

            K_Omega = block_diag(*d)

            # Torsional to translational:
            r_s = stage.d[0]*1.0e-3/2
            r_p = stage.d[1]*1.0e-3/2
            r_c = stage.a_w *1.0e-3
            
            d = [R_(r_c)]
            [d.append(R_(r_p)) for i in range(stage.N_p)]
            d.append(R_(r_s))
            d.append(I3)

            R = block_diag(*d)

        # Torsional to translational:
        K_b     = R.T @ K_b     @ R
        K_m     = R.T @ K_m     @ R
        K_Omega = R.T @ K_Omega @ R

        K_b[-6:, -6:] += stage.output_shaft.stiffness_matrix('Lin_Parker_99')*0

        # removing spurious elements:
        K_b[    abs(K_b)     <= 1.0e-4] = 0.0
        K_m[    abs(K_m)     <= 1.0e-4] = 0.0
        K_Omega[abs(K_Omega) <= 1.0e-4] = 0.0

        return {'K_b'    : K_b,
                'K_m'    : K_m,
                'K_Omega': K_Omega}

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
                self.mass = array([ 0.4,    0.66,   2.35])
                self.J_x  = array([ 0.39,   0.61,   3.0 ])
                self.d    = array([77.4,  100.3,  275.0 ])
                self.carrier = dummy_carrier()
            
            def sub_set(self, opt):
                val = dummy_stage()
                val.k_mesh = 5.0e8
                return val
        
        stage = dummy_stage()

        tmp = Lin_Parker_99(stage)


        print(stage.sub_set('tmp').k_mesh)

###############################################################################
if(__name__ == '__main__'):
    general_Kahraman_94.testing()
    pass
