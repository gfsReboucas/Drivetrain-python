# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:04:56 2020

@author: geraldod
"""
from numpy import pi, sin, cos, array, diag, argsort, sqrt, iscomplex, real, zeros, zeros_like, eye, ones, allclose, argmax, hstack, vstack
from scipy.linalg import eigh, cholesky, inv, block_diag
# import Drivetrain

class model:
    def __init__(self, dtrain):
        self.drivetrain = dtrain # Drivetrain()
        
        # self.x = 0
        self.M = 0
        self.K = 0
        # self.F = 0
        
        # self.n_DOF = 0
        
        # self.f_n = 0
        # self.mode_shape = 0
        
    def modal_analysis(self):
        
        # Cholesky decomposition:
        L = cholesky(self.M, lower = True)
        
        # Mass normalized stiffness matrix:
        K_tilde = inv(L) @ self.K @ inv(L.T)
       
        if(not allclose(K_tilde, K_tilde.T)):
            print('Matrix is NOT symmetric, but it should be.')
            K_tilde = (K_tilde + K_tilde.T)/2
        
        eig_val, mode_shape = eigh(K_tilde)
        
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

###############################################################################

class torsional_2DOF(model):
    def __init__(self, dtrain):
        super().__init__(dtrain)
        self.n_DOF = 2

        self.M = self.__inertia_matrix()
        self.K = self.__stiffness_matrix()
        
        modA = self.modal_analysis()
        self.f_n = modA['f_n']
        self.mode_shape = modA['mode_shape']
        
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

###############################################################################
class Kahraman_94(model):
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

class Lin_Parker_99(model):
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
          sub_range] += DT.main_shaft.inertia_matrix('Lin_Parker_99')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 3, N[i + 2])
            M[sub_range, 
              sub_range] += Lin_Parker_99.__stage_inertia_matrix(DT.stage[i])
    
        return M
    
    @staticmethod
    def __stage_inertia_matrix(stage):

        M_ = lambda m, J: [m, m, J]

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

        # flatten list:   
        d = [item for sublist in d for item in sublist]
        M = diag(d)
        
        M[-6:, -6:] += stage.output_shaft.inertia_matrix('Lin_Parker_99')
        
        return M
        
    def __stiffness_matrix(self):
        DT = self.drivetrain
        
        N = self.n_DOF
        
        K = zeros((N[-1], N[-1]))

        i = 0
        sub_range = slice(N[i], N[i + 1])
        K[sub_range, 
          sub_range] += DT.main_shaft.stiffness_matrix('Lin_Parker_99')
        
        for i in range(DT.N_st):
            sub_range = slice(N[i + 1] - 3, N[i + 2])
            K[sub_range, 
              sub_range] += Lin_Parker_99.__stage_stiffness_matrix(DT.stage[i])
    
        return K

    @staticmethod
    def __stage_stiffness_matrix(stage):

        # Bearing stiffness sub-matrix:
        K_b_ = lambda x, y: diag([x, y, 0])
        
        alpha_n = stage.alpha_n

        psi   = lambda i: (i - 1)*(2*pi/stage.N_p)
        psi_s = lambda i: psi(i) - alpha_n
        # psi_r = lambda i: psi(i) + alpha_n

        K_s1 = lambda k, i: k*array([[               sin(psi_s(i))**2, -cos(psi_s(i))*sin(psi_s(i)), -sin(psi_s(i))],
                                     [-cos(psi_s(i))*sin(psi_s(i))   ,  cos(psi_s(i))**2           ,  cos(psi_s(i))],
                                     [-              sin(psi_s(i))   ,  cos(psi_s(i))              ,         1     ]])
        K_s2 = lambda k, i: k*array([[ sin(psi_s(i))*sin(alpha_n),  sin(psi_s(i))*cos(alpha_n), -sin(psi_s(i))],
                                     [-cos(psi_s(i))*sin(alpha_n), -cos(psi_s(i))*cos(alpha_n),  cos(psi_s(i))],
                                     [-              sin(alpha_n), -              cos(alpha_n),         1     ]])
        K_s3 = lambda k   : k*array([[ sin(alpha_n)**2          ,  sin(alpha_n)*cos(alpha_n), -sin(alpha_n)],
                                     [ sin(alpha_n)*cos(alpha_n),  cos(alpha_n)**2          , -cos(alpha_n)],
                                     [-sin(alpha_n)             , -cos(alpha_n)             ,        1     ]])

        K_r3 = lambda k   : k*array([[ sin(alpha_n)**2          , -sin(alpha_n)*cos(alpha_n), -sin(alpha_n)],
                                     [-sin(alpha_n)*cos(alpha_n),  cos(alpha_n)**2          ,  cos(alpha_n)],
                                     [-sin(alpha_n)             ,  cos(alpha_n)             ,        1     ]])

        K_c1 = lambda k, i: k*array([[ 1          , 0          , -sin(psi(i))],
                                     [ 0          , 1          ,  cos(psi(i))],
                                     [-sin(psi(i)), cos(psi(i)),        1    ]])
        K_c2 = lambda k, i: k*array([[-cos(psi(i)),  sin(psi(i)), 0],
                                     [-sin(psi(i)), -cos(psi(i)), 0],
                                     [ 0          , -1          , 0]])
        K_c3 = lambda x, y: K_b_(x, y)
        
        if(stage.configuration == 'parallel'):
            # Bearing component:
            b_p = stage.bearing[:2]
            b_w = stage.bearing[3:]
            
            b_p  = b_p.parallel_association()
            k_px = b_p.k_y
            k_py = b_p.k_z

            K_b = block_diag(zeros((3, 3)), K_b_(k_px, k_py))

            # Mesh component:
            b_w  = b_w.parallel_association()
            k_wx = b_w.k_y
            k_wy = b_w.k_z

            k = stage.k_mesh
            K_m = array([[K_s3(k) + K_c3(k_wx, k_wy), K_s2(k, 1)],
                         [K_s2(k, 1)                , K_s1(k, 1)]])

            # Centripetal component:
            K_Omega = block_diag(K_b_(stage.mass[1], stage.mass[1]),
                                 K_b_(stage.mass[0], stage.mass[0]))
        elif(stage.configuration == 'planetary'):
            # Bearing component:
            b_c = stage.bearing[2:]
            b_c = b_c.parallel_association()

            k_cx = b_c.k_y
            k_cy = b_c.k_z

            K_cb = K_b_(k_cx, k_cy)

            K_sb = K_b_(0, 0)

            np = 3*stage.N_p
            K_b = block_diag(K_cb, zeros((np, np)), K_sb)

            # Mesh component:
            K_c = [K_c2(k, i) for i in range(1, stage.N_p + 1)]
            K_c = hstack(K_c)
            K_s = [K_s2(k, i) for i in range(1, stage.N_p + 1)]
            K_s = vstack(K_s)

            K_m = zeros_like(K_b)
            
            k_sp = stage.sub_set('sun-planet').k_mesh
            k_pr = stage.sub_set('planet-ring').k_mesh
            K_pp = K_c3(k_cx, k_cy) + K_r3(k_pr) + K_s3(k_sp)

            b_p = stage.bearing[:2]
            b_p = b_c.parallel_association()

            k_px = b_p.k_y
            k_py = b_p.k_z

            sum_Kc = 0
            sum_Ks = 0
            for i in range(stage.N_p):
                sum_Kc += K_c1( 1  , i + 1)
                sum_Ks += K_s1(k_sp, i + 1)

            sum_Kc = sum_Kc @ K_b_(k_px, k_py)

            d = [sum_Kc]
            [d.append(K_pp) for i in range(stage.N_p)]
            d.append(sum_Ks)

            K_m[ :2     ,  3:np + 3] = K_c
            K_m[3:np + 3, -3:      ] = K_s
            K_m += K_m.T
            K_m += diag(*d)

            pass

        K = lambda Om: K_b + K_m - K_Omega*Om**2
        return K

###############################################################################
if(__name__ == '__main__'):
    pass
