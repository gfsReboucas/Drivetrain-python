# -*- coding: utf-8 -*-
"""ISO 6336 pitting safety-factor calculator."""

import numpy as np

from ..Gear import GearSet
from ..components.materials import Material
from .factors import (
    _contact_ratio_factor,
    _face_load_factor,
    _interp_ZNT,
    _lub_vel_factor,
    _rough_factor,
    _tooth_contact_factor,
    _transv_load_factor,
    _zone_factor,
)


class ISO_6336:
    '''
    References:
        [1] ISO 6336-1:2006 Calculation of load capacity of spur and helical 
        gears -- Part 1: Basic principles, introduction and general influence 
        factors
        
        [2] ISO 6336-2:2006 Calculation of load capacity of spur and helical 
        gears -- Part 2: Calculation of surface durability (pitting)
        
        [3] ISO/TR 6336-30:2017 Calculation of load capacity of spur and 
        helical gears -- Calculation examples for the application of ISO 6336 
        parts 1, 2, 3, 5
        
        [4] Arnaudov, K., Karaivanov, D. (2019). Planetary Gear Trains. Boca 
        Raton: CRC Press, https://doi.org/10.1201/9780429458521
    '''
    def __init__(self, gset, **kwargs):
        
        if(not isinstance(gset, GearSet)):
            raise Exception('Not a GearSet object.')
            
        self.gear_set = gset
        
        Np = gset.N_p
        if(Np == 3):
            k_g = 1.1
        elif(Np == 4):
            k_g = 1.25
        elif(Np == 5):
            k_g = 1.35
        elif(Np == 6):
            k_g = 1.44
        elif(Np == 7):
            k_g = 1.47
        else:
            k_g = 1.0
        
        # Mesh load factor according to [7]:
        self.K_gamma = k_g
        # [-],  Minimum required safety factor for surface durability according to IEC 61400-4:
        self.S_Hmin = kwargs['S_Hmin'] if('S_Hmin' in kwargs) else 1.25
        # [-],  Minimum required safety factor for surface durability according to IEC 61400-4:
        self.S_Fmin = kwargs['S_Fmin'] if('S_Fmin' in kwargs) else 1.56
        # [h],  Required life:
        self.L_h    = kwargs['L_h']    if('L_h'    in kwargs) else 20.0*365.0*24.0
        # [-],  Application factor:
        self.K_A    = kwargs['K_A']    if('K_A'    in kwargs) else 1.25
        
        self.S_H = 0.0
        self.S_F = 0.0
        
    def Pitting(self, **kwargs):
        P      = kwargs['P']      if('P'      in kwargs) else 100.0
        n_1    = kwargs['n_1']    if('n_1'    in kwargs) else 1.0
        # [um], Maximum arithmetic mean roughness for external gears according to [7], Sec. 7.2.7.2.
        R_a    = kwargs['R_a']    if('R_a'    in kwargs) else 0.8
        # [um], 
        R_z    = kwargs['R_z']    if('R_z'    in kwargs) else 6.0*R_a
        # Default ISO viscosity grade: VG 220
        # [mm/s^2],  Nominal kinematic viscosity
        nu_40  = kwargs['nu_40']  if('nu_40'  in kwargs) else 220.0
        line   = kwargs['line']   if('line'   in kwargs) else 4
        
        # [N/mm^2],  Young's modulus:
        E          = Material().E*1.0e-6
        # [-],       Poisson's ratio:
        nu         = Material().nu
        # [N/mm^2],  Allowable contact stress number:
        sigma_Hlim = Material().sigma_Hlim*1.0e-6
        # [kg/mm^3], Density
        rho        = Material().rho*1.0e-9
        # Tip relief by running-in, Table 8, [1]:
        C_ay = (1.0/18.0)*(sigma_Hlim/97.0 - 18.45)**2 + 1.5

        C_a = kwargs['C_a']    if('C_a'    in kwargs) else C_ay
        
        # Preparatory calculations:
        T_1 = (P*1.0e3)/(n_1*np.pi/30.0)
        
        T_1 = abs(T_1)
        n_1 = abs(n_1)
        
        gset = self.gear_set
        
        if(gset.configuration == 'parallel'):
            SH  = self.__calculate_SH(gset, T_1, n_1, C_a, E, nu, rho, 
                                      sigma_Hlim, line, nu_40, R_z)
        elif(gset.configuration == 'planetary'):
            sun_pla = gset.sub_set('sun-planet')
            SH1 = self.__calculate_SH(sun_pla, T_1, n_1, C_a, E, nu, rho, 
                                      sigma_Hlim, line, nu_40, R_z)
            # pla_rng = gset.sub_set('planet-ring')
            # SH2 = self.__calculate_SH(pla_rng, T_1/gset.N_p, n_1, C_a, E, nu, rho, 
            #                           sigma_Hlim, line, nu_40, R_z)
            
            # SH = [SH1, SH2]
            # SH = [item for sublist in SH for item in sublist]
            SH = SH1
        
        return np.array(SH)
    
            
    def Bending(self, **kwargs):
        pass
    
    @staticmethod
    def benchmark():
        from .benchmarks import benchmark

        return benchmark()

    def __calculate_SH(self, gset, T_1, n_1, C_a, E, nu, rho, sigma_Hlim, line, 
                       nu_40, R_z):
        
            f_pb = max(gset.f_pt*np.cos(np.radians(gset.alpha_t)))
            if(f_pb >= 40.0): # [um]
                y_alpha = 3.0 # [um]
            else:
                y_alpha = f_pb*75.0e-3
            
            f_falpha = max(gset.f_falpha)
            # Estimated running allowance (pitch deviation):
            y_p = y_alpha
            # Estimated running allowance (flank deviation):
            y_f = f_falpha*75.0e-3
            f_pbeff = f_pb - y_p
            f_falphaeff = f_falpha - y_f # 0.925*y_f
            
            # Pitch line velocity:
            v = self.__pitch_line_velocity(n_1)
            Z_eps = _contact_ratio_factor(gset)
            # [N], Nominal tangential load:
            F_t = 2.0e3*(T_1/gset.d[0])/self.gear_set.N_p
            
            line_load = F_t*self.K_A*self.K_gamma/np.mean(gset.b)
            
            K_v = self.__dynamic_factor(gset, n_1, v, rho, line_load, C_a,
                                        f_pbeff, f_falphaeff)
            
            val = _face_load_factor(gset)
            K_Hbeta = val['K_Hbeta']
    
            # Determinant tangential load in a transverse plane:
            F_tH = F_t*self.K_gamma*self.K_A*K_v*K_Hbeta
            
            term = gset.c_gamma_alpha*(f_pb - y_alpha)/(F_tH/gset.b)
            val = _transv_load_factor(gset, term, Z_eps)
            K_Halpha = val['K_Halpha']
            
            # Zone factor: (sec. 6.1)
            Z_H = _zone_factor(gset)
            
            # Single pair tooth contact factor (sec. 6.2)
            val = _tooth_contact_factor(gset)
            Z_B = val['Z_B']
            Z_D = val['Z_D']
            
            # Elasticity factor: (sec. 7)
            Z_E = np.sqrt(E/(2.0*np.pi*(1.0 - nu**2)))
            
            # Helix angle factor: (sec. 9)
            Z_beta = 1.0/np.sqrt(np.cos(np.radians(gset.beta)))
    
            val = _lub_vel_factor(sigma_Hlim, nu_40, v)
            Z_L = val['Z_L']
            Z_v = val['Z_v']
    
            # Roughness factor:
            Z_R = _rough_factor(gset, R_z, sigma_Hlim)
    
            # Work hardening factor:
            Z_W = 1.0
    
            # Size factor:
            Z_X = 1.0
    
            # Number of load cycles:
            N_L1 = n_1*60.0*self.L_h # pinion
            N_L2 = N_L1/gset.u       # wheel
            
            # Life factor:
            # line = 2
            Z_NT1 = _interp_ZNT(line, N_L1)
            Z_NT2 = _interp_ZNT(line, N_L2)
    
            # Contact stress:
            # Nominal contact stress at pitch point:
            num = F_t*(gset.u + 1.0)
            den = gset.d[0]*gset.b*gset.u
            sigma_H0 = Z_H*Z_E*Z_eps*Z_beta*np.sqrt(num/den)
       
            # nominal contact stress at pitch point:
            sigma_H1 = Z_B*sigma_H0*np.sqrt(self.K_gamma*self.K_A*K_v*K_Hbeta*K_Halpha) # pinion
            sigma_H2 = Z_D*sigma_H0*np.sqrt(self.K_gamma*self.K_A*K_v*K_Hbeta*K_Halpha) # wheel
            
            # Permissible contact stress:
            sigma_HP1 = sigma_Hlim*Z_NT1*Z_L*Z_v*Z_R*Z_W*Z_X/self.S_Hmin
            sigma_HP2 = sigma_Hlim*Z_NT2*Z_L*Z_v*Z_R*Z_W*Z_X/self.S_Hmin
    
            # Safety factor for surface durability (against pitting):
            S_H1 = sigma_HP1*self.S_Hmin/sigma_H1 # pinion/planet
            S_H2 = sigma_HP2*self.S_Hmin/sigma_H2 # wheel/sun
            
            return [S_H1, S_H2]
    
    def __pitch_line_velocity(self, n):
        
        gs = self.gear_set
        if(gs.configuration == 'parallel'):
            v = (np.pi*n/60.0e3)*gs.d[0]
        elif(gs.configuration == 'planetary'):
            v = (np.pi*n/60.0e3)*(gs.d[0] - gs.a_w/gs.u)
        
        return v

    def __dynamic_factor(self, gset, n_1, v, rho, line_load, C_a, f_pbeff, 
                         f_falphaeff):
        
        z1 = self.gear_set.z[0]
        uu = self.gear_set.u
        cond = (v*z1/100.0)*np.sqrt((uu**2)/(1.0 + uu**2))
        if(cond < 3.0): # [m/s]
            print('Calculating K_v using method B outside of its useful range. ' + 
                  'More info at the end of Sec. 6.3.2 of ISO 6336-1.')
        
        if(self.gear_set.configuration == 'parallel'):
            # Based on Sec. 6.5.9 of ISO 6336-1, Eq. (30), assuming gears:
            # - of solid construction, and
            # - with the same density
            num = np.pi*rho*(gset.u**2)*gset.d_m[0]**4
            den = 8.0*(gset.u**2 + 1.0)*gset.d_b[0]**2
            m_red = num/den

            # Resonance running speed, Eq. (6) [1]:
            n_E1 = 3.0e4*np.sqrt(gset.c_gamma_alpha/m_red)/(np.pi*gset.z[0]) # [1/min]

            # Resonance ratio, Eq. (9) [1]:
            N = n_1/n_E1
            
            K_v = ISO_6336.__dynamic_factor_from_range(gset, line_load, N, C_a, 
                                                       f_pbeff, f_falphaeff)
            
        elif(self.gear_set.configuration == 'planetary'):
            m_sun = rho*(np.pi/8.0)*(self.gear_set.d_m[0]**4)/(self.gear_set.d_b[0]**2) # sun
            m_pla = rho*(np.pi/8.0)*(self.gear_set.d_m[1]**4)/(self.gear_set.d_b[1]**2) # planet
            m_red1 = (m_pla*m_sun)/(m_sun + m_pla*self.gear_set.N_p)
            m_red2 =  m_pla

            # Resonance running speed:
            n_E11 = 3.0e4*np.sqrt(self.gear_set.cprime/m_red1)*1.0/(np.pi*self.gear_set.z[0]) # [1/min]
            n_E12 = 3.0e4*np.sqrt(self.gear_set.cprime/m_red2)*1.0/(np.pi*self.gear_set.z[1]) # [1/min]

            # Resonance ratio:
            N_1 =  n_1/n_E11
            N_2 = (n_1/self.gear_set.u)/n_E12

            K_v1 = ISO_6336.__dynamic_factor_from_range(self.gear_set, line_load, 
                                                        N_1, C_a, f_pbeff, 
                                                        f_falphaeff)
            K_v2 = ISO_6336.__dynamic_factor_from_range(self.gear_set, line_load,
                                                        N_2, C_a, f_pbeff,
                                                        f_falphaeff)
            K_v = max(K_v1, K_v2)
        
        if(K_v < 1.05): # according to [7?], Sec. 7.2.3.2
            K_v = 1.05
            
        return K_v

    @staticmethod
    def __dynamic_factor_from_range(gset, line_load, N, C_a, f_pbeff, f_falphaeff):
        
        #Table 8, [1]:
        C_v1 = 0.32
        C_v5 = 0.47
        
        esp_g = gset.eps_gamma
        if(1.0 < esp_g <= 2.0):
            C_v2 = 0.34
            C_v3 = 0.23
            C_v4 = 0.90
            C_v6 = 0.47
        elif(esp_g > 2.0):
            C_v2 =  0.57 /(esp_g - 0.3)
            C_v3 =  0.096/(esp_g - 1.56)
            C_v4 = (0.57 - 0.05*esp_g)/(esp_g - 1.44)
            C_v6 =  0.12/(esp_g - 1.74)
        
        if(1.0 < esp_g <= 1.5):
            C_v7 = 0.75
        elif(1.5 < esp_g <= 2.5):
            C_v7 = 0.125*np.sin(np.pi*(esp_g - 2.0)) + 0.875
        elif(esp_g > 2.5):
            C_v7 = 1.0
        
        cp = gset.cprime
        B_p = cp*f_pbeff/line_load        # Eq. (15)
        B_f = cp*f_falphaeff/line_load    # Eq. (16)
        B_k = abs(1.0 - cp*C_a/line_load) # Eq. (17)
        
        # Dynamic factor:
        if(line_load < 100): # [N/mm]
            N_S = 0.5 + 0.35*np.sqrt(line_load/100.0) # Eq. (11), [1]
        else:
            N_S = 0.85 # Eq. (12) [1]
        
        if(N <= N_S):
            # Section 6.5.4, Eq. (13), [1]: 
            K = C_v1*B_p + C_v2*B_f + C_v3*B_k 
            K_v = N*K + 1.0 # Eq. (14)
        elif(N_S < N <= 1.15):
            # Section 6.5.5, Eq. (20), [1]: 
            K_v = C_v1*B_p + C_v2*B_f + C_v4*B_k + 1.0
        elif(1.15 < N < 1.5):
            K_vN115 = C_v1*B_p + C_v2*B_f + C_v4*B_k + 1.0
            K_vN15  = C_v5*B_p + C_v6*B_f + C_v7
            # Section 6.5.7, Eq. (22), [1]: 
            K_v = K_vN15 + (K_vN115 - K_vN15)*(1.5 - N)/0.35
        elif(N >= 1.5):
            # Section 6.5.6, Eq. (21), [1]: 
            K_v = C_v5*B_p + C_v6*B_f + C_v7
            
        return K_v
