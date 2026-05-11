# -*- coding: utf-8 -*-
"""NREL 5 MW reference drivetrain model."""

import numpy as np

from ..components.bearings import Bearing
from ..components.configs import DrivetrainConfig
from ..components.shafts import Shaft
from ..Gear import GearSet
from ..dynamic_formulation import torsional_2DOF
from .drivetrain import Drivetrain


class NREL_5MW(Drivetrain):
    '''
    This class contains some of the properties of the NREL 5MW wind turbine 
    gearbox proposed by Nejad et. al. [1]. More information about the 
    NREL 5MW wind turbine can be found at [2].
    
    References:
        [1] Nejad, A. R., Guo, Y., Gao, Z., Moan, T. (2016). Development of a
        5 MW reference gearbox for offshore wind turbines. Wind Energy. 
        https://doi.org/10.1002/we.1884
        
        [2] Jonkman, J., Butterfield, S., Musial, W., & Scott, G. Definition
        of a 5-MW Reference Wind Turbine for Offshore System Development. 
        https://doi.org/10.2172/947422
        
        [3] Anaya-Lara, O., Tande, J.O., Uhlen, K., Merz, K. and Nejad, A.R.
        (2019). Modelling and Analysis of Drivetrains in Offshore Wind
        Turbines. In Offshore Wind Energy Technology (eds O. Anaya-Lara, J.O.
        Tande, K. Uhlen and K. Merz). 
        https://doi.org/10.1002/9781119097808.ch3
    '''
    
    def __init__(self, config: DrivetrainConfig | None = None, **kwargs):
        if config is not None:
            if config.stage is not None:
                kwargs.setdefault("stage", config.stage)
            if config.main_shaft is not None:
                kwargs.setdefault("main_shaft", config.main_shaft)
            kwargs.setdefault("N_st", config.N_st)
            kwargs.setdefault("P_rated", config.P_rated)
            kwargs.setdefault("n_rated", config.n_rated)
            kwargs.setdefault("m_Rotor", config.m_Rotor)
            kwargs.setdefault("J_Rotor", config.J_Rotor)
            kwargs.setdefault("m_Gen", config.m_Gen)
            kwargs.setdefault("J_Gen", config.J_Gen)
            kwargs.setdefault("dynamic_model", config.dynamic_model)
        name = ['m_n1', 'b_1', 'd_1', 'L_1',  # Stage 01
                'm_n2', 'b_2', 'd_2', 'L_2',  # Stage 02
                'm_n3', 'b_3', 'd_3', 'L_3',  # Stage 03
                'm_R' , 'J_R' ,               # Rotor: mass and M. M. Inertia
                'm_G' , 'J_G' ,               # Generator: mass and M. M. Inertia
                               'd_s' , 'L_s'] # Main shaft
        
        # [-], Scaling factors:
        self.gamma = kwargs['gamma']     if('gamma'     in kwargs) else dict.fromkeys(name, 1.0)
        # [-], Rated power scaling factor:
        gamma_P    = kwargs['gamma_P']   if('gamma_P'   in kwargs) else 1.0
        # [-], Rated rotor speed scaling factor:
        gamma_n    = kwargs['gamma_n']   if('gamma_n'   in kwargs) else 1.0
        # [-], Dynamic model:
        # Use the simplest completed model as the default import smoke path.
        # Higher-order formulations are still part of the dynamics roadmap.
        dyn_mod    = kwargs['dynamic_model'] if('dynamic_model' in kwargs) else torsional_2DOF
        
        p_r = kwargs['P_rated'] if('P_rated' in kwargs) else 5.0e3*gamma_P
        n_r = kwargs['n_rated'] if('n_rated' in kwargs) else 12.1*gamma_n
        
        if('m_Rotor' in kwargs):
            m_R = kwargs['m_Rotor']
        else:
            m_R = 110.0e3
            m_R *= self.gamma['m_R'] if('m_R' in self.gamma) else 1.0
        if('J_Rotor' in kwargs):
            J_R = kwargs['J_Rotor']
        else:
            J_R = 57231535.0
            J_R *= self.gamma['J_R'] if('J_R' in self.gamma) else 1.0
        if('m_Gen' in kwargs):
            m_G = kwargs['m_Gen']
        else:
            m_G = 1900.0
            m_G *= self.gamma['m_G'] if('m_G' in self.gamma) else 1.0
        if('J_Gen' in kwargs):
            J_G = kwargs['J_Gen']
        else:
            J_G = 534.116
            J_G *= self.gamma['J_G'] if('J_G' in self.gamma) else 1.0

        if('stage' in kwargs):
            stage = kwargs['stage']
        else:
            stage = [None]*3
            for idx in range(3):
                gm_idx = dict(filter(lambda item: str(idx + 1) in item[0], self.gamma.items()))
                stage[idx] = NREL_5MW.gear_set(idx).apply_lambda(gm_idx)
        
        if('main_shaft' in kwargs):
            main_shaft = kwargs['main_shaft']
        else:
            inp_shaft = NREL_5MW.shaft(-1)
            
            d_s = inp_shaft.d
            d_s *= self.gamma['d_s'] if('d_s' in self.gamma) else 1.0
            L_s = inp_shaft.L
            L_s *= self.gamma['L_s'] if('L_s' in self.gamma) else 1.0
            main_shaft = Shaft(d_s, L_s)
        
        n_st = kwargs['N_st'] if('N_st' in kwargs) else 3
        
        super().__init__(P_rated       = p_r,
                         n_rated       = n_r,
                         stage         = stage,
                         main_shaft    = main_shaft,
                         m_Rotor       = m_R,
                         J_Rotor       = J_R,
                         m_Gen         = m_G,
                         J_Gen         = J_G,
                         N_st          = n_st,
                         dynamic_model = dyn_mod)

    def save(self, filename):
        pass
            # file = open(filename, 'w')
            # dump({'P_rated': self.P_rated}, file)
            # dump({'n_rated': self.n_rated}, file)
            # dump(self.gamma,   file)
        
    @staticmethod
    def gear_set(idx):
        '''
        returns the gear stages of the NREL 5 MW wind turbine drivetrain 
        according to [4], Table V. The values for the tip alteration 
        coefficients k, were taken from KISSsoft [1] reports.
        
        References
        ----------
        [1] https://www.kisssoft.ch/

        Parameters
        ----------
        idx : int
            DESCRIPTION.

        Raises
        ------
        Exception
            if idx is out of range.

        Returns
        -------
        GearSet
            the idx-th gear stage.

        '''
        
        alphaN = 20.0  # [deg.], Pressure angle (at reference cylinder)
        rack_type = 'A' # [-],    Type of the basic rack from A to D
        
        if(idx == 0):
            conf = 'planetary' # [-],    GearSet configuration
            Np   =   3         # [-],    Number of planet gears
            mn   =  45.0       # [mm],   Normal module
            beta =   0.0       # [deg.], Helix angle (at reference cylinder)
            bb   = 491.0       # [mm],   Face width
            aw   = 863.0       # [mm],   Center distance
            z_s  =  19         # [-],    Number of teeth (sun)
            z_p  =  17         # [-],    Number of teeth (planet)
            z_r  =  56         # [-],    Number of teeth (ring)
            x_s  =   0.617     # [-],    Profile shift coefficient (sun)
            x_p  =   0.802     # [-],    Profile shift coefficient (planet)
            x_r  =  -0.501     # [-],    Profile shift coefficient (ring)
            k_s  = -10.861/mn  # [-],    Tip alteration coefficient (sun)
            k_p  = -10.861/mn  # [-],    Tip alteration coefficient (planet)
            k_r  =   0.0       # [-],    Tip alteration coefficient (ring)
            
            bore_Rs = 80.0/171.0
            bore_Rp = 80.0/153.0
            bore_Rr = 1.2
            
            zz = np.array([z_s, z_p, z_r])
            xx = np.array([x_s, x_p, x_r])
            kk = np.array([k_s, k_p, k_r])
            bore_R = np.array([bore_Rs, bore_Rp, bore_Rr])
            
        elif(idx == 1):
            conf = 'planetary' # [-],    GearSet configuration
            Np   =   3         # [-],    Number of planet gears
            mn   =  21.0       # [mm],   Normal module
            beta =   0.0       # [deg.], Helix angle (at reference cylinder)
            bb   = 550.0       # [mm],   Face width
            aw   = 584.0       # [mm],   Center distance
            z_s  =  18         # [-],    Number of teeth (sun)
            z_p  =  36         # [-],    Number of teeth (planet)
            z_r  =  93         # [-],    Number of teeth (ring)
            x_s  =   0.389     # [-],    Profile shift coefficient (sun)
            x_p  =   0.504     # [-],    Profile shift coefficient (planet)
            x_r  =   0.117     # [-],    Profile shift coefficient (ring)
            k_s  =  -1.75/mn   # [-],    Tip alteration coefficient (sun)
            k_p  =  -1.75/mn   # [-],    Tip alteration coefficient (planet)
            k_r  =   0.0       # [-],    Tip alteration coefficient (ring)
            
            bore_Rs = 100.0/189.0
            bore_Rp = 95.0/189.0
            bore_Rr = 1.2

            zz = np.array([z_s, z_p, z_r])
            xx = np.array([x_s, x_p, x_r])
            kk = np.array([k_s, k_p, k_r])
            bore_R = np.array([bore_Rs, bore_Rp, bore_Rr])
            
        elif(idx == 2):
            conf = 'parallel' # [-],    GearSet configuration
            Np   =   1        # [-],    Number of planet gears
            mn   =  14.0      # [mm],   Normal module
            beta =  10.0      # [deg.], Helix angle (at reference cylinder)
            bb   = 360.0      # [mm],   Face width
            aw   = 861.0      # [mm],   Center distance
            z_1  =  24        # [-],    Number of teeth (pinion)
            z_2  =  95        # [-],    Number of teeth (wheel)
            x_1  =   0.480    # [-],    Profile shift coefficient (pinion)
            x_2  =   0.669    # [-],    Profile shift coefficient (wheel)
            k_1  =  -0.938/mn # [-],    Tip alteration coefficient (pinion)
            k_2  =  -0.938/mn # [-],    Tip alteration coefficient (wheel)
            
            bore_R1 = 1809.0/3086.0
            bore_R2 = 3385.0/9143.0

            zz = np.array([z_1, z_2])
            xx = np.array([x_1, x_2])
            kk = np.array([k_1, k_2])
            bore_R = np.array([bore_R1, bore_R2])
            
        else:
            raise Exception('Option [{}] is NOT valid.'.format(idx))
        
        bearing = NREL_5MW.bearing(idx)
        shaft = NREL_5MW.shaft(idx)
        
        return GearSet(configuration = conf,
                       m_n = mn,
                       alpha_n = alphaN,
                       z = zz,
                       b = bb,
                       x = xx,
                       beta = beta,
                       k = kk,
                       bore_ratio = bore_R,
                       N_p = Np,
                       a_w = aw,
                       rack_type = rack_type,
                       bearing = bearing,
                       shaft = shaft)

    @staticmethod
    def bearing(idx):
        
        damping = np.array([453.0 , 42000.0, 30600.0, 0.0, 34.3 , 47.8 ])
        
        if(idx == -1): # main shaft
            #                      x,      y,       z,       a,    b,      g
            brg = Bearing(np.array([[0.0   , 1.50e10, 1.50e10, 0.0,  5.0e6,  5.0e6],     # stiffness
                                 [4.06e8, 1.54e10, 1.54e10, 0.0,  0.0  ,  0.0  ]]).T,
                          np.array([[0.0   , 42000.0, 30600.0, 0.0, 34.3  , 47.8  ],    # damping
                                 damping]).T,
                          name =       ['INP-A', 'INP-B'], 
                          type =       [ 'CARB',   'SRB'], 
                          OD   = np.array([ 1750.0,  1220.0]), 
                          ID   = np.array([ 1250.0,   750.0]), 
                          B    = np.array([  375.0,   365.0]))
        elif(idx == 0): # stage 1
            #                      x,     y,     z,     a,   b,     g
            brg = Bearing(np.array([[9.1e4, 9.4e9, 3.2e9, 0.0, 1.4e6, 4.5e6],
                                 [9.1e4, 9.4e9, 3.2e9, 0.0, 1.4e6, 4.5e6],
                                 [6.6e4, 1.7e9, 1.1e9, 0.0, 5.6e5, 1.3e5],
                                 [6.6e7, 1.7e9, 1.1e9, 0.0, 5.6e5, 1.3e5]]).T, # stiffness
                          np.array([damping,
                                 damping,
                                 damping,
                                 damping]).T, # damping
                          name =       ['PL-A', 'PL-B', 'PLC-A', 'PLC-B'], 
                          type =       [ 'CRB',  'CRB',   'SRB',   'CRB'],
                          OD   = np.array([ 600.0,  600.0,  1030.0,  1220.0]),
                          ID   = np.array([ 400.0,  400.0,   710.0,  1000.0]),
                          B    = np.array([ 272.0,  272.0,   315.0,   128.0]))
        elif(idx == 1): # stage 2
            #                      x,     y,     z,     a,   b,     g
            brg = Bearing(np.array([[9.1e4, 6.0e7, 1.2e9, 0.0, 7.5e4, 7.5e4],
                                 [9.1e4, 6.0e7, 1.2e9, 0.0, 7.5e4, 7.5e4],
                                 [9.1e4, 6.0e7, 1.2e9, 0.0, 7.5e4, 7.5e4],
                                 [9.1e7, 6.0e7, 1.2e9, 0.0, 7.5e4, 7.5e4]]).T, # stiffness
                          np.array([damping,
                                 damping,
                                 damping,
                                 damping]).T, # damping
                          name =       ['IMS-PL-A', 'IMS-PL-B', 'IMS-PLC-A', 'IMS-PLC-B'], 
                          type =       [     'CRB',      'CRB',      'CARB',       'CRB'],
                          OD   = np.array([     520.0,      520.0,      1030.0,       870.0]),
                          ID   = np.array([     380.0,      380.0,       710.0,       600.0]),
                          B    = np.array([     140.0,      140.0,       236.0,       155.0]))
        elif(idx == 2): # stage 3
            #                      x,     y,     z,     a,   b,     g
            brg = Bearing(np.array([[0.0,   6.0e7, 1.2e9, 0.0, 7.5e4, 7.5e4],
                                 [7.4e7, 5.0e8, 5.0e8, 0.0, 1.6e6, 1.8e6],
                                 [7.8e7, 7.4e8, 3.3e8, 0.0, 1.1e6, 2.5e6],
                                 [1.3e8, 8.2e8, 8.2e8, 0.0, 1.7e5, 1.0e6],
                                 [6.7e7, 8.0e8, 1.3e8, 0.0, 1.7e5, 1.0e6],
                                 [8.0e7, 1.0e9, 7.3e7, 0.0, 1.7e5, 1.0e6]]).T, # stiffness
                          np.array([damping,
                                 damping,
                                 damping,
                                 damping,
                                 damping,
                                 damping]).T, # damping
                          name =       ['IMS-A', 'IMS-B', 'IMS-C', 'HS-A', 'HS-B', 'HS-C'], 
                          type =       [  'CRB',   'TRB',   'TRB',  'CRB',  'TRB',  'TRB'],
                          OD   = np.array([  360.0,   460.0,   460.0,  500.0,  550.0,  550.0]),
                          ID   = np.array([  200.0,   200.0,   200.0,  400.0,  410.0,  410.0]),
                          B    = np.array([   98.0,   100.0,   100.0,  100.0,   86.0,   86.0]))
        else:
            raise Exception('Option [{}] is NOT valid.'.format(idx))
    
        return brg
    
    @staticmethod
    def shaft(idx):
        if(idx == -1): # Main shaft, Low-speed shaft, LSS
            d = 700.0  # 1000.0
            L = 2000.0 # 3000.0
        elif(idx == 0): # stage 1, Intermediate-speed shaft, ISS
            d = 533.0 # 800.0
            L = 500.0 # 750.0
        elif(idx == 1): # stage 2, High-speed Intermediate-speed shaft, HSIS
            d = 333.0 # 500.0
            L = 666.0 # 1000.0
        elif(idx == 2): # stage 3, High-speed shaft, HSS
            d = 333.0
            L = 1000.0
        else:
            raise Exception('Option [{}] is NOT valid.'.format(idx))
    
        return Shaft(d, L)
        
    @staticmethod
    def property_estimation(idx):
        return 0.0


if __name__ == '__main__':
    DT = NREL_5MW()
    print(DT.f_n)
