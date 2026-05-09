"""Basic rack tooth profile geometry."""

from dataclasses import dataclass

import numpy as np
from scipy import interpolate

@dataclass
class Rack:
    '''
    Implements some characteristics of the standard basic rack tooth profile 
    for cylindrical involute gears (external or internal) for general and 
    heavy engineering.
    
    References:
        [1] ISO 53:1998 Cylindrical gears for general and heavy engineering 
        -- Standard basic rack tooth profile
        
        [2] ISO 54:1996 Cylindrical gears for general engineering and for 
        heavy engineering -- Modules
       
    written by:
        Geraldo Rebouças
        - gfs.reboucas@gmail.com
        - https://gfsreboucas.github.io
    '''
    
    type: str = 'A'      # [-],    Type of basic rack tooth profile.
    m: float = 1.0       # [mm],   Module.
    alpha_P: float = 20.0 # [deg.], Pressure angle.

    def __post_init__(self):
        self.m = self.module(self.m)

        # secondary attributes:
        if(self.type == 'A'):
            k_c_P    = 0.25
            k_rho_fP = 0.38
        elif(self.type == 'B'):
            k_c_P    = 0.25
            k_rho_fP = 0.30
        elif(self.type == 'C'):
            k_c_P    = 0.25
            k_rho_fP = 0.25
        elif(self.type == 'D'):
            k_c_P    = 0.40
            k_rho_fP = 0.39
        else:
            print('Rack type [{}] is NOT defined.'.format(self.type))
            
        self.c_P   = k_c_P*self.m          # [mm], Bottom clearance
        self.h_fP  = (1.0 + k_c_P)*self.m  # [mm], Dedendum
        self.e_P   = self.m/2.0            # [mm], Spacewidth
        self.h_aP  = self.m                # [mm], Addendum
        self.h_FfP = self.h_fP - self.c_P  # [mm], Straight portion of the dedendum
        self.h_P   = self.h_aP + self.h_fP # [mm], Tooth depth
        self.h_wP  = self.h_P - self.c_P   # [mm], Common depth of rack and tooth
        self.p     = np.pi*self.m             # [mm], Pitch
        self.s_P   = self.e_P              # [mm], Tooth thickness
        self.rho_fP = k_rho_fP*self.m      # [mm], Fillet radius
        
        self.U_FP = 0.0     # [mm],   Size of undercut
        self.alpha_FP = 0.0 # [deg.], Angle of undercut

    def __repr__(self):
        '''
        Return a string containing a printable representation of an object.

        Returns
        -------
        None.

        '''
        val = ('Rack type:                   {}     -\n'.format(self.type) +
               'Module,          m       = {:7.3f} mm\n'.format(self.m) +
               'Pressure angle,  alpha_P = {:7.3f} deg.\n'.format(self.alpha_P) +
               'Addendum,        h_aP    = {:7.3f} mm\n'.format(self.h_aP) +
               'Dedendum,        h_fP    = {:7.3f} mm\n'.format(self.h_fP) +
               'Tooth depth,     h_P     = {:7.3f} mm\n'.format(self.h_P) +
               'Pitch,           p       = {:7.3f} mm\n'.format(self.p) +
               'Tooth thickness, s_P     = {:7.3f} mm\n'.format(self.s_P) +
               'Fillet radius,   rho_fP  = {:7.3f} mm\n'.format(self.rho_fP))
        
        return val
       
    def save(self, filename):
        pass
    
    @staticmethod
    def module(m_x, **kwargs):
        '''
        Returns the values of normal modules for straight and helical gears 
        according to ISO 54:1996 [2]. According to this standard, preference
        should be given to the use of the normal modules as given in series I 
        and the module 6.5 in series II should be avoided.
            
        The module is defined as the quotient between:
            - the pitch, expressed in millimetres, to the number pi, or;
            - the reference diameter, expressed in millimetres, by the number 
            of teeth.
        '''
        option = kwargs['option'] if('option' in kwargs) else 'calc'
        method = kwargs['method'] if('method' in kwargs) else 'nearest'
        
        m_1 = [1.000, 1.250, 1.50, 2.00, 2.50, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0]
        m_2 = [1.125, 1.375, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.0,  9.0, 11.0, 14.0, 18.0, 22.0, 28.0, 36.0, 45.0]
        
        if(option == 'show'):
            print('... to be implemented...')
        elif(option == 'calc'):
            # removing the value 6.5 which should be avoided according to [2]
            # and inserting additional values due to (Nejad et. al., 2015) and
            # (Wang et. al., 2020). See the classes NREL_5MW and DTU_10MW for 
            # detailed references about these works.
            m_2[8] = 21.0 
            m_2.append(30.0)
            x = sorted(m_1 + m_2)
        elif(option == 'calc_1'):
            x = m_1
        elif(option == 'calc_2'):
            x = m_2
        else:
            print('Option [{}] is NOT valid.'.format(option))
        
        idx = interpolate.interp1d(x, list(range(len(x))), kind = method, 
                                   fill_value = (0, len(x) - 1))
        return x[idx(m_x).astype(int)]

    def max_fillet_radius(self):
        ''' 
        returns the maximum fillet radius of the basic rack according to 
        ISO 53:1998 [1], Sec. 5.9.
        '''
        
        if(self.alpha_P != 20.0):
            print('The pressure angle is not 20.0 [deg.]')
        else:
            if((self.c_P <= 0.295*self.m) and (self.h_FfP == self.m)):
                rho_fP_max = self.c_P/(1.0 - np.sin(np.radians(self.alpha_P)))
            elif((0.295*self.m < self.c_P) and (self.c_P <= 0.396*self.m)):
                rho_fP_max = (np.pi*self.m/4.0 - self.h_fP*np.tan(np.radians(self.alpha_P)))/np.tan(np.radians(90.0 - self.alpha_P)/2.0)
            
        return rho_fP_max
