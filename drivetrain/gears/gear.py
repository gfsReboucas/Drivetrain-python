# -*- coding: utf-8 -*-
"""Cylindrical involute gear geometry."""

from math import ceil, floor

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from ..components.materials import Material
from ..components.racks import Rack


class Gear(Rack):
    '''
    This class implements some of the geometric concepts and parameters for 
    cylindrical gears with involute helicoid tooth flanks. It also implements 
    the concepts and parameters for cylindrical gear pairs with parallel axes 
    and a constant gear ratio.
        
    References:
        [1] ISO 21771:2007 Gears -- Cylindrical involute gears and gear pairs
        -- Concepts and geometry
        
        [2] ISO 1328-1:1995 Cylindrical gears -- ISO system of accuracy -- 
        Part 1: Definitions and allowable values of deviations relevant to
        corresponding flanks of gear teeth
        
        [3] ISO 1122-1: Vocabulary of gear terms -- Part 1: Definitions
        related to geometry.
     
    Some basic definitions:
        - Driving gear: that gear of a gear pair which turns the other.
        - Driven gear: that gear of a gear pair which is turned by the other.
        - Pinion: that gear of a pair which has the smaller number of teeth.
        - Wheel: that gear of a pair which has the larger number of teeth.
        - Gear ratio: quotient of the number of teeth of the wheel divided by 
        the number of teeth of the pinion.
        - Transmission ratio: quotient of the angular speed of the first
        driving gear divided by the angular speed of the last driven gear of 
        a gear train.
        - (Right/Left)-handed teeth: teeth whose sucessive transverse 
        profiles show (clockwise/anti-clockwise) displacement with increasing
        distance from an observer looking along the straight line generators
        of the reference surface.
            
    written by:
        Geraldo Rebouças
        - gfs.reboucas@gmail.com
        - https://gfsreboucas.github.io
    '''
    
    def __init__(self, **kwargs):
            
        m_n        = kwargs['m_n']        if('m_n'        in kwargs) else 1.0
        alpha_n    = kwargs['alpha_n']    if('alpha_n'    in kwargs) else 20.0
        rack_type  = kwargs['rack_type']  if('rack_type'  in kwargs) else 'A'
        
        # calling the parent constructor:
        super().__init__(type = rack_type, m = m_n, alpha_P = alpha_n)
        
        # redefining the normal module and pressure angle to be according to 
        # the notation in [1, 3]:
        # [mm],   Normal module:
        self.m_n        = self.m       
        # [deg.], Pressure angle at ref. cylinder
        self.alpha_n    = self.alpha_P 
        
        # main attributes:
        # [-],    Number of teeth:
        self.z          = kwargs['z']          if('z'          in kwargs) else 13
        # [mm],   Face width:
        self.b          = kwargs['b']          if('b'          in kwargs) else 13.0
        # [-],    Profile shift coefficient:
        self.x          = kwargs['x']          if('x'          in kwargs) else 0.0
        # [deg.], Helix angle at reference cylinder:
        self.beta       = kwargs['beta']       if('beta'       in kwargs) else 0.0
        # [-],    Ratio between bore and reference diameters:
        self.bore_ratio = kwargs['bore_ratio'] if('bore_ratio' in kwargs) else 0.5
        # [-],    Tip alteration coefficient:
        self.k          = kwargs['k']          if('k'          in kwargs) else 0.0
        
        # Secondary attributes:
        # [mm],     Transverse module:
        self.m_t     = self.m_n/np.cos(np.radians(self.beta))
        # [deg.],   Transverse pressure angle:
        self.alpha_t = np.degrees(np.arctan(np.tan(np.radians(self.alpha_n))/np.cos(np.radians(self.beta))))
        # [deg.],   Base helix angle:
        self.beta_b  = np.degrees(np.arctan(np.tan(np.radians(self.beta))*np.cos(np.radians(self.alpha_t))))
        # [mm],     Transverse pitch:
        self.p_t     = np.pi*self.m_t
        # [mm],     Transverse base pitch:
        self.p_bt    = self.p_t*np.cos(np.radians(self.alpha_t))
        # [mm],     Transverse base pitch on the path of contact:
        self.p_et    = self.p_bt
        # [mm],     Tooth depth:
        self.h       = self.h_aP + self.k*self.m_n + self.h_fP
        # [mm],     Reference diameter:
        self.d       = np.abs(self.z)*self.m_t
        # [mm],     Tip diameter:
        self.d_a     = self.d + 2.0*np.sign(self.z)*(self.x*self.m_n + self.h_aP + self.k*self.m_n)
        # [mm],     Base diameter:
        self.d_b     = self.d*np.cos(np.radians(self.alpha_t))
        # [mm],     Root diameter:
        self.d_f     = self.d - 2.0*np.sign(self.z)*(self.h_fP - self.x*self.m_n)
        # [mm],     Mean tooth diameter:
        self.d_m     = (self.d_a + self.d_f)/2.0
        # [mm],     Bore diameter:
        self.d_bore  = self.bore_ratio*self.d
        # [-],      Virtual number of teeth:
        self.z_n     = self.z/(np.cos(np.radians(self.beta))*np.cos(np.radians(self.beta_b))**2)
        
        r_out = (self.d_a + self.d_f)/4.0
        r_ins = self.d_bore/2.0
        
        # [m**3],    Volume:
        self.V       = self.b*np.pi*(r_out**2 - r_ins**2)*1.0e-9
        # [kg],      Mass:
        self.mass    = Material().rho*self.V
        # [kg-m**2], Mass moment of inertia (x axis, rot.):
        self.J_x     = (self.mass/2.0)*(r_out**2 + r_ins**2)*1.0e-6
        # [kg-m**2], Mass moment of inertia (y axis):
        self.J_y     = (self.mass/12.0)*(3.0*(r_out**2 + r_ins**2) + self.b**2)*1.0e-6
        # [kg-m**2], Mass moment of inertia (z axis):
        self.J_z     = self.J_y
        
    def __repr__(self):
        val = ('Normal module,                                m_n     = {:6.3f} mm\n'.format(self.m_n) +
               'Pressure angle,                               alpha_n = {:6.3f} deg.\n'.format(self.alpha_n) +
               'Number of teeth,                              z       = {:6.3f} -\n'.format(self.z) +
               'Face width,                                   b       = {:6.3f} mm\n'.format(self.b) +
               'Profile shift coefficient,                    x       = {:6.3f} -\n'.format(self.x) +
               'Helix angle,                                  beta    = {:6.3f} deg.\n'.format(self.beta) +
               'Tip alteration coefficient,                   k       = {:6.3f} -\n'.format(self.k))
        
        return val

    def rectangle(self, C = (0, 0), color = 'r'):
       
        ax = plt.gca()
        rect = Rectangle(C, self.b, self.d, color = color, edgecolor = 'k', \
                         linestyle = '-', facecolor = color)
        
        ax.add_patch(rect)
        
    def d_w(self, alpha_wt):
        '''
        Working pitch diameter, [mm]
        '''
        return self.d_b/np.cos(np.radians(alpha_wt))
    
    @staticmethod
    def __interval_calc(interval, x):
        '''
        Credits for the original version of this method go to:
            [1] E. M. F. Donéstevez, “Python library for design of spur and 
            helical gears transmissions.” Zenodo, 09-Feb-2020, 
            doi: 10.5281/ZENODO.3660527.
        
        It was modified to account for the case where x is an array.

        Parameters
        ----------
        interval : list
            DESCRIPTION.
        x : float or list
            DESCRIPTION.

        Returns
        -------
        val: float or list
            DESCRIPTION.

        '''
        
        flag = False
        if(not isinstance(x, (list, np.ndarray))):
            x = [x]
            flag = True
        
        val = np.zeros(len(x))
        
        for idx in range(len(x)):
            jdx = 1
            while(jdx < len(interval)):
                if(interval[jdx - 1] <= x[idx] <= interval[jdx]):
                    val[idx] = np.sqrt(interval[jdx - 1]*interval[jdx])
                    break
                jdx += 1
        
        if(flag):
            return val[0]
        else:
            return val
