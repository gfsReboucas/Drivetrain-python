# -*- coding: utf-8 -*-
"""Gear set geometry and derived mesh quantities."""

from math import ceil, floor

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from ..components.bearings import Bearing
from ..components.materials import Material
from ..components.racks import Rack
from ..components.shafts import Shaft
from ..components.utils import scaling_factor
from .carrier import Carrier
from .gear import Gear


class GearSet(Gear):
    '''
    This class implements SOME procedures for the calculation of the load
    capacity of cylindrical involute gears with external or internal teeth. 
    Specifically the calculation of contact stresses for the assessment of the
    surface durability of cylindrical gears. 
    In a planetary gear there are two different gear pairs:
        (1) sun-planet;
        (2) planet-ring;
        
    References:
        [1] ISO 6336-1:2006 Calculation of load capacity of spur and helical 
        gears -- Part 1: Basic principles, introduction and general influence 
        factors
        
        [2] ISO 6336-2:2006 Calculation of load capacity of spur and helical 
        gears -- Part 2: Calculation of surface durability (pitting) 
        
        [3] ISO/TR 6336-30:2017 Calculation of load capacity of spur and 
        helical gears -- Calculation examples for the application of ISO 6336 
        parts 1, 2, 3, 5
        
        [4] Nejad, A. R., Guo, Y., Gao, Z., Moan, T. (2016). Development of a
        5 MW reference gearbox for offshore wind turbines. Wind Energy. 
        https://doi.org/10.1002/we.1884
        
        [5] IEC 61400-4:2012 Wind Turbines -- Part 4: Design Requirements for
        wind turbine gearboxes
        
        [6] ISO 21771:2007 Gears -- Cylindrical involute gears and gear pairs
        -- Concepts and geometry
        
    written by:
        Geraldo Rebouças
        - gfs.reboucas@gmail.com
        - https://gfsreboucas.github.io
    '''
    
    def __init__(self, **kwargs):
        
        m_n        = kwargs['m_n']        if('m_n'        in kwargs) else 1.0
        alpha_n    = kwargs['alpha_n']    if('alpha_n'    in kwargs) else 20.0
        rack_type  = kwargs['rack_type']  if('rack_type'  in kwargs) else 'A'
        z          = kwargs['z']          if('z'          in kwargs) else 13*np.ones(2)
        b          = kwargs['b']          if('b'          in kwargs) else 13.0*np.ones(2)
        x          = kwargs['x']          if('x'          in kwargs) else np.zeros(2)
        beta       = kwargs['beta']       if('beta'       in kwargs) else 0.0
        k          = kwargs['k']          if('k'          in kwargs) else np.zeros(2)
        bore_ratio = kwargs['bore_ratio'] if('bore_ratio' in kwargs) else 0.5*np.ones(2)
        
        if((len(z) != len(x)) and (len(x) != len(k)) and (len(k) != len(bore_ratio))):
            raise Exception('The lengths of z, x, k and bore ratio should be equal.')
            
        if(len(z) < 2):
            raise Exception('There should be at least two gears.')
        elif(len(z) == 3):
            z[2] = -np.abs(z[2]) # because the ring is an internal gear

        # calling the parent constructor:
        super().__init__(m_n = m_n,
                         alpha_n = alpha_n, 
                         rack_type = rack_type,
                         z = z,
                         b = b,
                         x = x,
                         beta = beta,
                         k = k,
                         bore_ratio = bore_ratio)
            
        # Main attributes:
        # [-], Configuration of the gear set (e.g. parallel, planetary);
        self.configuration = kwargs['configuration'] if('configuration' in kwargs) else 'parallel'
        # [-], Number of planets:
        self.N_p           = kwargs['N_p']           if('N_p'           in kwargs) else 1
        # [mm], Center distance:
        self.a_w           = kwargs['a_w']           if('a_w'           in kwargs) else 13.0
        # [-], Bearing array:
        self.bearing       = kwargs['bearing']       if('bearing'       in kwargs) else [Bearing()]*2
        # [-], Output shaft:
        self.output_shaft  = kwargs['shaft']         if('shaft'         in kwargs) else Shaft()
        # [-], ISO accuracy grade:
        self.Q             = kwargs['Q']             if('Q'             in kwargs) else 6.0
        
        r_out = np.zeros(len(self.z))
        r_ins = np.zeros(len(self.z))
        for idx, zz in enumerate(self.z):
            if(zz > 0.0):
                r_out[idx] = (self.d_a[idx] + self.d_f[idx])/4.0
                r_ins[idx] = self.d_bore[idx]/2.0
            else:
                r_ins[idx] = (self.d_a[idx] + self.d_f[idx])/4.0
                r_out[idx] = self.d_bore[idx]/2.0
        
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
        
        # Geometric mean for parameters according to Sec. 5.3 of ISO 1328-1 [2]:
        # inspired on the accepted answer in:
        # https://stackoverflow.com/questions/2236906/first-python-list-index-greater-than-x
        geo_mean = lambda r, x: next(np.sqrt(r[i - 1]*r[i]) 
                                         for i, v in enumerate(sorted(r)) 
                                         if v > x)
        
        range_b = [0.0, 4.0, 10.0, 20.0, 40.0, 80.0, 160.0, 250.0, 400.0, 
                  650.0, 1.0e3]
        range_d = [0.0, 5.0, 20.0, 50.0, 125.0, 280.0, 560.0, 1.0e3, 1.6e3,
                  2.5e3, 4.0e3, 6.0e3, 8.0e3, 10.0e3]
        range_m = [0.0, 0.5, 2.0, 3.5, 6.0, 10.0, 16.0, 25.0, 40.0, 70.0]
        
        b_int =        geo_mean(range_b,               self.b)
        d_int = np.array([geo_mean(range_d, dd) for dd in self.d])
        m_int =        geo_mean(range_m,               self.m_n)
        
        # Single pitch deviation according to Section 6.1 of ISO 1328-1 [2]:
        f_pt = 0.3*(m_int + 0.4*np.sqrt(d_int)) + 4.0
        self.f_pt = self.__round_ISO(f_pt, self.Q)
        # Total cumulative pitch deviation according to Sec. 6.3 of ISO 1328-1 [2]:
        F_p = 0.3*m_int + 1.25*np.sqrt(d_int) + 7.0
        self.F_p = self.__round_ISO(F_p, self.Q)
        # Total profile deviation according to Section 6.4 of ISO 1328-1 [2]:
        F_alpha = 3.2*np.sqrt(m_int) + 0.22*np.sqrt(d_int) + 0.7
        self.F_alpha = self.__round_ISO(F_alpha, self.Q)
        # Total helix deviation according to Section 6.5 of ISO 1328-1 [2]:
        f_beta = 0.1*np.sqrt(d_int) + 0.63*np.sqrt(b_int) + 4.2
        self.f_beta = self.__round_ISO(f_beta, self.Q)
        # Profile form deviation according to App. B.2.1 of ISO 1328-1 [2]:
        f_falpha = 2.5*np.sqrt(m_int) + 0.17*np.sqrt(d_int) + 0.5
        self.f_falpha = self.__round_ISO(f_falpha, self.Q)
        # Profile slope deviation according to App. B.2.2 of ISO 1328-1 [2]:
        f_Halpha = 2.0*np.sqrt(m_int) + 0.14*np.sqrt(d_int) + 0.5
        self.f_Halpha = self.__round_ISO(f_Halpha, self.Q)
        # Helix form deviation according to App. B.2.3 of ISO 1328-1 [2]:
        f_fbeta = (0.07*np.sqrt(d_int) + 0.45*np.sqrt(b_int) + 3.0)
        self.f_fbeta = self.__round_ISO(f_fbeta, self.Q)
        # Helix slope deviation according to App. B.2.3 of ISO 1328-1 [2]:
        self.f_Hbeta = self.f_fbeta
        
        # [-], Gear ratio:
        self.u             = self.__gear_ratio()
        # [deg.], Working transverse pressure angle:
        self.alpha_wt      = self.__working_transverse_pressure_angle()

        # [mm], Root form diameter:
        self.d_Ff = self.__root_form_diameter()
        # Start of active profile diameter:
        self.d_Nf = self.__SAP_diameter()
        # [mm], Active tip diameter:
        self.d_Na = self.__active_tip_diameter()

        # [N/(mm-um)], Theoretical single stiffness:
        self.cprime_th     = self.__theo_single_tooth_stiffness()
        # [N/(mm-um)], Maximum single stiffness of a tooth pair:
        self.cprime        = self.__max_single_tooth_stiffness()
        # [-],         Transverse contact ratio:
        self.eps_alpha     = self.__transverse_contact_ratio()
        # [-],         Overlap ratio:
        self.eps_beta      = np.mean(self.b)*np.sin(np.radians(self.beta))/(np.pi*self.m_n)
        # [-],         Total contact ratio:
        self.eps_gamma     = self.eps_alpha + self.eps_beta
        # [N/(mm-um)], Mean value of mesh stiffness per unit face witdh (used for K_v, K_Halpha, K_Falpha):
        self.c_gamma_alpha = self.cprime*(0.75*self.eps_alpha + 0.25)
        # [N/(mm-um)], Mean value of mesh stiffness per unit face witdh (used for K_Hbeta, K_Fbeta):
        self.c_gamma_beta  = 0.85*self.c_gamma_alpha
        # [N/(mm-um)], Mean value of mesh stiffness per unit face witdh:
        self.c_gamma       = self.c_gamma_alpha + self.c_gamma_beta
        # [N/m],       Mean value of mesh stiffness:
        self.k_mesh        = self.c_gamma*np.mean(self.b)*1.0e6
        
        if(self.configuration == 'planetary'):
            # [-], planet carrier:
            self.carrier = self.__carrier()
            # gear ratio of sun-planet mesh:
            self.u_12    =     self.z[1]/self.z[0]
            # gear ratio of planet-ring mesh:
            self.u_23    = np.abs(self.z[2])/self.z[1]
            # note that: u = 1 + u_12 * u_23
    
    def __repr__(self):
        '''
        produces a formatted output to the GearSet object.

        Returns
        -------
        None.

        '''

        with np.printoptions(precision = 3):
            val = ('Gear ratio,                            u       = {:7.3f} -\n'.format(self.u) +
                   'Number of elements,                    p       = {:7} -\n'.format(self.N_p) +
                   'Normal module,                         m_n     = {:7.3f} mm\n'.format(self.m_n) +
                   'Pressure angle,                        alpha_n = {} deg.\n'.format(self.alpha_n) +
                   'Helix angle,                           beta    = {} deg.\n'.format(self.beta) +
                   'Face width,                            b       = {} mm\n'.format(self.b) +
                   'Center distance,                       a_w     = {} mm\n'.format(self.a_w) +
                   'Number of teeth,                       z       = {} -\n'.format(self.z) +
                   'Profile shift coefficient,             x       = {} -\n'.format(self.x) +
                   'Reference diameter,                    d       = {} mm\n'.format(self.d) +
                   'Mass,                                  m       = {} kg\n'.format(self.mass) +
                   'Mass moment of inertia (x axis, rot.), J_x     = {} kg-m**2\n'.format(self.J_x))
            
        return val
    
    def rectangle(self, C = (0, 0), color = ['r', 'g', 'b', 'c', 'm']):
       
        if(self.configuration == 'parallel'):
            C_p = (C[0]   + self.b/2.0, C[1] + self.a_w)
            C_w = (C_p[0], C[1])
            C_s = (C_p[0] + self.b/2.0 + self.output_shaft.L/2.0, C_p[1])
            
            self.__gear(0).rectangle(C_p,    color[0])
            self.__gear(1).rectangle(C_w,    color[1])
            self.output_shaft.rectangle(C_s, color[4])
        elif(self.configuration == 'planetary'):
            C_c = (C[0] + self.carrier.b/2.0, C[1])
            C_p = (C_c[0], C_c[1] + self.a_w)
            C_s = (C_c[0] + self.carrier.b/2.0 + self.output_shaft.L/2.0, C_c[1])
            
            # self.carrier.rectangle(  C_c, color[3])
            self.__gear(2).rectangle(C_c, color[2])
            self.__gear(0).rectangle(C_c, color[0])
            self.__gear(1).rectangle(C_p, color[1])
            self.output_shaft.rectangle(C_s, color[4])
    
    def apply_lambda(self, gamma):
        mn  = self.m_n*scaling_factor('m_n', gamma)
        bb  = self.b*scaling_factor('b', gamma)
        sha = self.output_shaft.apply_lambda(gamma)
        aw  = self.a_w*(Rack.module(mn)/self.m_n)
        
        return GearSet(m_n = mn, b = bb, a_w = aw, shaft = sha, 
                       configuration = self.configuration,
                       alpha_n       = self.alpha_n,
                       rack_type     = self.type,
                       z             = self.z,
                       x             = self.x,
                       beta          = self.beta,
                       k             = self.k,
                       bore_ratio    = self.bore_ratio,
                       bearing       = self.bearing,
                       N_p           = self.N_p)

    def __gear(self, idx):
        '''
         returns the idx-th gear of the GearSet object.
        '''
        if(0 <= idx < self.N_p):
            val = Gear(m_n        = self.m_n,
                       alpha_n    = self.alpha_n,
                       rack_type  = self.type,
                       z          = self.z[idx], 
                       b          = self.b,
                       x          = self.x[idx],
                       beta       = self.beta,
                       k          = self.k[idx],
                       bore_ratio = self.bore_ratio)
        else:
            raise Exception('idx = {0} is OUT of range(0, {1}).'.format(idx, self.N_p))

        return val
    
    def sub_set(self, option):
        if(not self.configuration == 'planetary'):
            raise Exception('Not a planetary GearSet.')
        
        if(option == 'sun-planet'):
            idx = [0, 1]
            s = 1
        elif(option == 'planet-ring'):
            idx = [1, 2]
            s = -1

        return GearSet(configuration = 'parallel',
                       m_n           =  self.m_n,
                       alpha_n       =  self.alpha_n,
                       rack_type     =  self.type,
                       a_w           =  self.a_w*s,
                       b             =  self.b,
                       z             =  self.z[idx],
                       x             =  self.x[idx],
                       beta          =  self.beta,
                       k             =  self.k[idx],
                       bore_ratio    =  self.bore_ratio[idx],
                       shaft         =  self.output_shaft)

    def set_gear_speed(self, speeds):
        """Return gear speeds from partial known speeds.

        This keeps the structure of MATLAB ``Gear_Set.set_gear_speed``. For
        parallel stages, ``speeds`` is ``[pinion, wheel]`` with one known
        speed. For planetary stages, ``speeds`` is ``[sun, planet, ring,
        carrier]`` with one fixed zero speed and one nonzero input speed;
        unknown values are ``np.nan``. The planetary calculation uses Willis'
        equation and the carrier-frame sun-planet mesh relation.
        """

        speeds = np.array(speeds, dtype=float)

        if(self.configuration == 'parallel'):
            if speeds.shape != (2,):
                raise ValueError("Parallel gear speeds must be [pinion, wheel].")
            idx_zero = np.nan
            U = np.flip(np.diag([1.0/self.u, self.u]), axis=0)
        elif(self.configuration == 'planetary'):
            if speeds.shape != (4,):
                raise ValueError("Planetary gear speeds must be [sun, planet, ring, carrier].")

            zero_indices = np.flatnonzero(speeds == 0.0)
            if zero_indices.size == 0:
                raise ValueError("Planetary speeds must define one fixed zero-speed element.")
            if zero_indices.size > 1:
                raise ValueError("Planetary speeds must define only one fixed zero-speed element.")

            input_indices = np.flatnonzero(~np.isnan(speeds) & (speeds != 0.0))
            if input_indices.size != 1:
                raise ValueError("Gear speeds must define exactly one nonzero input speed.")

            sun_teeth, planet_teeth, ring_teeth = self.z
            ring_teeth = abs(ring_teeth)

            # Willis equation for a simple planetary train:
            #   z_s*w_s + z_r*w_r = (z_s + z_r)*w_c.
            # Planet spin follows the sun-planet mesh in the carrier frame:
            #   w_p = w_c - (z_s/z_p)*(w_s - w_c).
            coefficients = np.array(
                [
                    [sun_teeth, 0.0, ring_teeth, -(sun_teeth + ring_teeth)],
                    [sun_teeth/planet_teeth, 1.0, 0.0, -(1.0 + sun_teeth/planet_teeth)],
                ]
            )
            values = np.zeros(2)

            for index, speed in enumerate(speeds):
                if not np.isnan(speed):
                    row = np.zeros(4)
                    row[index] = 1.0
                    coefficients = np.vstack((coefficients, row))
                    values = np.append(values, speed)

            speeds = np.linalg.solve(coefficients, values)
            idx_zero = int(zero_indices[0])
            input_index = int(input_indices[0])
            return speeds, idx_zero, input_index
        else:
            raise ValueError("Unsupported gear-set configuration: {}".format(self.configuration))

        input_indices = np.flatnonzero(~np.isnan(speeds))
        if input_indices.size != 1:
            raise ValueError("Gear speeds must define exactly one nonzero input speed.")

        nullspace = np.linalg.svd(np.eye(len(U)) - U)[2][-1]
        input_index = int(input_indices[0])
        nullspace = nullspace/nullspace[input_index]
        return nullspace*speeds[input_index], idx_zero, input_index

    def body_speed_ratios(self, reference="input"):
        """Return normalized body speed ratios for gyroscopic scaling.

        The default drivetrain convention is pinion input for parallel stages
        and fixed ring, sun input, carrier output for planetary stages.
        """

        if reference != "input":
            raise ValueError("Only input-referenced speed ratios are supported.")

        if(self.configuration == 'parallel'):
            speeds, _, _ = self.set_gear_speed([1.0, np.nan])
            return {"pinion": speeds[0], "wheel": speeds[1]}

        if(self.configuration == 'planetary'):
            speeds, _, _ = self.set_gear_speed([1.0, np.nan, 0.0, np.nan])
            return {
                "sun": speeds[0],
                "planet": speeds[1],
                "ring": speeds[2],
                "carrier": speeds[3],
            }

        raise ValueError("Unsupported gear-set configuration: {}".format(self.configuration))

    def __carrier(self):
        '''
        Initializes the planet carrier object.

        Raises
        ------
        Exception
            only applicable for planetary GearSet objects.

        Returns
        -------
        val : Carrier

        '''
        if(self.configuration == 'planetary'):
            val = Carrier(self.a_w, self.b)
        else:
            raise Exception('Only Planetary gear sets have a planet carrier.')
        
        return val

    def __gear_ratio(self):
        if(self.configuration == 'parallel'):
            val = np.abs(self.z[1])/self.z[0]
        elif(self.configuration == 'planetary'):
            val = 1.0 + np.abs(self.z[2])/self.z[0]
        else:
            raise Exception('Configuration [{}] is NOT defined.'.format(self.configuration.upper()))
        
        return val
    
    def __theo_single_tooth_stiffness(self):
        '''
        [N/(mm-um)], Theoretical single stiffness:
        '''
        
        C_1 =  0.04723;      C_2 =  0.15551;      C_3 =  0.25791
        C_4 = -0.00635;      C_5 = -0.11654;      C_6 = -0.00193
        C_7 = -0.24188;      C_8 =  0.00529;      C_9 =  0.00182
        
        # q' is the minimum value for the flexibility of a pair of teeth
        qprime = C_1                         + \
                 C_2           /self.z_n[0]  + \
                 C_3           /self.z_n[1]  + \
                 C_4* self.x[0]              + \
                 C_6* self.x[1]              + \
                 C_5*(self.x[0]/self.z_n[0]) + \
                 C_7*(self.x[1]/self.z_n[1]) + \
                 C_8* self.x[0]**2           + \
                 C_9* self.x[1]**2 # [mm-um/N]

        return 1.0/qprime
    
    def __working_transverse_pressure_angle(self):
        '''
        [deg.], Working transverse pressure angle, according to Sec. 5.2.4 of
        [6] ISO 21771.
        '''
        
        num = self.m_n*np.cos(np.radians(self.alpha_t))
        den = 2.0*self.a_w*np.cos(np.radians(self.beta))

        ang = np.degrees(np.arccos(np.abs(sum(self.z[:2]))*num/den))

        if(ang > 90.0):
            ang = 180 - ang
        elif(ang < 0.0):
            ang *= -1.0

        return ang
    
    def __max_single_tooth_stiffness(self):
        '''
        [N/(mm-um)], Maximum single stiffness of a tooth pair:
        '''
        
        # Correction factor for solid disk gears:
        C_M = 0.8
        # Gear blank factor for solid disk gears:
        C_R = 1.0
        
        # Basic rack factor:
        alpha_Pn = self.alpha_n # [deg.], Normal pressure angle of basic rack
        C_B1 = (1.0 + 0.5*(1.2 - self.h_fP/self.m_n))*(1.0 - 0.02*(20.0 - alpha_Pn)) # 0.975
        C_B2 = (1.0 + 0.5*(1.2 - self.h_fP/self.m_n))*(1.0 - 0.02*(20.0 - alpha_Pn))

        C_B = 0.5*(C_B1 + C_B2) # 0.975
        
        return self.cprime_th*C_M*C_R*C_B*np.cos(np.radians(self.beta))
    
    def __transverse_contact_ratio(self):
        '''
        Returns the transverse contact ratio of a GearSet object according to 
        Section 8.3.1 of ISO 6336-2 [2].
        Calculations do not take undercut into account.

        Raises
        ------
        Exception
            If all of the calculated roll angles xi_fw are negative.

        Returns
        -------
        float
            Transverse contact ratio, [-].

        '''

        xi_Nfw1 = np.zeros(3)
        xi_Nfw2 = np.zeros(3)
        
        # roll angles from the root form diameter to the working pitch point, 
        # limited by the:
        # (1) base diameters: Eq. (33)
        xi_Nfw1[0] = np.tan(np.radians(self.alpha_wt))
        xi_Nfw2[0] = xi_Nfw1[0]
        
        # (2) root form diameters: Eq. (34-35)
        xi_Nfw1[1] = xi_Nfw1[0] - np.tan(np.arccos(self.d_b[0]/self.d_Nf[0]))
        xi_Nfw2[1] = xi_Nfw2[0] - np.tan(np.arccos(self.d_b[1]/self.d_Nf[1]))

        # (3) active tip diameters of the wheel/pinion: Eq. (36-37)
        xi_Nfw1[2] = (np.tan(np.arccos(self.d_b[1]/self.d_Na[1])) - xi_Nfw1[0])*self.z[1]/self.z[0]
        xi_Nfw2[2] = (np.tan(np.arccos(self.d_b[0]/self.d_Na[0])) - xi_Nfw2[0])*self.z[0]/self.z[1]

        # xi_Nfw1 = array([i for i in xi_Nfw1 if i > 0])
        # xi_Nfw2 = array([i for i in xi_Nfw2 if i > 0])
        
        # Angles shouldn't be negative:
        if((xi_Nfw1.size == 0) or (xi_Nfw2.size == 0)):
            raise Exception('Auxiliary coefficients xi_Nfw are all negative.')

        xi_Nfw1 = min(xi_Nfw1)
        xi_Nfw2 = min(xi_Nfw2)

        # roll angle from the working pitch point to the active tip diameter: Eq. (38)
        xi_Naw1 = xi_Nfw2*self.z[1]/self.z[0]

        # pinion angular pitch:
        tau_1 = 2.0*np.pi/self.z[0]

        # Eq. (32)
        return (xi_Nfw1 + xi_Naw1)/tau_1

    def __transverse_contact_ratio_v2006(self):
        '''
        Returns the transverse contact ratio of a GearSet object according to 
        Section 8.3.1 of ISO 6336-2:2006 [2].
        Calculations do not take undercut into account.

        Raises
        ------
        Exception
            If all of the calculated roll angles xi_fw are negative.

        Returns
        -------
        float
            Transverse contact ratio, [-].

        '''
        # Sec. A.1, Eq. (A.5)
        B = lambda x: (self.h_fP - x*self.m_n + self.rho_fP*(np.sin(np.radians(self.alpha_n)) - 1.0))
        # Sec. A.3, Eq. (A.10)
        d_soi = lambda i: 2.0*np.sqrt((self.d[i]/2.0 - B(self.x[i]))**2 + (B(self.x[i])/np.tan(np.radians(self.alpha_t)))**2)
        
        xi_fw1 = np.zeros(3)
        xi_fw2 = np.zeros(3)
        
        # roll angles from the root form diameter to the working pitch point, 
        # limited by the:
        # (1) base diameters: Eq. (28)
        xi_fw1[0] = np.tan(np.radians(self.alpha_wt))
        xi_fw2[0] = np.tan(np.radians(self.alpha_wt))
        
        # (2) root form diameters: Eq. (29-30)
        xi_fw1[1] = xi_fw1[0] - np.tan(np.arccos(self.d_b[0]/d_soi(0)))
        xi_fw2[1] = xi_fw2[0] - np.tan(np.arccos(self.d_b[1]/d_soi(1)))

        # (3) tip diameters of the wheel/pinion: Eq. (31-32)
        xi_fw1[2] = (np.tan(np.arccos(self.d_b[1]/self.d_a[1])) - xi_fw1[0])*(self.z[1]/self.z[0])
        xi_fw2[2] = (np.tan(np.arccos(self.d_b[0]/self.d_a[0])) - xi_fw2[0])*(self.z[0]/self.z[1])
       
        # Angles shouldn't be negative:
        xi_fw1 = np.array([i for i in xi_fw1 if i > 0])
        xi_fw2 = np.array([i for i in xi_fw2 if i > 0])
        
        if((xi_fw1.size == 0) or (xi_fw2.size == 0)):
            raise Exception('Auxiliary coefficients xi_fw are all negative.')
        
        xi_fw1 = min(xi_fw1)
        xi_fw2 = min(xi_fw2)
        
        # roll angle from the working pitch point to the tip diameter: Eq. (33)
        xi_aw1 = xi_fw2*self.z[1]/self.z[0]

        # pinion angular pitch: Eq. (34)
        tau_1 = 2.0*np.pi/self.z[0]

        # Eq. (27)
        return (xi_fw1 + xi_aw1)/tau_1

    @staticmethod
    def __round_ISO(x, Q):
        '''
        Credits for the original version of this method go to:
            [1] E. M. F. Donéstevez, “Python library for design of spur and 
            helical gears transmissions.” Zenodo, 09-Feb-2020, 
            doi: 10.5281/ZENODO.3660527.

        It was modified to account for the case where x is an array and to 
        remove the second argument
        
        Parameters
        ----------
        x : float
            DESCRIPTION.

        Returns
        -------
        val : TYPE
            DESCRIPTION.

        '''
        # 
        x *= pow(2.0, (Q - 5.0)/2.0)
        
        val = np.zeros(len(x))
        
        for idx in range(len(x)):
            xx = x[idx]
            if(xx >= 10.0):
                val[idx] = round(xx)
            elif(5.0 <= xx <= 10.0):
                if((xx % 1 <= 0.25) or (0.5 <= xx % 1 % 1 <= 0.75)):
                    val[idx] = floor(2.0*xx)/2.0
                else:
                    val[idx] = ceil(2.0*xx)/2.0
            else:
                val[idx] = round(xx, 1)
        
        return val

    @staticmethod
    def example_01_ISO6336():
        return  GearSet(configuration = 'parallel',
                        m_n           = 8.0,
                        z             = np.array([17, 103]),
                        alpha_n       = 20.0,
                        beta          = 15.8,
                        b             = 100.0,
                        a_w           = 500.0,
                        x             = np.array([0.145, 0.0]),
                        rack_type     = 'D',
                        Q             = 5)
    
    def __root_form_diameter(self):
        '''
        ISO 21771, Sec. 7.6, Eq. (128)
        '''
        B = (self.h_fP - self.x*self.m_n + self.rho_fP*(np.sin(np.radians(self.alpha_n)) - 1.0))
        return np.sqrt((self.d*np.sin(np.radians(self.alpha_t)) - 2.0*B/np.sin(np.radians(self.alpha_t)))**2 + self.d_b**2)

    def __SAP_diameter(self):
        '''
        ISO 21771, Sec. 5.4.1, Eqs. (64-67)
        '''
        d_Fa = self.d_a
        dNf1 = np.sqrt((2.0*self.a_w*np.sin(np.radians(self.alpha_wt)) -np.sign(self.z[1])*np.sqrt(d_Fa[1]**2 -self.d_b[1]**2))**2 + self.d_b[0]**2)
        dNf2 = np.sqrt((2.0*self.a_w*np.sin(np.radians(self.alpha_wt)) -                np.sqrt(d_Fa[0]**2 -self.d_b[0]**2))**2 + self.d_b[1]**2)
        
        dNf = [dNf1, dNf2]

        if(self.configuration == 'planetary'):
            dNf.append(np.nan)
        
        for i in range(len(self.z)):
            if(self.d_Ff[i] > dNf[i]):
                dNf[i] = self.d_Ff[i]

        return np.array(dNf)

    def __active_tip_diameter(self):
        '''
        ISO 21771, Sec. 5.4.1, Eqs. (68-69)
        '''
        aw = 2.0*self.a_w*np.sin(np.radians(self.alpha_wt))
        if(self.d_Nf[0] == self.d_Ff[0]):
            dNa2 = np.sqrt((aw -                 np.sqrt(self.d_Ff[0]**2 - self.d_b[0]**2))**2 + self.d_b[1]**2)
        else:
            dNa2 = self.d_a[1] # d_Fa

        if(self.d_Nf[1] == self.d_Ff[1]):
            dNa1 = np.sqrt((aw - np.sign(self.z[1])*np.sqrt(self.d_Ff[1]**2 - self.d_b[1]**2))**2 + self.d_b[0]**2)
        else:
            dNa1 = self.d_a[0] # d_Fa
        
        dNa = [dNa1, dNa2]
        if(self.configuration == 'planetary'):
            dNa.append(np.nan)

        return np.array(dNa)
