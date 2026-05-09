"""Rolling bearing stiffness and damping containers."""

from dataclasses import dataclass, field

import numpy as np
import scipy.stats as stats

@dataclass
class Bearing:
    '''
    This class contains some geometric and dynamic properties of rolling 
    bearings.
    
    written by:
        Geraldo Rebouças
        - gfs.reboucas@gmail.com
        - https://gfsreboucas.github.io
    '''
    
    stiffness: np.ndarray = field(default_factory=lambda: np.zeros(6))
    damping: np.ndarray = field(default_factory=lambda: np.zeros(6))
    name: object = '-*-'
    type: object = 'none'
    OD: object = 0.0
    ID: object = 0.0
    B: object = 0.0

    def __post_init__(self):
        # [N/m],     Translational stiffness, x axis:
        self.k_x     = self.stiffness[0]
        # [N/m],     Translational stiffness, y axis:
        self.k_y     = self.stiffness[1]
        # [N/m],     Translational stiffness, z axis:
        self.k_z     = self.stiffness[2]
        # [N-m/rad], Torsional stiffness, x axis (rot.):
        self.k_alpha = self.stiffness[3]
        # [N-m/rad], Torsional stiffness, y axis:
        self.k_beta  = self.stiffness[4]
        # [N-m/rad], Torsional stiffness, z axis:
        self.k_gamma = self.stiffness[5]
        
        # [N-s/m],     Translational damping, x axis:
        self.d_x     = self.damping[0]
        # [N-s/m],     Translational damping, y axis:
        self.d_y     = self.damping[1]
        # [N-s/m],     Translational damping, z axis:
        self.d_z     = self.damping[2]
        # [N-m-s/rad], Torsional damping, x axis (rot.):
        self.d_alpha = self.damping[3]
        # [N-m-s/rad], Torsional damping, y axis:
        self.d_beta  = self.damping[4]
        # [N-m-s/rad], Torsional damping, z axis:
        self.d_gamma = self.damping[5]
        
    def __repr__(self):
        
        val = ('Bearing type:                                  {}   -\n'.format(self.type) +
               'Translational Stiffness @ x axis,    k_x     = {:.5e} N/m\n'.format(self.k_x) +
               'Translational Stiffness @ y axis,    k_y     = {:.5e} N/m\n'.format(self.k_y) +
               'Translational Stiffness @ z axis,    k_z     = {:.5e} N/m\n'.format(self.k_z) +
               'Torsional Stiffness @ x axis (rot.), k_alpha = {:.5e} N-m/rad\n'.format(self.k_alpha) +
               'Torsional Stiffness @ y axis,        k_beta  = {:.5e} N-m/rad\n'.format(self.k_beta) +
               'Torsional Stiffness @ z axis,        k_gamma = {:.5e} N-m/rad\n'.format(self.k_gamma) +
               'Translational damping @ x axis,      d_x     = {:.5e} N-s/m\n'.format(self.d_x) +
               'Translational damping @ y axis,      d_y     = {:.5e} N-s/m\n'.format(self.d_y) +
               'Translational damping @ z axis,      d_z     = {:.5e} N-s/m\n'.format(self.d_z) +
               'Torsional damping @ x axis (rot.),   d_alpha = {:.5e} N-m-s/rad\n'.format(self.d_alpha) +
               'Torsional damping @ y axis,          d_beta  = {:.5e} N-m-s/rad\n'.format(self.d_beta) +
               'Torsional damping @ z axis,          d_gamma = {:.5e} N-m-s/rad\n'.format(self.d_gamma) +
               'Outer diameter,                      OD      = {} mm\n'.format(self.OD) +
               'Inner diameter,                      ID      = {} mm\n'.format(self.ID) +
               'Width,                               B       = {} mm\n'.format(self.B))
        
        return val

    def __len__(self):
        return len(self.k_x)

    def __getitem__(self, key):
        return Bearing(np.array([self.k_x[key],
                              self.k_y[key],
                              self.k_z[key],
                              self.k_alpha[key],
                              self.k_beta[key],
                              self.k_gamma[key]]),
                       np.array([self.d_x[key],
                              self.d_y[key],
                              self.d_z[key],
                              self.d_alpha[key],
                              self.d_beta[key],
                              self.d_gamma[key]]))
                             
    def series_association(self):
        if(np.isscalar(self.k_x)):
            print('Only one bearing.')
            return self
        else:
            kx = stats.hmean(self.k_x)    /self.k_x.size()
            ky = stats.hmean(self.k_y)    /self.k_y.size()
            kz = stats.hmean(self.k_z)    /self.k_z.size()
            ka = stats.hmean(self.k_alpha)/self.k_alpha.size()
            kb = stats.hmean(self.k_beta) /self.k_beta.size()
            kg = stats.hmean(self.k_gamma)/self.k_gamma.size()
            
            dx = stats.hmean(self.d_x)    /self.d_x.size()
            dy = stats.hmean(self.d_y)    /self.d_y.size()
            dz = stats.hmean(self.d_z)    /self.d_z.size()
            da = stats.hmean(self.d_alpha)/self.d_alpha.size()
            db = stats.hmean(self.d_beta) /self.d_beta.size()
            dg = stats.hmean(self.d_gamma)/self.d_gamma.size()
            
        k = [kx, ky, kz, ka, kb, kg]
        d = [dx, dy, dz, da, db, dg]
            
        return Bearing(k, d, name = ' / '.join(self.name), type = 'series', \
                       OD = np.mean(self.OD), ID = np.mean(self.ID), B = np.mean(self.B))
        
    def parallel_association(self):
        if(np.isscalar(self.k_x)):
            print('Only one bearing.')
            return self
        else:
            kx = sum(self.k_x)
            ky = sum(self.k_y)
            kz = sum(self.k_z)
            ka = sum(self.k_alpha)
            kb = sum(self.k_beta)
            kg = sum(self.k_gamma)
            
            dx = sum(self.d_x)
            dy = sum(self.d_y)
            dz = sum(self.d_z)
            da = sum(self.d_alpha)
            db = sum(self.d_beta)
            dg = sum(self.d_gamma)
            
        k = np.array([kx, ky, kz, ka, kb, kg])
        d = np.array([dx, dy, dz, da, db, dg])

        return Bearing(k, d, name = ' / '.join(self.name), type = 'parallel', \
                       OD = np.mean(self.OD), ID = np.mean(self.ID), B = np.mean(self.B))
        
    def stiffness_matrix(self):
        if(np.isscalar(self.k_x)):
            return np.diag([self.k_x,     self.k_y,    self.k_z, 
                         self.k_alpha, self.k_beta, self.k_gamma])
        else:
            print('Only one bearing.')
        
    def damping_matrix(self):
        if(np.isscalar(self.d_x)):
            return np.diag([self.d_x,     self.d_y,    self.d_z, 
                         self.d_alpha, self.d_beta, self.d_gamma])
        else:
            print('Only one bearing.')
