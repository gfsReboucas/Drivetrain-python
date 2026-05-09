# -*- coding: utf-8 -*-
"""Lower-level ISO 6336 factor calculations."""

from scipy import interpolate
import numpy as np


def _face_load_factor(gset):
    # The face load factor (contact stress) is not calculated from 
    # ISO 6336 [1], but according to [4], Eq.(17.14):
    K_Hbeta = 1.0 + 0.4*(gset.b/gset.d[0])**2
    
    # Method B:
    # h_1 = h_aP + h_fP + k_1*m_n;
    h_1 = abs(gset.d_a[0] - gset.d_f[0])/2.0
    bh1 = gset.b/h_1
    h_2 = abs(gset.d_a[1] - gset.d_f[1])/2.0
    bh2 = gset.b/h_2

    bh = min(bh1, bh2)

    if(bh < 3.0):
        bh = 3.0

    hb = 1.0/bh

    N_F = 1.0/(1.0 + hb + hb**2)

    # Face load factor (root stress):
    K_Fbeta = pow(K_Hbeta, N_F)
    
    return {'K_Hbeta': K_Hbeta,
            'K_Fbeta': K_Fbeta}

def _transv_load_factor(gset, term, Z_eps):
    # Section 8.3.2 of [1]:
    if(gset.eps_gamma <= 2.0): # Eq. (73)
        K_Falpha = (0.9 + 0.4*term)*(gset.eps_gamma/2.0)
    else: # Eq. (74)
        K_Falpha = 0.9 + 0.4*np.sqrt(2.0*(gset.eps_gamma - 1.0)/gset.eps_gamma)*term
    
    K_Halpha = K_Falpha
    
    # Section 8.3.4 of [1]:
    # Transverse load factor (contact stress), Eq. (75):
    K_Halpha_lim = gset.eps_gamma/(gset.eps_alpha*Z_eps**2)
    if(K_Halpha > K_Halpha_lim):
        K_Halpha = K_Halpha_lim
    elif(K_Halpha < 1.0):
        K_Halpha = 1.0

    # Section 8.3.5 of [1]:
    # Transverse load factor (root stress), Eq. (76):
    K_Falpha_lim = gset.eps_gamma/(0.25*gset.eps_alpha + 0.75)
    if(K_Falpha > K_Falpha_lim):
        K_Falpha = K_Falpha_lim
    elif(K_Falpha < 1.0):
        K_Falpha = 1.0
    
    return {'K_Halpha': K_Halpha,
            'K_Falpha': K_Falpha}
    
def _zone_factor(gset):
    num = 2.0*np.cos(np.radians(gset.beta_b))*np.cos(np.radians(gset.alpha_wt))
    den = np.sin(np.radians(gset.alpha_wt))*np.cos(np.radians(gset.alpha_t))**2
    return np.sqrt(num/den)

def _tooth_contact_factor(gset):
    
    M_1 = np.tan(gset.alpha_wt)/np.sqrt((np.sqrt((gset.d_a[0]/gset.d_b[0])**2 - 1.0) - 
                                   2.0*np.pi/gset.z[0])*(np.sqrt((gset.d_a[1]/gset.d_b[1])**2 - 1.0) - 
                                                      (gset.eps_alpha - 1.0)*2.0*np.pi/gset.z[1]))
    M_2 = np.tan(gset.alpha_wt)/np.sqrt((np.sqrt((gset.d_a[1]/gset.d_b[1])**2 - 1.0) - 
                                   2.0*np.pi/gset.z[1])*(np.sqrt((gset.d_a[0]/gset.d_b[0])**2 - 1.0) - 
                                                      (gset.eps_alpha - 1.0)*2.0*np.pi/gset.z[0]))

    if((gset.eps_beta == 0.0) and (gset.eps_alpha > 1.0)):
        if(M_1 > 1.0):
            Z_B = M_1
        else:
            Z_B = 1.0

        if(M_2 > 1.0):
            Z_D = M_2
        else:
            Z_D = 1.0
    elif((gset.eps_alpha > 1.0) and (gset.eps_beta >= 1.0)):
        Z_B = 1.0
        Z_D = 1.0
    elif((gset.eps_alpha > 1.0) and (gset.eps_beta <  1.0)):
        Z_B = M_1 - gset.eps_beta*(M_1 - 1.0)
        Z_D = M_2 - gset.eps_beta*(M_2 - 1.0)
    
    return {'Z_B': Z_B,
            'Z_D': Z_D}

def _lub_vel_factor(sigma_Hlim, nu_40, v):
    if(sigma_Hlim  < 850.0): # [N/mm^2]
        C_ZL = 0.83
    elif(850.0 <= sigma_Hlim  < 1200.0):
        C_ZL = sigma_Hlim/4375.0 + 0.6357
    else:
        C_ZL = 0.91

    # Lubricant factor:
    Z_L = C_ZL + 4.0*(1.0 - C_ZL)/(1.2 + 134.0/nu_40)**2

    # Velocity factor:
    C_Zv = C_ZL + 0.02
    Z_v = C_Zv + 2.0*(1.0 - C_Zv)/np.sqrt(0.8 + 32.0/v)
    
    return {'Z_v': Z_v,
            'Z_L': Z_L}

def _rough_factor(gset, R_zh, sigma_Hlim):
    rho_1 = 0.5*gset.d_b[0]*np.tan(np.radians(gset.alpha_wt))
    rho_2 = 0.5*gset.d_b[1]*np.tan(np.radians(gset.alpha_wt))

    rho_red = (rho_1*rho_2)/(rho_1 + rho_2)

    R_z10 = R_zh*pow(10.0/rho_red, 1.0/3.0)

    if(sigma_Hlim  < 850.0): # [N/mm^2]
        C_ZR = 0.15
    elif(850.0 <= sigma_Hlim < 1200.0):
        C_ZR = 0.32 - sigma_Hlim*2.0e-4
    else:
        C_ZR = 0.08

    return pow(3.0/R_z10, C_ZR)

def _interp_ZNT(line, N):

    if(line == 1):
        # St, V, GGG (perl. bai.), GTS (perl.), Eh, IF (when limited pitting is permitted)
        x = [6.0e5, 1.0e7, 1.0e9, 1.0e10]
        y = [1.6,   1.3,   1.0,  0.85]
    elif(line == 2):
        # St, V, GGG (perl. bai.), GTS (perl.), Eh, IF
        x = [1.0e5, 5.0e7, 1.0e9, 1.0e10]
        y = [1.6,   1.0,   1.0,   0.85]
    elif(line == 3):
        # GG, GGG (ferr.), NT (nitr.), NV (nitr.)
        x = [1.0e5, 2.0e6, 1.0e10]
        y = [1.3,   1.0,   0.85]
    elif(line == 4):
        # NV (nitrocar.)
        x = [1.0e5, 2.0e6, 1.0e10]
        y = [1.1,   1.0,   0.85]
    else:
        raise Exception('Invalid input [{}].\n'.format(line))
    
    x = np.array(x)
    y = np.array(y)

    fun = interpolate.interp1d(np.log(x), y, kind = 'linear', 
                               fill_value = (y[0], y[-1]),
                               bounds_error = False)

    return fun(np.log(N)).item()

def _contact_ratio_factor(gset):
    eps_a = gset.eps_alpha
    eps_b = gset.eps_beta
    
    if(gset.beta == 0.0):
        Z_eps = np.sqrt((4.0 - eps_a)/3.0)
    else:
        if(eps_b < 1.0):
            Z_eps = np.sqrt((1.0 - eps_b)*(4.0 - eps_a)/3.0 + eps_b/eps_a)
        else:
            Z_eps = np.sqrt(1.0/eps_a)
    
    return Z_eps

# def

###############################################################################
