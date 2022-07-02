#!/bin/python3

import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

@jit
def LasRI_RsaL_qs(t, r, p):
    lasR, LasR, rsaL, RsaL, lasI, LasI, AI1, AI1_ext, LasRAI1 = r

    # LasR/LasI
    N         = p[0]
    k_lasR    = p[1]
    g_lasR    = p[2]
    k_LasR    = p[3]
    g_LasR    = p[4]
    a_rsaL    = p[5]
    b_rsaL    = p[6]
    K1        = p[7]
    h1        = p[8]
    g_rsaL    = p[9]
    k_RsaL    = p[10]
    g_RsaL    = p[11]
    a_lasI    = p[12]
    b_lasI    = p[13]
    K2        = p[14]
    h2        = p[15]
    g_lasI    = p[16]
    k_LasI    = p[17]
    g_LasI    = p[18]
    k_AI1     = p[19]
    g_AI1     = p[20]
    g_AI1_ext = p[21]
    u_LasRAI1 = p[22]
    s_LasRAI1 = p[23]
    g_LasRAI1 = p[24]
    # Cell parameters
    d         = p[25]
    d_away    = p[26]

    x = ( LasRAI1 /K1 )** h1   # Activator
    y = ( RsaL / K2 )** h2      # Repressor
    
    dlasR = k_lasR - lasR * g_lasR
    dLasR = lasR * k_LasR + LasRAI1 * s_LasRAI1 - AI1 * LasR * u_LasRAI1 - LasR * g_LasR
    drsaL = a_rsaL + b_rsaL *(x/(1+x)) - rsaL * g_rsaL
    dRsaL = rsaL * k_RsaL - RsaL * g_RsaL
    dlasI = a_lasI + ( b_lasI *(x/((1+x)*(1+y)))) - lasI * g_lasI
    dLasI = lasI * k_LasI - LasI * g_LasI
    dAI1 = LasI * k_AI1 + LasRAI1 * s_LasRAI1 - AI1 * LasR * u_LasRAI1 - ( d *( AI1 - AI1_ext )) - AI1 * g_AI1
    dAI1_ext = ( N * d *( AI1 - AI1_ext )) - ( AI1_ext *( g_AI1_ext + d_away ))
    dLasRAI1 = AI1 * LasR * u_LasRAI1 - ( LasRAI1*( s_LasRAI1 + g_LasRAI1 ))
    
    return np.array([dlasR, dLasR, drsaL, dRsaL, dlasI, dLasI, dAI1, dAI1_ext, dLasRAI1])

# LasR/LasI
N = 500           # 0
k_lasR = 0.004    # 1
g_lasR = 0.002    # 2
k_LasR = 0.4      # 3
g_LasR = 0.35     # 4
a_rsaL = 0.00036  # 5
b_rsaL = 0.0058   # 6
K1 = 1.2          # 7
h1 = 1.4          # 8
g_rsaL = 0.001    # 9
k_RsaL = 0.9      # 10
g_RsaL = 0.2      # 11
a_lasI = 0.00036  # 12
b_lasI = 0.0058   # 13
K2 = 1.4          # 14
h2 = 1.2          # 15
g_lasI = 0.001    # 16
k_LasI = 0.7      # 17
g_LasI = 0.12     # 18
k_AI1 = 1         # 19
g_AI1 = 0.3       # 20
g_AI1_ext = 0.8   # 21
u_LasRAI1 = 0.05  # 22
s_LasRAI1 = 0.25  # 23
g_LasRAI1 = 0.14  # 24
# Cell parameters   
d = 0.8           # 25
d_away = 1.2      # 26
p = (N, k_lasR, g_lasR, k_LasR, g_LasR, a_rsaL, b_rsaL, K1, h1, g_rsaL, k_RsaL, g_RsaL, a_lasI, b_lasI, K2, h2, g_lasI, k_LasI, g_LasI, k_AI1, g_AI1, g_AI1_ext, u_LasRAI1, s_LasRAI1, g_LasRAI1, d, d_away)