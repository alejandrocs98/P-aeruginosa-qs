#!/bin/python3

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import jit

# Default parameters
#-----------------------------------------------------------------------------------------------------------------------

N = 10            # 0
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
d = 0.8           # 25
d_away = 1.2      # 26

params = np.array([N, k_lasR, g_lasR, k_LasR, g_LasR, a_rsaL, b_rsaL, K1, h1, g_rsaL, k_RsaL, g_RsaL, a_lasI, K2, \
              h2, g_lasI, k_LasI, g_LasI, k_AI1, g_AI1, g_AI1_ext, u_LasRAI1, s_LasRAI1, g_LasRAI1, d, d_away])

t = [0, 170]

r0 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# Functions
#-----------------------------------------------------------------------------------------------------------------------

@jit
def LasRI_RsaL_qs(t, r, params=params):

    lasR, LasR, rsaL, RsaL, lasI, LasI, AI1, AI1_ext, LasRAI1 = r

    N         = params[0]
    k_lasR    = params[1]
    g_lasR    = params[2]
    k_LasR    = params[3]
    g_LasR    = params[4]
    a_rsaL    = params[5]
    b_rsaL    = params[6]
    K1        = params[7]
    h1        = params[8]
    g_rsaL    = params[9]
    k_RsaL    = params[10]
    g_RsaL    = params[11]
    a_lasI    = params[12]
    b_lasI    = params[13]
    K2        = params[14]
    h2        = params[15]
    g_lasI    = params[16]
    k_LasI    = params[17]
    g_LasI    = params[18]
    k_AI1     = params[19]
    g_AI1     = params[20]
    g_AI1_ext = params[21]
    u_LasRAI1 = params[22]
    s_LasRAI1 = params[23]
    g_LasRAI1 = params[24]
    d         = params[25]
    d_away    = params[26]

    LasRAI1_ = (LasRAI1/K1 )**h1   # Activator
    RsaL_ = (RsaL/K2 )**h2      # Repressor
    
    dlasR = k_lasR - lasR*g_lasR
    dLasR = lasR*k_LasR + LasRAI1*s_LasRAI1 - AI1*LasR*u_LasRAI1 - LasR*g_LasR
    drsaL = a_rsaL + b_rsaL*(LasRAI1_/(1+LasRAI1_)) - rsaL*g_rsaL
    dRsaL = rsaL*k_RsaL - RsaL*g_RsaL
    dlasI = a_lasI + (b_lasI*(LasRAI1_/((1+LasRAI1_)*(1+RsaL_)))) - lasI*g_lasI
    dLasI = lasI*k_LasI - LasI*g_LasI
    dAI1 = LasI*k_AI1 + LasRAI1*s_LasRAI1 - AI1*LasR*u_LasRAI1 - (d*(AI1 - AI1_ext)) - AI1*g_AI1
    dAI1_ext = (N*d*(AI1 - AI1_ext)) - (AI1_ext*(g_AI1_ext + d_away))
    dLasRAI1 = AI1*LasR*u_LasRAI1 - (LasRAI1*(s_LasRAI1 + g_LasRAI1))
    
    return np.array([dlasR, dLasR, drsaL, dRsaL, dlasI, dLasI, dAI1, dAI1_ext, dLasRAI1])

def solve_qs(qs_sys=LasRI_RsaL_qs, t=t, r0=r0, params=params):
    return solve_ivp(qs_sys, t, r0, args=params)

def plot_det(qs_dynamics, N=N, action='display'):

    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI$_1$', 'AI$_{1,ext}$', 'LasR$\cdot$AI$_1$']
    fig = plt.figure(figsize=(12,7))
    for i in range(len(x)):
        plt.plot(qs_dynamics.t, qs_dynamics.y[i], lw=2, label=x[i], color=cm.tab10(i))
    plt.grid()
    plt.legend(loc='center right', bbox_to_anchor=(1.20, 0.50), ncol=1, fancybox=True, shadow=False, fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)

    if action == 'save':
        fig.savefig(f'qs_dynamics_{N}.png')
    else:
        return fig