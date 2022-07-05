#!/bin/python3

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import jit

# Default parameters
#-----------------------------------------------------------------------------------------------------------------------

N = 10              

k_lasR = 0.004      # 0
g_lasR = 0.002      # 1
k_LasR = 0.9        # 2
g_LasR = 0.35       # 3
a_rsaL = 0.00036    # 4
b_rsaL = 0.0058     # 5
K1 = 1.2            # 6
h1 = 1.4            # 7
g_rsaL = 0.001      # 8
k_RsaL = 0.7        # 9
g_RsaL = 0.2        # 10
a_lasI = 0.00036    # 11
b_lasI = 0.0058     # 12
K2 = 1.4            # 13
h2 = 1.2            # 14
g_lasI = 0.001      # 15
k_LasI = 0.7        # 16
g_LasI = 0.12       # 17
k_AI1 = 2           # 18
g_AI1 = 0.3         # 19
g_AI1_ext = 0.8     # 20
u_LasRAI1 = 0.05    # 21
s_LasRAI1 = 0.25    # 22
g_LasRAI1 = 0.14    # 23
d = 0.8             # 24
d_away = 0.6        # 25

params = (k_lasR, g_lasR, k_LasR, g_LasR, a_rsaL, b_rsaL, K1, h1, g_rsaL, k_RsaL, g_RsaL, a_lasI, b_lasI, \
          K2, h2, g_lasI, k_LasI, g_LasI, k_AI1, g_AI1, g_AI1_ext, u_LasRAI1, s_LasRAI1, g_LasRAI1, d, d_away)

t_span = [0, 170]

r0 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# Functions
#-----------------------------------------------------------------------------------------------------------------------

@jit
def LasRI_RsaL_qs(t, r, N, params):

    lasR, LasR, rsaL, RsaL, lasI, LasI, AI1, AI1_ext, LasRAI1 = r

    k_lasR    = params[0]
    g_lasR    = params[1]
    k_LasR    = params[2]
    g_LasR    = params[3]
    a_rsaL    = params[4]
    b_rsaL    = params[5]
    K1        = params[6]
    h1        = params[7]
    g_rsaL    = params[8]
    k_RsaL    = params[9]
    g_RsaL    = params[10]
    a_lasI    = params[11]
    b_lasI    = params[12]
    K2        = params[13]
    h2        = params[14]
    g_lasI    = params[15]
    k_LasI    = params[16]
    g_LasI    = params[17]
    k_AI1     = params[18]
    g_AI1     = params[19]
    g_AI1_ext = params[20]
    u_LasRAI1 = params[21]
    s_LasRAI1 = params[22]
    g_LasRAI1 = params[23]
    d         = params[24]
    d_away    = params[25]

    LasRAI1_ = (LasRAI1/K1 )**h1
    RsaL_ = (RsaL/K2 )**h2
    
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

def solve_qs(qs_sys=LasRI_RsaL_qs, t_span=t_span, r0=r0, N=N, params=params):
    return solve_ivp(lambda t, r: qs_sys(t, r, N, params), t_span, r0)

def plot_det(qs_dynamics, N=N, action='display'):

    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI$_1$', 'AI$_{1,ext}$', 'LasR$\cdot$AI$_1$']
    fig = plt.figure(figsize=(12,7))
    fig.suptitle(f'QS dynamics for {N} cells', fontsize=16)
    for i in range(len(x)):
        plt.plot(qs_dynamics.t, qs_dynamics.y[i], lw=2, label=x[i], color=cm.tab10(i))
    plt.grid()
    plt.legend(loc='center right', bbox_to_anchor=(1.20, 0.50), ncol=1, fancybox=True, shadow=False, fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)

    if action == 'save':
        fig.savefig(f'qs_dynamics_{N}.png', bbox_inches='tight')
    else:
        return fig

def plot_det2(qs_dynamics_short, qs_dynamics_long, N=N, action='display'):

    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI$_1$', 'AI$_{1,ext}$', 'LasR$\cdot$AI$_1$']
    fig = plt.figure(figsize=(22,7))
    fig.suptitle(f'QS dynamics for {N} cells', fontsize=16)
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1,2,1)
    for i in range(len(x)):
        plt.plot(qs_dynamics_short.t, qs_dynamics_short.y[i], lw=2, label=x[i], color=cm.tab10(i), alpha=0.7)
    plt.grid()
    plt.legend(loc='center right', bbox_to_anchor=(1.20, 0.50), ncol=1, fancybox=True, shadow=False, fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)
    plt.title('LasR/LasI with RsaL short dynamics', fontsize=14)
    plt.subplot(1,2,2)
    for i in range(len(x)):
        plt.plot(qs_dynamics_long.t, qs_dynamics_long.y[i], lw=2, label=x[i], color=cm.tab10(i), alpha=0.7)
    plt.grid()
    plt.legend(loc='center right', bbox_to_anchor=(1.20, 0.50), ncol=1, fancybox=True, shadow=False, fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)
    plt.title('LasR/LasI with RsaL long dynamics', fontsize=14)

    if action == 'save':
        fig.savefig(f'qs_dynamics2_{N}.png', bbox_inches='tight')
    else:
        return fig