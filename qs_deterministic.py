#!/bin/python3

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt; plt.rc('font', size=16)
import matplotlib.cm as cm
from numba import jit

# Default parameters
#-----------------------------------------------------------------------------------------------------------------------

N = 10              
# V_cell = 1.8e-9
# V_ext = 1e-6

k_lasR = 1                  # 0
g_lasR = 0.247              # 1
k_LasR = 50                 # 2
g_LasR = 0.027              # 3
a_rsaL = 0.01               # 4
b_rsaL = 1.5                # 5
K1 = 4000                   # 6
h1 = 1.2                    # 7
g_rsaL = 0.247              # 8
k_RsaL = 50                 # 9
g_RsaL = 0.027              # 10
a_lasI = 0.01               # 11
b_lasI = 1.5                # 12
K2 = 6500                   # 13
h2 = 1.4                    # 14
g_lasI = 0.247              # 15
k_LasI = 50                 # 16
g_LasI = 0.015              # 17
k_AI1 = 0.04                # 18
g_AI1 = 0.008               # 19
g_AI1_ext = 0.057           # 20
s_LasRAI1 = 10              # 21
u_LasRAI1 = s_LasRAI1/100   # 22
g_LasRAI1 = 0.017           # 23
D = 8                       # 24
D_away = 0.1                # 25

params = (k_lasR, g_lasR, k_LasR, g_LasR, a_rsaL, b_rsaL, K1, h1, g_rsaL, k_RsaL, g_RsaL, a_lasI, b_lasI, \
          K2, h2, g_lasI, k_LasI, g_LasI, k_AI1, g_AI1, g_AI1_ext, s_LasRAI1, u_LasRAI1, g_LasRAI1, D, D_away)

t_span = [0, 800]

r0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# Functions
#-----------------------------------------------------------------------------------------------------------------------

@jit
def LasRI_RsaL_qs(t, r, N, params):

    V_cell = 1.8e-9
    V_ext = 1e-6

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
    s_LasRAI1 = params[21]
    u_LasRAI1 = params[22]
    g_LasRAI1 = params[23]
    D         = params[24]
    D_away    = params[25]

    V_c = V_cell/V_ext
    LasRAI1_ = (LasRAI1/K1)**h1
    RsaL_ = (RsaL/K2)**h2
    
    dlasR = k_lasR - lasR*g_lasR
    dLasR = lasR*k_LasR + LasRAI1*s_LasRAI1 - AI1*LasR*u_LasRAI1 - LasR*g_LasR
    drsaL = a_rsaL + b_rsaL*(LasRAI1_/(1+LasRAI1_)) - rsaL*g_rsaL
    dRsaL = rsaL*k_RsaL - RsaL*g_RsaL
    dlasI = a_lasI + (b_lasI*(LasRAI1_/((1+LasRAI1_)*(1+RsaL_)))) - lasI*g_lasI
    dLasI = lasI*k_LasI - LasI*g_LasI
    dAI1 = LasI*k_AI1 + LasRAI1*s_LasRAI1 - AI1*(LasR*u_LasRAI1 + g_AI1) - D*(AI1 - V_c*AI1_ext)
    dAI1_ext = N*D*(AI1 - V_c*AI1_ext) - (AI1_ext*(g_AI1_ext + D_away))
    dLasRAI1 = AI1*LasR*u_LasRAI1 - (LasRAI1*(s_LasRAI1 + g_LasRAI1))
    
    return np.array([dlasR, dLasR, drsaL, dRsaL, dlasI, dLasI, dAI1, dAI1_ext, dLasRAI1])

def solve_qs(qs_sys=LasRI_RsaL_qs, t_span=t_span, r0=r0, N=N, params=params):
    return solve_ivp(lambda t, r: qs_sys(t, r, N, params), t_span, r0)

def plot_det(qs_dynamics, N=N, action='display'):

    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI$_1$', 'AI$_{1,ext}$', 'LasR$\cdot$AI$_1$']
    fig = plt.figure(figsize=(9,6))
    fig.suptitle(f'QS dynamics for {N} cells')
    for i in range(len(x)):
        plt.plot(qs_dynamics.t, qs_dynamics.y[i], lw=2, label=x[i], color=cm.tab10(i))
    plt.grid()
    # plt.legend(loc='center right', bbox_to_anchor=(1.20, 0.50), ncol=1, fancybox=True, shadow=False, fontsize=12)
    plt.legend(loc=0, ncol=3, framealpha=0.4, fontsize=14)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration')

    if action == 'save':
        fig.savefig(f'qs_dynamics_{N}.pdf', format='pdf', bbox_inches='tight')
    else:
        return fig

def subplots_det(qs_dynamics, N=N, action='display'):

    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI$_1$', 'AI$_{1,ext}$', 'LasR$\cdot$AI$_1$']
    fig = plt.figure(figsize=(18,12))

    for i in [0,2,4]:
        plt.subplot(2,2,1)
        plt.plot(qs_dynamics.t, qs_dynamics.y[i], lw=2, color=cm.tab10(i), label=x[i])
    plt.legend(loc=0, ncol=1, framealpha=0.4, fontsize=14)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration')
    plt.grid()

    for i in [1,3,5,8]:
        plt.subplot(2,2,2)
        plt.plot(qs_dynamics.t, qs_dynamics.y[i], lw=2, color=cm.tab10(i), label=x[i])
    plt.legend(loc=0, ncol=2, framealpha=0.4, fontsize=14)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration')
    plt.grid()

    plt.subplot(2,2,3)
    plt.plot(qs_dynamics.t, qs_dynamics.y[6], lw=2, color=cm.tab10(6), label=x[6])
    plt.legend(loc=0, ncol=1, framealpha=0.4, fontsize=14)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration')
    plt.grid()

    plt.subplot(2,2,4)
    plt.plot(qs_dynamics.t, qs_dynamics.y[7], lw=2, color=cm.tab10(7), label=x[7])
    plt.legend(loc=0, ncol=1, framealpha=0.4, fontsize=14)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration')
    plt.grid()

    if action == 'save':
        fig.savefig(f'qs_dynamics_sub_{N}.pdf', format='pdf', bbox_inches='tight')
    else:
        return fig