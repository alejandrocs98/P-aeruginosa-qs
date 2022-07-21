#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt; plt.rc('font', size=16)
import pandas as pd
from numba import njit

# Default parameters
#-----------------------------------------------------------------------------------------------------------------------

ensemble_size = 100 #int(0.5e3)

N = 10
AI1_ext = 0

t_max = 600
dt_reg = 0.01

x0=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

reactions = np.array([[1, 0, 0, 0, 0, 0, 0, 0],    # + lasR
                      [-1, 0, 0, 0, 0, 0, 0, 0],   # - lasR
                      [0, 1, 0, 0, 0, 0, 0, 0],    # + LasR
                      [0, 1, 0, 0, 0, 0, 1, -1],   # + LasR ; + AI1 ; - LasRAI1
                      [0, -1, 0, 0, 0, 0, -1, 1],  # - LasR ; - AI1 ; + LasRAI1
                      [0, -1, 0, 0, 0, 0, 0, 0],   # - LasR
                      [0, 0, 1, 0, 0, 0, 0, 0],    # + rsaL
                      [0, 0, -1, 0, 0, 0, 0, 0],   # - rsaL
                      [0, 0, 0, 1, 0, 0, 0, 0],    # + RsaL
                      [0, 0, 0, -1, 0, 0, 0, 0],   # - RsaL
                      [0, 0, 0, 0, 1, 0, 0, 0],    # + lasI
                      [0, 0, 0, 0, -1, 0, 0, 0],   # - lasI
                      [0, 0, 0, 0, 0, 1, 0, 0],    # + LasI
                      [0, 0, 0, 0, 0, -1, 0, 0],   # - LasI
                      [0, 0, 0, 0, 0, 0, 1, 0],    # + AI1
                      [0, 0, 0, 0, 0, 0, -1, 0],   # - AI1
                      [0, 0, 0, 0, 0, 0, -1, 0],   # - AI1 ; + AI1_ext
                      [0, 0, 0, 0, 0, 0, 0, -1],   # - LasRAI1
                      [0, 0, 0, 0, 0, 0, 1, 0]],   # + AI1 ; - AI1_ext
                dtype=np.int32)

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
s_LasRAI1 = 10              # 20
u_LasRAI1 = s_LasRAI1/100   # 21
g_LasRAI1 = 0.017           # 22
D = 8                       # 23

params = np.array([k_lasR, g_lasR, k_LasR, g_LasR, a_rsaL, b_rsaL, K1, h1, g_rsaL, k_RsaL, g_RsaL, a_lasI, \
                   b_lasI, K2, h2, g_lasI, k_LasI, g_LasI, k_AI1, g_AI1, s_LasRAI1, u_LasRAI1, g_LasRAI1, D], dtype=np.float32)

@njit()
def pa_qs1(x, params, N, AI1_ext, reactions):
    
    lasR, LasR, rsaL, RsaL, lasI, LasI, AI1, LasRAI1 = x
    propensities = np.empty(len(reactions), dtype=np.float64)
    
    V_cell    = 1.8e-9
    V_ext     = 1e-6
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
    s_LasRAI1 = params[20]
    u_LasRAI1 = params[21]
    g_LasRAI1 = params[22]
    D         = params[23]
    V_c       = (N*V_cell)/V_ext

    LasRAI1_ = (LasRAI1/K1)**h1
    RsaL_ = (RsaL/K2)**h2

    propensities[0]  = k_lasR                                                   # + lasR
    propensities[1]  = lasR*g_lasR                                              # - lasR
    propensities[2]  = lasR*k_LasR                                              # + LasR
    propensities[3]  = LasRAI1*s_LasRAI1                                        # + LasR ; + AI1 ; - LasRAI1
    propensities[4]  = AI1*LasR*u_LasRAI1                                       # - LasR ; - AI1 ; + LasRAI1
    propensities[5]  = LasR*g_LasR                                              # - LasR
    propensities[6]  = a_rsaL + b_rsaL*(LasRAI1_/(1+LasRAI1_))                  # + rsaL
    propensities[7]  = rsaL*g_rsaL                                              # - rsaL
    propensities[8]  = rsaL*k_RsaL                                              # + RsaL
    propensities[9]  = RsaL*g_RsaL                                              # - RsaL
    propensities[10] = a_lasI + (b_lasI*(LasRAI1_/((1+LasRAI1_)*(1+RsaL_))))    # + lasI
    propensities[11] = lasI*g_lasI                                              # - lasI
    propensities[12] = lasI*k_LasI                                              # + LasI
    propensities[13] = LasI*g_LasI                                              # - LasI
    propensities[14] = LasI*k_AI1                                               # + AI1
    propensities[15] = AI1*g_AI1                                                # - AI1
    propensities[16] = AI1*D                                                    # - AI1 ; "+ AI1_ext"
    propensities[17] = LasRAI1*g_LasRAI1                                        # - LasRAI1
    propensities[18] = AI1_ext*D*V_c                                            # + AI1 ; "- AI1_ext"

    Stot = propensities.sum()
    U = np.random.rand()
    Tau = np.random.exponential(scale=1/Stot)

    s = 0
    while s < len(propensities):
        if U < propensities[:s+1].sum()/Stot:
            return Tau, s
        s += 1

@njit()
def ensemble_simulation(N=N, AI1_ext=AI1_ext, x0=x0, params=params, reactions=reactions, t_max=t_max, dt_reg=dt_reg, ensemble_size=ensemble_size):

    t_size = round(t_max/dt_reg)
    time = np.zeros(t_size, dtype=np.float32)
    ensemble = np.zeros((ensemble_size, x0.size, t_size), dtype=np.int32)
    ensemble[:, :, 0] = x0.copy()

    cell = 0
    while cell < ensemble_size:
        t = 0
        x = x0.copy()
        i = 1
        while i < t_size:
            t_i = i*dt_reg
            while t < t_i:
                Tau, j = pa_qs1(x, params, N, AI1_ext, reactions)
                x += reactions[j]
                t += Tau
            ensemble[cell, :, i] = x
            time[i] = t_i
            i += 1
        cell += 1

    print(f'Simulation completed! \n Population size: {N} \n AI1_ext: {AI1_ext} \n Simulation time: {t_max} minutes \n Ensemble size: {ensemble_size}')
    return ensemble, time

@njit
def regularized_ensemble_stats(ensemble, time, ss=-1):

    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI1', 'LasRAI1']
    x_size = len(x)
    ensemble_size = ensemble.shape[0]
    t_size = time.size

    x_means = np.zeros((x_size, t_size), dtype=np.float64)
    x_stds = np.zeros((x_size, t_size), dtype=np.float64)
    for t in range(t_size):
        for i in range(x_size):
            x_ensemble = np.zeros(ensemble_size, dtype=np.int32)
            for n in range(ensemble_size):
                x_ensemble[n] = ensemble[n, i, t]
            x_means[i,t] = x_ensemble.mean()
            x_stds[i,t] = x_ensemble.std()

    time_ss = time[ss]
    x_ss = np.zeros((x_size, ensemble_size), dtype=np.int32)
    for i in range(x_size):
        for n in range(ensemble_size):
            x_ss[i,n] = ensemble[n,i,ss]

    x_ss_mean = np.zeros(x_size, dtype=np.float64)
    x_ss_std = np.zeros(x_size, dtype=np.float64)
    x_ss_noise = np.zeros(x_size, dtype=np.float64)
    for i in range(x_size):
        x_ss_mean[i] = x_ss[i].mean()
        x_ss_std[i] = x_ss[i].std()
        x_ss_noise[i] = 0 if x_ss[i].mean()==0 else x_ss_std[i]/x_ss[i].mean()

    return x_means, x_stds, time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise

def save_ensemble_ss_stats(x_ss, x_ss_mean, x_ss_std, x_ss_noise, N=N, AI1_ext=AI1_ext):

    ensemble_size = x_ss.shape[1]
    ss_stats = pd.DataFrame(np.array((x_ss_mean, x_ss_std, x_ss_noise)).T, \
            columns=['x_ss_mean', 'x_ss_std', 'x_ss_noise'])
    ss_stats.insert(0, 'x', ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI1', 'LasRAI1'])
    ss_stats.insert(0, 'AI1_ext', AI1_ext)
    ss_stats.set_index('AI1_ext', inplace=True)
    ss_stats.to_csv(f'PaQS1ss_{ensemble_size}_{N}_{AI1_ext}.tsv', sep='\t')
    del ss_stats

def plot_ensemble(ensemble, time, x_means, x_stds, N=N, AI1_ext=AI1_ext, action='display'):

    ensemble_size = ensemble.shape[0]
    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI$_1$', 'LasR$\cdot$AI$_1$']
    fig = plt.figure(figsize=(20,25))
    fig.subplots_adjust(hspace=0.3)
    subplot = 1
    for i in range(len(x)):
        plt.subplot(4,2,subplot)
        for j in ensemble:
            plt.plot(time, j[i], alpha=0.4)
        plt.plot(time, x_means[i], 'k', label='$\mu$', lw=2.5)
        plt.legend(loc=0)
        plt.xlabel('Time (s)')
        plt.ylabel(f'# {x[i]}')
        plt.title(f'{x[i]} evolution in ensemble')
        plt.grid()
        subplot += 1

    if action == 'save':
        fig.savefig(f'PaQS1_{ensemble_size}_{N}_{AI1_ext}.pdf', format='pdf', bbox_inches='tight')
    else:
        return fig

def plot_ensamble_ss(time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise, N=N, AI1_ext=AI1_ext, action='display'):

    ensemble_size = x_ss.shape[1]
    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI$_1$', 'LasR$\cdot$AI$_1$']
    fig = plt.figure(figsize=(20,25))
    fig.subplots_adjust(hspace=0.3)
    subplot = 1
    for i in range(len(x)):
        plt.subplot(4,2,subplot)
        plt.hist(x_ss[i], color='grey', label=f'$\mu=${np.around(x_ss_mean[i],2)} \n $\sigma=${np.around(x_ss_std[i],2)} \n $\eta=${np.around(x_ss_noise[i],2)}')
        plt.legend(loc=0)
        plt.xlabel(f'# {x[i]}')
        plt.ylabel('# Cells')
        plt.title(f'# {x[i]} at {np.around(time_ss)} minutes')
        plt.grid()
        subplot += 1

    if action == 'save':
        fig.savefig(f'PaQS1ss_{ensemble_size}_{N}_{AI1_ext}.pdf', format='pdf', bbox_inches='tight')
    else:
        return fig