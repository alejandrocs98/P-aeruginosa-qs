#!/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from numba import jit

# Default parameters
#-----------------------------------------------------------------------------------------------------------------------

N=10

t_max=10000

b_0=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)

e_0=np.array([0], dtype=np.int32)

b_update = np.array([[1, 0, 0, 0, 0, 0, 0, 0],    # + lasR
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
                     [0, 0, 0, 0, 0, 0, 1, 0],    # + AI1 ; - AI1_ext
                     [0, 0, 0, 0, 0, 0, 0, 0]],   # - AI1_ext
            dtype=np.int32)

# e_update = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1], dtype=np.int32)
e_update = np.array([[0],   # + lasR
                     [0],   # - lasR
                     [0],   # + LasR
                     [0],   # + LasR ; + AI1 ; - LasRAI1
                     [0],   # - LasR ; - AI1 ; + LasRAI1
                     [0],   # - LasR
                     [0],   # + rsaL
                     [0],   # - rsaL
                     [0],   # + RsaL
                     [0],   # - RsaL
                     [0],   # + lasI
                     [0],   # - lasI
                     [0],   # + LasI
                     [0],   # - LasI
                     [0],   # + AI1
                     [0],   # - AI1
                     [+1],  # - AI1 ; + AI1_ext
                     [0],   # - LasRAI1
                     [-1],  # + AI1 ; - AI1_ext
                     [-1]], # - AI1_ext
            dtype=np.int32)

k_lasR = 0.004    # 0
g_lasR = 0.002    # 1
k_LasR = 0.4      # 2
g_LasR = 0.35     # 3
a_rsaL = 0.00036  # 4
b_rsaL = 0.0058   # 5
K1 = 1.2          # 6
h1 = 1.4          # 7
g_rsaL = 0.001    # 8
k_RsaL = 0.9      # 9
g_RsaL = 0.2      # 10
a_lasI = 0.00036  # 11
b_lasI = 0.0058   # 12
K2 = 1.4          # 13
h2 = 1.2          # 14
g_lasI = 0.001    # 15
k_LasI = 0.7      # 16
g_LasI = 0.12     # 17
k_AI1 = 1         # 18
g_AI1 = 0.3       # 19
u_LasRAI1 = 0.05  # 20
s_LasRAI1 = 0.25  # 21
g_LasRAI1 = 0.14  # 22
d = 0.8           # 23

params_b = np.array([k_lasR, g_lasR, k_LasR, g_LasR, a_rsaL, b_rsaL, K1, h1, g_rsaL, k_RsaL, g_RsaL, a_lasI, \
                     b_lasI, K2, h2, g_lasI, k_LasI, g_LasI, k_AI1, g_AI1, u_LasRAI1, s_LasRAI1, g_LasRAI1, d], dtype=np.float32)

g_AI1_ext = 0.8  # 0
d = 0.8          # 1
d_away = 1.2     # 2

params_e = np.array([g_AI1_ext, d, d_away], dtype=np.float32)

b_file = 'bacteria.tsv'
e_file = 'environment.tsv'

# Functions
#-----------------------------------------------------------------------------------------------------------------------

@jit
def b_rules(x, params_b):
    
    lasR, LasR, rsaL, RsaL, lasI, LasI, AI1, LasRAI1 = x
    propensities = np.empty(18, dtype=np.float64)
    
    k_lasR    = params_b[0]
    g_lasR    = params_b[1]
    k_LasR    = params_b[2]
    g_LasR    = params_b[3]
    a_rsaL    = params_b[4]
    b_rsaL    = params_b[5]
    K1        = params_b[6]
    h1        = params_b[7]
    g_rsaL    = params_b[8]
    k_RsaL    = params_b[9]
    g_RsaL    = params_b[10]
    a_lasI    = params_b[11]
    b_lasI    = params_b[12]
    K2        = params_b[13]
    h2        = params_b[14]
    g_lasI    = params_b[15]
    k_LasI    = params_b[16]
    g_LasI    = params_b[17]
    k_AI1     = params_b[18]
    g_AI1     = params_b[19]
    u_LasRAI1 = params_b[20]
    s_LasRAI1 = params_b[21]
    g_LasRAI1 = params_b[22]
    d         = params_b[23]

    LasRAI1_ = (LasRAI1/K1 )**h1
    RsaL_ = (RsaL/K2 )**h2

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
    propensities[16] = d*AI1                                                    # - AI1 ; + AI1_ext
    propensities[17] = LasRAI1*g_LasRAI1                                        # - LasRAI1

    Stot = propensities.sum()
    U = np.random.rand()
    Tau = np.random.exponential(scale=1/Stot)

    s = 0
    while s < len(propensities):
        if U < propensities[:s+1].sum()/Stot:
            return Tau, s
        s += 1

@jit
def e_rules(x, params_e):
    
    AI1_ext = x[0]
    propensities = np.empty(2, dtype=np.float64)
    
    g_AI1_ext   = params_e[0]
    d           = params_e[1]
    d_away      = params_e[2]
    
    if AI1_ext == 0:                        
        return 0, 99  # no reaction
    else:
        propensities[0] = AI1_ext*d                     # + AI1 ; - AI1_ext
        propensities[1] = AI1_ext*(d_away+g_AI1_ext)    # - AI1_ext

        Stot = propensities.sum()
        U = np.random.rand()
        Tau = np.random.exponential(scale=1/Stot)

        s = 0
        while s < len(propensities):
            if U < propensities[:s+1].sum()/Stot:
                return Tau, 18+s
            s += 1

def save_bacteria(i, t, bacteria, b_file):
    
    with open(b_file, 'a') as bacteria_file:
        bacteria_file.write(f'{i}\t')
        bacteria_file.write(f'{t}\t')
        x = 0
        while x < len(bacteria[i]):
            if x == len(bacteria[i])-1:
                bacteria_file.write(f'{bacteria[i,x]}')
            else:
                bacteria_file.write(f'{bacteria[i,x]}\t')
            x += 1
        bacteria_file.write(f'\n')

def save_environment(t, env, e_file):

    with open(e_file, 'a') as env_file:
        env_file.write(f'{t}\t')
        x = 0
        while x < len(env):
            if x == len(env)-1:
                env_file.write(f'{env[x]}')
            else:
                env_file.write(f'{env[x]}\t')
            x += 1
        env_file.write(f'\n')

@jit
def waiting_time(reactions_sorted, h):
    
    if h > 0:
        return reactions_sorted[h,0] - reactions_sorted[h-1,0]
    else:
        return reactions_sorted[h,0]

def multicompartmental_gillespie(N=N, t_max=t_max, b_0=b_0, e_0=e_0, b_update=b_update, e_update=e_update, \
    params_b=params_b, params_e=params_e, b_file=b_file, e_file=e_file):
    
    if os.path.exists(b_file):
        os.remove(b_file)
    if os.path.exists(e_file):
        os.remove(e_file)
    
    # Initial conditions
    t = 0

    # Simulation initialization
    bacteria = np.zeros((N, len(b_0)), dtype=np.int64)
    i = 0
    while i < N:
        x = 0
        while x < len(b_0):
            bacteria[i,x] = b_0[x]
            x += 1
        i += 1
    bacteria_file = pd.DataFrame(bacteria, columns=['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI1', 'LasRAI1'])
    bacteria_file.insert(0, 't', np.zeros(N, dtype=np.int32))
    bacteria_file.insert(0, 'cell', np.array(range(N), dtype=np.int64))
    bacteria_file.set_index('cell', inplace=True)
    bacteria_file.to_csv(b_file, sep='\t')
    del bacteria_file

    env = np.zeros(len(e_0), dtype=np.int64)
    x = 0
    while x < len(e_0):
        env[x] = e_0[x]
        x += 1
    env_file = pd.DataFrame(env, columns=['AI1_ext'])
    env_file.insert(0, 't', np.zeros(1, dtype=np.int32))
    env_file.set_index('t', inplace=True)
    env_file.to_csv(e_file, sep='\t')
    del env_file
    
    # Simulation
    while t < t_max:
        reactions = np.zeros((N+1, 3), dtype=np.float64)  # Tau, j, i
        i = 0
        while i < N:
            reactions[i][0] = b_rules(bacteria[i], params_b)[0] # Tau
            reactions[i][1] = b_rules(bacteria[i], params_b)[1] # j
            reactions[i][2] = i                                 # i
            i += 1
        reactions[N][0] = e_rules(env, params_e)[0] # Tau
        reactions[N][1] = e_rules(env, params_e)[1] # j
        reactions[N][2] = N                         # i
        
        reactions_sorted = reactions[reactions[:,0].argsort()]

        h = 0
        while h < N+1:
            compartment = int(reactions_sorted[h,2])
            reaction = int(reactions_sorted[h,1])
            if compartment == N:
                if reaction == 99:
                    h += 1
                    continue
                else:
                    t += waiting_time(reactions_sorted, h)
                    reactions_sorted[h,0] = 0
                    env += e_update[reaction]
                    save_environment(t, env, e_file)
                    if reaction == 18:
                        i = np.random.randint(0,N)
                        bacteria[i] += b_update[reaction]
                        save_bacteria(i, t, bacteria, b_file)
                        # Replacement
                        y = 0
                        while y < N+1:
                            if reactions_sorted[y,2] == i:
                                if y > h:
                                    reactions_sorted[y,0] = b_rules(bacteria[i], params_b)[0]
                                    reactions_sorted[y,1] = b_rules(bacteria[i], params_b)[1]
                                    reactions_sorted = reactions_sorted[reactions_sorted[:,0].argsort()]
                            y += 1
            
            else:
                t += waiting_time(reactions_sorted, h)
                reactions_sorted[h,0] = 0
                i = compartment
                bacteria[i] += b_update[reaction]
                save_bacteria(i, t, bacteria, b_file)
                if reaction == 16:
                    env += e_update[reaction]
                    save_environment(t, env, e_file)
                    # Replacement
                    y = 0
                    while y < N+1:
                        if reactions_sorted[y,2] == N:
                            if y > h:
                                reactions_sorted[y,0] = e_rules(env, params_e)[0]
                                reactions_sorted[y,1] = e_rules(env, params_e)[1]
                                reactions_sorted = reactions_sorted[reactions_sorted[:,0].argsort()]
                        y += 1
                        
            h += 1

    print(f'Simulation completed! \n Cells: {N} \n Simulation time: {t_max} seconds')

def load_data(b_file=b_file, e_file=e_file):

    bacteria = pd.read_table(b_file, sep='\t', index_col=0)
    bacteria_grouped = [bacteria.loc[i] for i in bacteria.index.unique()]
    environment = pd.read_table(e_file, sep='\t', index_col=0)
    return bacteria_grouped, environment

def plot_mcg(bacteria_grouped, environment, action='display'):

    N = len(bacteria_grouped)
    x = ['lasR', 'LasR', 'rsaL', 'RsaL', 'lasI', 'LasI', 'AI1', 'LasRAI1']
    fig_b = plt.figure(figsize=(20,27))
    subplot = 1
    for i in x:
        plt.subplot(4,2,subplot)
        for j in bacteria_grouped:
            plt.plot(j['t'], j[i])
        plt.xlabel('Time (s)')
        plt.ylabel(i)
        plt.grid()
        subplot += 1

    fig_e = plt.figure(figsize=(9,5))
    plt.plot(environment.index, environment['AI1_ext'])
    plt.xlabel('Time (s)')
    plt.ylabel('AI1_ext')
    plt.grid()

    if action == 'save':
        fig_b.savefig(f'bacteria_{N}.png', bbox_inches='tight')
        fig_e.savefig('environment.png', bbox_inches='tight')
    else:
        return fig_b, fig_e