#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt; plt.rc('font', size=16)
import pandas as pd
from numba import njit
from qs_stochastic import *

N = 500

ensemble, time = ensemble_simulation(ensemble_size=1, t_max=1)
x_means, x_stds, time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise = regularized_ensemble_stats(ensemble, time)

AI1_ext = np.linspace(0, 300000, 25, dtype=int)

for i in range(len(AI1_ext)):
    ensemble, time = ensemble_simulation(N=N, AI1_ext=AI1_ext[i])
    x_means, x_stds, time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise = regularized_ensemble_stats(ensemble, time)
    save_ensemble_ss_stats(x_ss, x_ss_mean, x_ss_std, x_ss_noise, N=N, AI1_ext=AI1_ext[i])
    if i == 0:
        plot_ensemble(ensemble, time, x_means, x_stds, N=N, AI1_ext=AI1_ext[i], action='save')
        plot_ensamble_ss(time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise, N=N, AI1_ext=AI1_ext[i], action='save')
    elif i == len(AI1_ext)-1:
        plot_ensemble(ensemble, time, x_means, x_stds, N=N, AI1_ext=AI1_ext[i], action='save')
        plot_ensamble_ss(time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise, N=N, AI1_ext=AI1_ext[i], action='save')
    del ensemble, time, x_means, x_stds, time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise