#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt; plt.rc('font', size=16)
import pandas as pd
from numba import njit
from qs_stochastic import *

ensemble, time = ensemble_simulation(ensemble_size=1, t_max=1)
x_means, x_stds, time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise = regularized_ensemble_stats(ensemble, time)

AI1_ext = np.linspace(0, 8500, 10, dtype=int)

for i in range(len(AI1_ext)):
    ensemble, time = ensemble_simulation()
    x_means, x_stds, time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise = regularized_ensemble_stats(ensemble, time)
    save_ensemble_ss_stats(x_ss, x_ss_mean, x_ss_std, x_ss_noise, AI1_ext=AI1_ext[i])
    plot_ensemble(ensemble, time, x_means, x_stds, AI1_ext=AI1_ext[i], action='save')
    plot_ensamble_ss(time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise, AI1_ext=AI1_ext[i], action='save')
    del ensemble, time, x_means, x_stds, time_ss, x_ss, x_ss_mean, x_ss_std, x_ss_noise