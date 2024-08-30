"""
Module for performing likelihood estimation.

@author: Rikako Kono
"""

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import time

import SportVU_IO as sio
import SportVU_ControlField as spc 

N_GAMES = 64 # 64 for 4000 shot and 600 turnover sences
epsilon = 1e-10
version = "BIMOS" # "BIMOS" or "BMOS"
method = 'Nelder-Mead'
BMOS = True

initial_params = [7.0, 1.75, 30., 0.32, 0.32]   # [accel, kappa, lam, att_reaction_time, def_reaction_time]
bounds = [(1.0, 8.0), (1., None),(1., None), (0., 1.0), (0., 1.0)] 

def maximum_likelihood_estimation(accel, kappa, lam, att_reaction_time, def_reaction_time, n_game, version):
    params = spc.default_model_params(accel=accel, kappa=kappa, lam=lam,
                                      att_reaction_time=att_reaction_time, def_reaction_time=def_reaction_time)
    score_array, expected_OBSO, _ = sio.making_likelihood_dataset(params, n_game, version, BMOS=BMOS)
    obso_0 = sum(np.log(1-obso + epsilon) for i, obso in enumerate(expected_OBSO) if score_array[i] == 0)
    obso_1 = sum(np.log(obso + epsilon) for i, obso in enumerate(expected_OBSO) if score_array[i] == 1)
    return obso_0 + obso_1

def log_likelihood(params, n_game=N_GAMES):
    accel, kappa, lam, att_reaction_time, def_reaction_time = params
    return -maximum_likelihood_estimation(accel, kappa, lam, att_reaction_time, def_reaction_time, n_game, version)

start_time = time.time()

# minimize
result = minimize(log_likelihood,
                    initial_params, 
                    method = method, 
                    bounds = bounds
                    )
    
# gain optimal param
accel_best, kappa_best, lam_best, att_reaction_time_best, def_reaction_time_best = result.x
print(accel_best, kappa_best, lam_best, att_reaction_time_best, def_reaction_time_best)

# Display elapsed time
elapsed_time = time.time() - start_time
elapsed_time_minutes = elapsed_time / 60
print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")