import math
import itertools
import numpy as np
import json
import random

other_params_points = 40

e_n_eff = 1.445
std_n_eff = 0.002
n_effs = np.random.normal(loc=e_n_eff, scale=std_n_eff, size=other_params_points)

e_grating_period = 537e-9
std_grating_period = 1e-9
grating_periods = np.random.normal(loc=e_grating_period, scale=std_grating_period, size=other_params_points)

e_delta_n_eff = 5e-4
std_delta_n_eff = 1e-4
delta_n_effs = np.random.normal(loc=e_delta_n_eff, scale=std_delta_n_eff, size=other_params_points)

e_X_z = 0.5
std_X_z = 0.1
X_z_s = np.random.normal(loc=e_X_z, scale=std_X_z, size=other_params_points)

cartesian_product = itertools.product(n_effs, grating_periods, delta_n_effs, X_z_s)

print("Saving to file")

result = []
for index, combination in enumerate(cartesian_product):
    n_eff, grating_period, delta_n_eff, X_z = combination
    result.append({
        "n_eff": n_eff,
        "grating_period": grating_period,
        "delta_n_eff": delta_n_eff,
        "X_z": X_z
    })

with open("./input_data_generated/new_input_40_gauss.json", "w") as outfile:
    json.dump(result, outfile, indent=4)
