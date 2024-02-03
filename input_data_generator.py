import math
import itertools
import numpy as np
import json
import random

other_params_points = 20

min_n_eff = 1.440
max_n_eff = 1.450
n_effs = np.linspace(min_n_eff, max_n_eff, other_params_points)

min_grating_period = 535e-9
max_grating_period = 540e-9
grating_periods = np.linspace(min_grating_period, max_grating_period, other_params_points)

min_X = 0.01
max_X = 0.99
X = random.uniform(min_X, max_X/2)

min_delta_n_eff = 1e-5
max_delta_n_eff = 1e-4
delta_n_effs = np.linspace(min_delta_n_eff, max_delta_n_eff, other_params_points)

cartesian_product = itertools.product(n_effs, grating_periods, delta_n_effs)

print("Saving to file")

result = []
for index, combination in enumerate(cartesian_product):
    n_eff, grating_period, delta_n_eff = combination
    result.append({
        "n_eff": n_eff,
        "grating_period": grating_period,
        "delta_n_eff": delta_n_eff
    })

with open("input.json", "w") as outfile:
    json.dump(result, outfile, indent=4)
