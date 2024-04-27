import json
import cmath
import random
import numpy as np
from display_data import display_data, save_plots_to_one_gif
import os


with open('input_test.json', 'r') as file:
    data = json.load(file)

with open('model_config.json', 'r') as file:
    config = json.load(file)

L = config["L"]  # 4mm, tu w metrach
num_points = config["num_points"]
start_value = config["start_value"]  # początkowy zakres fal
end_value = config["end_value"]  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)
fringe = config["fringe"]

model_data = []

# quarter_length = len(data) // 9
# random.shuffle(data)
# data = data[:quarter_length]

for single_case in data:
    n_eff = single_case["n_eff"]
    period = single_case["grating_period"]
    delta_n_eff = single_case["delta_n_eff"]
    X_z = single_case["X_z"]
    bragg_wavelength = 2 * n_eff * period
    final_reflectance = []
    final_transmittance = []
    for index, single_point in enumerate(wavelengths):
        sigma = 2 * cmath.pi * n_eff * (1 / single_point - 1 / bragg_wavelength) + (
                2 * cmath.pi / single_point) * delta_n_eff
        # kappa = ((cmath.pi / single_point) * fringe * delta_n_eff)
        kappa = (X_z * cmath.pi * fringe * delta_n_eff) / single_point
        gamma_b = cmath.sqrt(kappa ** 2 - sigma ** 2)
        reflectance = (cmath.sinh(gamma_b * L) ** 2) / (
                cmath.cosh(gamma_b * L) ** 2 - sigma ** 2 / kappa ** 2)
        final_reflectance.append(reflectance.real)
        # final_transmittance.append(1 - reflectance.real)

    ylabel = "Reflectance"
    title = "Reflectance - numerical model"
    display_data(wavelengths, final_reflectance, delta_n_eff, n_eff, period, ylabel, title, False, False, X_z)

    # ylabel = "Transmittance"
    # title = "Transmittance - numerical model"
    # display_data(wavelengths, final_reflectance, delta_n_eff, n_eff, period, ylabel, title, True, False, X_z)

#     if max(final_reflectance) >= 0.1:
#         model_data.append({
#             "reflectance": final_reflectance,
#             "delta_n_eff": delta_n_eff,
#             "n_eff": n_eff,
#             "period": period,
#             "X_z": X_z
#         })
#
# # TO SAVE CALCULATIONS AS JSON
# with open(f"data_model_input.json", "w") as outfile:
#     json.dump(model_data, outfile, indent=4)


# TO SAVE PLOTS AS ONE GIF
# print("Creating gif")
# directory = './plots/'
# filenames = os.listdir(directory)

# save_plots_to_one_gif(filenames)


