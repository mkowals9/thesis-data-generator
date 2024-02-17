import json
import numpy as np
import os

from display_data import display_data, save_plots_to_one_gif

with open('/home/marcelina/Documents/misc/data_model_input_with_X_z_0.json', 'r') as file:
    data = json.load(file)

with open('model_config.json', 'r') as file:
    config = json.load(file)

want_create_plots = False

# TODO DOSTOSOWAĆ DO TEGO CO ZOSTAŁO WYGENEROWANE
num_points = config["num_points"]
start_value = config["start_value"]  # początkowy zakres fal
end_value = config["end_value"]
wavelengths = np.linspace(start_value, end_value, num_points)

if want_create_plots:
    for single_case in data:
        n_eff = single_case["n_eff"]
        period = single_case["period"]
        delta_n_eff = single_case["delta_n_eff"]
        X_z = single_case["X_z"]
        final_reflectance = single_case["reflectance"]
        ylabel = "Reflectance"
        title = "Reflectance - numerical model"
        display_data(wavelengths, final_reflectance, delta_n_eff, n_eff, period, ylabel, title, True, False, X_z)

print("Creating gif")
directory = './plots/'
filenames = os.listdir(directory)

save_plots_to_one_gif(filenames)
