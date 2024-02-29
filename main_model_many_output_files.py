import json
import cmath
import random
import numpy as np
from display_data import display_data, save_plots_to_one_gif
import os

PARAM_25_JSON = 'new_input_25_per_param.json'

PARAM_40_JSON = 'new_input_40_per_param.json'

INPUT_JSON = 'new_input.json'

#moze input jako tak?
#[[x1, y1],[x2,y2],(…)]

def divide_array(array, num_subarrays):
    # Calculate the size of each subarray
    subarray_size = len(array) // num_subarrays
    # Divide the array into subarrays
    subarrays = [array[i * subarray_size: (i + 1) * subarray_size] for i in range(num_subarrays)]
    # Handle the case where len(array) is not perfectly divisible by num_subarrays
    if len(array) % num_subarrays != 0:
        subarrays[-1] += array[num_subarrays * subarray_size:]

    return subarrays


with open(PARAM_40_JSON, 'r') as file:
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
random.shuffle(data)
subarrays = divide_array(data, 20)

for i, subarray in enumerate(subarrays):
    print(f"Subarray {i + 1}")
    for single_case in subarray:
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

        # ylabel = "Reflectance"
        # title = "Reflectance - numerical model"
        # display_data(wavelengths, final_reflectance, delta_n_eff, n_eff, period, ylabel, title, X_z)

        # ylabel = "Transmittance"
        # title = "Transmittance - numerical model"
        # display_data(wavelengths, final_transmittance, delta_n_eff, n_eff, period, ylabel, title, X_z)

        if max(final_reflectance) >= 0.1:
            model_data.append({
                "reflectance": final_reflectance,
                "delta_n_eff": delta_n_eff,
                "n_eff": n_eff,
                "period": period,
                "X_z": X_z
            })

    with open(f"data_model_40_param_input_with_X_z_{i}.json", "w") as outfile:
        json.dump(model_data, outfile, indent=4)
    model_data = []

# print("Creating gif")
# directory = './plots/'
# filenames = os.listdir(directory)

# save_plots_to_one_gif(filenames)

# with open("data_model_input_with_X_z.json", "w") as outfile:
#     json.dump(model_data, outfile, indent=4)
