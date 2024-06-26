import json
import cmath
import numpy as np
from display_data import display_data

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

M = 200  # na ile sekcji dzielimy siatke
r_0 = 1
s_0 = 0
delta_z = L / M  # dlugosc odcinka i-tego siatki

for single_case in data:
    n_eff = single_case["n_eff"]
    period = single_case["grating_period"]
    delta_n_eff = single_case["delta_n_eff"]
    X_z = single_case["X_z"]
    bragg_wavelength = 2 * n_eff * period
    # condition_for_M = M < (2 * n_eff * L) // bragg_wavelength
    final_reflectance = []
    final_transmittance = []
    R_0 = 1  # R - the forward propagating wave
    S_0 = 0  # S - the backward propagating wave

    for wavelength in wavelengths:
        all_R_matrices_per_wavelength = []
        sigma = 2 * cmath.pi * n_eff * (1 / wavelength - 1 / bragg_wavelength) + (
                2 * cmath.pi / wavelength) * delta_n_eff
        kappa = (X_z * cmath.pi * fringe * delta_n_eff) / wavelength
        for index in range(1, M):
            gamma_B = cmath.sqrt(kappa ** 2 - sigma ** 2)
            left_top = cmath.cosh(gamma_B * delta_z) - 1j * (sigma / gamma_B) * cmath.sinh(gamma_B * delta_z)
            left_down = 1j * (kappa / gamma_B) * cmath.sinh(gamma_B * delta_z)
            right_top = -1j * (kappa / gamma_B) * cmath.sinh(gamma_B * delta_z)
            right_down = cmath.cosh(gamma_B * delta_z) + 1j * (sigma / gamma_B) * cmath.sinh(gamma_B * delta_z)
            matrix_f_i = np.array([[left_top, right_top], [left_down, right_down]])
            all_R_matrices_per_wavelength.append(matrix_f_i)
        multiplied_R_matrices = all_R_matrices_per_wavelength[0]
        for matrix_index in range(1, len(all_R_matrices_per_wavelength)):
            multiplied_R_matrices = np.matmul(all_R_matrices_per_wavelength[matrix_index], multiplied_R_matrices)
        final_output_R = np.array(multiplied_R_matrices) * R_0
        reflectance = np.abs(final_output_R[1, 0] / final_output_R[0, 0]) ** 2
        final_reflectance.append(reflectance)
        final_transmittance.append(1 - reflectance)

    ylabel = "Reflektancja"
    title = "Reflektancja - model macierzowy"
    display_data(wavelengths, final_reflectance, delta_n_eff, n_eff, period, ylabel, title, True, False, X_z)
    #
    # ylabel = "Transmittance"
    # title = "Transmittance - piecewise model"
    # display_data(wavelengths, final_transmittance, delta_n_eff, n_eff, period, ylabel, title, False, False, X_z)
    model_data.append({
        "reflectance": final_reflectance,
        "delta_n_eff": delta_n_eff,
        "n_eff": n_eff,
        "period": period,
        "X_z": X_z
    })

# TO SAVE CALCULATIONS AS JSON
with open(f"data_piecewise_model_input.json", "w") as outfile:
    json.dump(model_data, outfile, indent=4)
