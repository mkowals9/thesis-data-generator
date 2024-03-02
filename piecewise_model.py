import datetime
import json
import cmath
import random

import numpy as np
from display_data import display_data

with open('new_input_40_per_param.json', 'r') as file:
    data = json.load(file)

with open('model_config.json', 'r') as file:
    config = json.load(file)

N = 1000000 #ile przykladow (wykresow) chcemy
all_examples = []

L = config["L"]  # tu w metrach
num_points = config["num_points"]  # liczba dlugosci fal
start_value = config["start_value"]  # początkowy zakres fal
end_value = config["end_value"]  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)  # dlugosci fal
fringe = config["fringe"]

M = 50  # na ile sekcji dzielimy siatke
# condition_for_M = M < (2 * n_eff * L) // bragg_wavelength
delta_z = L / M  # dlugosc i-tego odcinka siatki

for example_index in range(N):
    ct = datetime.datetime.now().timestamp()
    ct = str(ct).replace(".", "_")
    # indeks danego elementu = indeks sekcji
    n_eff_all_sections = []
    delta_n_eff_all_sections = []
    X_z_all_sections = []
    period_all_sections = []

    R_0 = 1  # R - the forward propagating wave
    S_0 = 0  # S - the backward propagating wave

    # ustalamy parametry per sekcja
    for grating_section_index in range(0, M):
        random.seed(grating_section_index + M + random.random() + example_index)
        single_case = random.choice(data)

        n_eff_all_sections.append(single_case["n_eff"])
        delta_n_eff_all_sections.append(single_case["delta_n_eff"])
        X_z_all_sections.append(single_case["X_z"])
        period_all_sections.append(single_case["grating_period"])

    final_reflectance = []
    final_transmittance = []

    #reflektancja per długość fali
    for wavelength in wavelengths:

        all_R_matrices_per_wavelength = []
        for param_index in range(0, M):
            n_eff = n_eff_all_sections[param_index]
            delta_n_eff = delta_n_eff_all_sections[param_index]
            X_z = X_z_all_sections[param_index]
            grating_period = period_all_sections[param_index]
            bragg_wavelength = 2 * n_eff * grating_period

            sigma = 2 * cmath.pi * n_eff * (1 / wavelength - 1 / bragg_wavelength) + (
                    2 * cmath.pi / wavelength) * delta_n_eff
            kappa = (X_z * cmath.pi * fringe * delta_n_eff) / wavelength
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
        #final_transmittance.append(1 - reflectance)

    #ylabel = "Reflectance"
    #title = "Reflectance - piecewise model"
    #display_data(wavelengths, final_reflectance, ylabel, title, ct, False, False)
    #display_data(np.linspace(0, 50, 50), n_eff_all_sections, "n_eff", "n_eff per section", ct, False, False)
    all_examples.append({
            "wavelengths": wavelengths.tolist(),
            "reflectance": final_reflectance,
            "delta_n_eff": delta_n_eff_all_sections,
            "n_eff": n_eff_all_sections,
            "period": period_all_sections,
            "X_z": X_z_all_sections
        })
    print(f"Done example {example_index}")
    if example_index % 200000 == 0:
        with open(f"data_model_input_{example_index}.json", "w") as outfile:
            json.dump(all_examples, outfile, indent=4)
        all_examples = []

# TO SAVE CALCULATIONS AS JSON
with open(f"data_model_input_last.json", "w") as outfile:
    json.dump(all_examples, outfile, indent=4)

# ylabel = "Transmittance"
# title = "Transmittance - piecewise model"
# display_data(wavelengths, final_transmittance, delta_n_eff, n_eff, period, ylabel, title)
