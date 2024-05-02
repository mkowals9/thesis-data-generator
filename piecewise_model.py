import datetime
import json
import cmath
import os
import random

import numpy as np
from display_data import display_data, save_plots_to_one_gif, display_sections_data

with open('model_config.json', 'r') as file:
    config = json.load(file)

N = 250  # ile przykladow (wykresow) chcemy
L = config["L"]  # tu w metrach
num_points = config["num_points"]  # liczba dlugosci fal
start_value = config["start_value"]  # początkowy zakres fal
end_value = config["end_value"]  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)  # dlugosci fal
fringe = config["fringe"]

M = 15  # na ile sekcji dzielimy siatke
# condition_for_M = M < (2 * n_eff * L) // bragg_wavelength
delta_z = L / M  # dlugosc i-tego odcinka siatki
ct_prem = str(datetime.datetime.now().timestamp()).replace(".", "_")

all_examples_wavelengths = []
all_examples_reflectances = []
all_examples_delta_n_eff = []
all_examples_n_eff = []
all_examples_period = []
all_examples_X_z = []


def set_standard_params_per_section():
    random.seed(random.gauss())
    temp_n_eff_all_sections = []
    temp_delta_n_eff_all_sections = []
    temp_X_z_all_sections = []
    temp_period_all_sections = []
    with open('./input_data_generated/linear_input.json', 'r') as json_file:
        data = json.load(json_file)
    for section_index in range(0, M):
        random.seed(section_index + M + random.random() + example_index)
        single_case = random.choice(data)

        temp_n_eff_all_sections.append(single_case["n_eff"])
        temp_delta_n_eff_all_sections.append(single_case["delta_n_eff"])
        temp_X_z_all_sections.append(single_case["X_z"])
        temp_period_all_sections.append(single_case["grating_period"])
    return temp_n_eff_all_sections, temp_delta_n_eff_all_sections, temp_X_z_all_sections, temp_period_all_sections


def set_gauss_params_per_section():
    random.seed(random.gauss())
    temp_n_eff_all_sections = []
    temp_delta_n_eff_all_sections = []
    temp_X_z_all_sections = []
    temp_period_all_sections = []
    with open('./input_data_generated/new_input_40_gauss.json', 'r') as json_file:
        data = json.load(json_file)
    for section_index in range(0, M):
        random.seed(section_index + M + random.random() + example_index)
        single_case = random.choice(data)

        temp_n_eff_all_sections.append(single_case["n_eff"])
        temp_delta_n_eff_all_sections.append(single_case["delta_n_eff"])
        temp_X_z_all_sections.append(single_case["X_z"])
        temp_period_all_sections.append(single_case["grating_period"])
    return temp_n_eff_all_sections, temp_delta_n_eff_all_sections, temp_X_z_all_sections, temp_period_all_sections


def set_many_gauss_params_per_section():
    n_effs = np.load('./input_data_generated/gauss_600_n_eff.npy')
    delta_n_effs = np.load('./input_data_generated/gauss_600_delta_n_eff.npy')
    periods = np.load('./input_data_generated/gauss_600_period.npy')
    Xzs = np.load('./input_data_generated/gauss_600_X_z.npy')
    i = 2
    random.seed(random.gauss() + i + M + N)
    random_value_distr = random.randint(0, 11)
    random_value_array = random.randint(0, 4980)
    temp_n_eff_all_sections = n_effs[random_value_distr][random_value_array:random_value_array + M]
    random_value_distr = random.randint(0, 11)
    random_value_array = random.randint(0, 4980)
    temp_delta_n_eff_all_sections = delta_n_effs[random_value_distr][random_value_array:random_value_array + M]
    random_value_distr = random.randint(0, 11)
    random_value_array = random.randint(0, 4980)
    temp_X_z_all_sections = Xzs[random_value_distr][random_value_array:random_value_array + M]
    random_value_distr = random.randint(0, 11)
    random_value_array = random.randint(0, 4980)
    temp_period_all_sections = periods[random_value_distr][random_value_array:random_value_array + M]
    return temp_n_eff_all_sections, temp_delta_n_eff_all_sections, temp_X_z_all_sections, temp_period_all_sections


def set_parabolic_params_per_section():
    # n_effs = np.load('./input_data_generated/2nd_3rd_degree_n_eff.npy')
    # delta_n_effs = np.load('./input_data_generated/2nd_3rd_degree_delta_n_eff.npy')
    # periods = np.load('./input_data_generated/2nd_3rd_degree_period.npy')
    # Xzs = np.load('./input_data_generated/2nd_3rd_degree_X_z.npy')

    # n_effs = np.load('./input_data_generated/parabolic_n_eff_without_scaling.npy')
    # delta_n_effs = np.load('./input_data_generated/parabolic_delta_n_eff_without_scaling.npy')
    # periods = np.load('./input_data_generated/parabolic_period_without_scaling.npy')
    # Xzs = np.load('./input_data_generated/parabolic_X_z_without_scaling.npy')

    n_effs = np.load('./input_data_generated/parabolic_n_eff.npy')
    delta_n_effs = np.load('./input_data_generated/parabolic_delta_n_eff.npy')
    periods = np.load('./input_data_generated/parabolic_period.npy')
    Xzs = np.load('./input_data_generated/parabolic_X_z.npy')
    i = 2
    random.seed(random.gauss() + i + M + N)
    random_value_distr = random.randint(0, len(n_effs) - 1)
    temp_n_eff_all_sections = n_effs[random_value_distr]

    random.seed(random.gauss() + i + M + N)
    random_value_distr = random.randint(0, len(delta_n_effs) - 1)
    temp_delta_n_eff_all_sections = delta_n_effs[random_value_distr]

    random.seed(random.gauss() + i + M + N)
    random_value_distr = random.randint(0, len(Xzs) - 1)
    temp_X_z_all_sections = Xzs[random_value_distr]

    random.seed(random.gauss() + i + M + N)
    random_value_distr = random.randint(0, len(periods) - 1)
    temp_period_all_sections = periods[random_value_distr]

    return temp_n_eff_all_sections, temp_delta_n_eff_all_sections, temp_X_z_all_sections, temp_period_all_sections


def set_2nd_and_3rd_params_per_section():
    n_effs = np.load('./input_data_generated/2nd_3rd_degree_n_eff.npy')
    delta_n_effs = np.load('./input_data_generated/2nd_3rd_degree_delta_n_eff.npy')
    periods = np.load('./input_data_generated/2nd_3rd_degree_period.npy')
    Xzs = np.load('./input_data_generated/2nd_3rd_degree_X_z.npy')
    i = 2
    n = 20

    random.seed(random.gauss() + i + M + n)
    random_value_distr = random.randint(0, len(n_effs) - 1)
    temp_n_eff_all_sections = n_effs[random_value_distr]

    random.seed(random.gauss() + i + M + n)
    random_value_distr = random.randint(0, len(delta_n_effs) - 1)
    temp_delta_n_eff_all_sections = delta_n_effs[random_value_distr] * 1e-4

    random.seed(random.gauss() + i + n)
    random_value_distr = random.randint(0, len(Xzs) - 1)
    temp_X_z_all_sections = Xzs[random_value_distr] * 0.1

    random.seed(random.gauss() + i + n)
    random_value_distr = random.randint(0, len(periods) - 1)
    temp_period_all_sections = periods[random_value_distr] * 1e-7

    return temp_n_eff_all_sections, temp_delta_n_eff_all_sections, temp_X_z_all_sections, temp_period_all_sections


def set_positive_sin_params_per_section(i):
    n_effs = np.load('./input_data_generated/sin_n_eff.npy')
    delta_n_effs = np.load('./input_data_generated/sin_delta_n_eff.npy')
    periods = np.load('./input_data_generated/sin_period.npy')
    Xzs = np.load('./input_data_generated/sin_X_z.npy')
    n = 20

    random.seed(random.gauss() + i + M + n)
    random_value_distr = random.randint(0, len(n_effs) - 1)
    temp_n_eff_all_sections = n_effs[random_value_distr]

    random.seed(random.gauss() + i + M + n)
    random_value_distr = random.randint(0, len(delta_n_effs) - 1)
    temp_delta_n_eff_all_sections = delta_n_effs[random_value_distr] * 1e-4

    random.seed(random.gauss() + i + n)
    random_value_distr = random.randint(0, len(Xzs) - 1)
    temp_X_z_all_sections = Xzs[random_value_distr] * 0.1

    random.seed(random.gauss() + i + n)
    random_value_distr = random.randint(0, len(periods) - 1)
    temp_period_all_sections = periods[random_value_distr] * 1e-7

    return temp_n_eff_all_sections, temp_delta_n_eff_all_sections, temp_X_z_all_sections, temp_period_all_sections


for example_index in range(N):
    ct = str(datetime.datetime.now().timestamp()).replace(".", "_")
    # indeks danego elementu = indeks sekcji

    R_0 = 1  # R - the forward propagating wave
    S_0 = 0  # S - the backward propagating wave

    # ustalamy parametry per sekcja
    n_eff_all_sections, delta_n_eff_all_sections, X_z_all_sections, period_all_sections = (
        set_positive_sin_params_per_section(2))

    final_reflectance = []
    final_transmittance = []

    # reflektancja per długość fali
    for wavelength in wavelengths:

        all_R_matrices_per_wavelength = []
        for param_index in range(0, M - 1):
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
        reflectance = reflectance if reflectance >= 0.001 else 0
        final_reflectance.append(reflectance)
        # final_transmittance.append(1 - reflectance)

    if max(final_reflectance) > 0:
        ylabel = "Reflektancja"
        title = "Reflektancja - model macierzowy"
        display_data(wavelengths, final_reflectance, ylabel, title, ct, True, False)
        # display_sections_data(np.linspace(0, M - 1, M), X_z_all_sections,
        #                       "X_z", "X(z) dla kolejnych sekcji", ct, True)
        # display_sections_data(np.linspace(0, M - 1, M), period_all_sections,
        #                       "okres siatki", "Okresy siatki dla kolejnych sekcji", ct, True)
        # display_sections_data(np.linspace(0, M - 1, M), delta_n_eff_all_sections,
        #                       "delta_n_eff", "Delta_n_eff dla kolejnych sekcji", ct, True)
        # display_sections_data(np.linspace(0, M - 1, M), n_eff_all_sections,
        #                       "n_eff", "n_eff dla kolejnych sekcji", ct, True)
    # all_examples.append({
    #         "wavelengths": wavelengths.tolist(),
    #         "reflectance": final_reflectance,
    #         "delta_n_eff": delta_n_eff_all_sections,
    #         "n_eff": n_eff_all_sections,
    #         "period": period_all_sections,
    #         "X_z": X_z_all_sections
    #     })
    # all_examples_wavelengths.append(wavelengths)
    # all_examples_reflectances.append(final_reflectance)
    # all_examples_delta_n_eff.append(delta_n_eff_all_sections)
    # all_examples_period.append(period_all_sections)
    # all_examples_n_eff.append(n_eff_all_sections)
    # all_examples_X_z.append(X_z_all_sections)
    # # print(f"Done example {example_index}")
    # if example_index % 1000 == 0:
    #     print(f"Done example {example_index}")

# with open(f"model_input_wavelengths_{example_index}.json", "w") as outfile:
#     json.dump(all_examples, outfile, indent=4)
# all_examples = []
# np.save(f"./results_no_multi/model_input_wavelengths_{ct}.npy", np.array(all_examples_wavelengths))
# np.save(f"./results_no_multi/model_input_reflectances_{ct}.npy", np.array(all_examples_reflectances))
# np.save(f"./results_no_multi/model_input_delta_n_eff_{ct}.npy", np.array(all_examples_delta_n_eff))
# np.save(f"./results_no_multi/model_input_period_chunk_{ct}.npy", np.array(all_examples_period))
# np.save(f"./results_no_multi/model_input_n_eff_chunk_{ct}.npy", np.array(all_examples_n_eff))
# np.save(f"./results_no_multi/model_input_X_z_chunk_{ct}.npy", np.array(all_examples_X_z))

# all_examples_dict = {index: value for index, value in enumerate(all_examples)}
# # TO SAVE CALCULATIONS AS JSON
# with open(f"data_model_input_last.json", "w") as outfile:
#     json.dump(all_examples_dict, outfile, indent=4)

# ylabel = "Transmittance"
# title = "Transmittance - piecewise model"
# display_data(wavelengths, final_transmittance, delta_n_eff, n_eff, period, ylabel, title)

print("Creating gif")
directory = './plots/'
filenames = os.listdir(directory)

save_plots_to_one_gif(filenames)
