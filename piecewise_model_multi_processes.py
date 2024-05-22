import datetime
import json
import cmath
import random
from multiprocessing import Process
import numpy as np

import input_data_generator
from display_data import display_data, display_sections_data

with open('model_config.json', 'r') as file:
    config = json.load(file)

N = 120000  # ile przykladow (wykresow) chcemy
L = config["L"]  # tu w metrach
num_points = config["num_points"]  # liczba dlugosci fal
start_value = config["start_value"]  # początkowy zakres fal
end_value = config["end_value"]  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)  # dlugosci fal
fringe = config["fringe"]

n_effs = np.load('./input_data_generated/sin_n_eff_new_shifts_30_2105.npy')
delta_n_effs = np.load('./input_data_generated/sin_delta_n_eff_new_shifts_30_2105.npy')
periods = np.load('./input_data_generated/sin_period_new_shifts_30_2105.npy')
Xzs = np.load('./input_data_generated/sin_X_z_new_shifts_30_2105.npy')

M = 30  # na ile sekcji dzielimy siatke
# condition_for_M = M < (2 * n_eff * L) // bragg_wavelength
delta_z = L / M  # dlugosc i-tego odcinka siatki
ct_prem = str(datetime.datetime.now().timestamp()).replace(".", "_")


# dane dot. sekcji sa wybierane w sposob losowy z dostepnych
# przez co wykresy zmiennosci sa funkcjami schodkowymi
def basic_calculate(i):
    with open('./input_data_generated/new_input_40_gauss.json', 'r') as file:
        data = json.load(file)
    all_examples_wavelengths = []
    all_examples_reflectances = []
    all_examples_delta_n_eff = []
    all_examples_n_eff = []
    all_examples_period = []
    all_examples_X_z = []
    for example_index in range(N):
        ct = str(datetime.datetime.now().timestamp()).replace(".", "_")
        # indeks danego elementu = indeks sekcji
        n_eff_all_sections = []
        delta_n_eff_all_sections = []
        X_z_all_sections = []
        period_all_sections = []

        R_0 = 1  # R - the forward propagating wave
        S_0 = 0  # S - the backward propagating wave

        # ustalamy parametry per sekcja
        random.seed(random.gauss())
        for grating_section_index in range(0, M):
            # random.seed(grating_section_index + M + random.random() + example_index)
            single_case = random.choice(data)

            n_eff_all_sections.append(single_case["n_eff"])
            delta_n_eff_all_sections.append(single_case["delta_n_eff"])
            X_z_all_sections.append(single_case["X_z"])
            period_all_sections.append(single_case["grating_period"])

        final_reflectance = []
        final_transmittance = []

        # reflektancja per długość fali
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
            final_reflectance.append([wavelength, reflectance])
            # final_transmittance.append(1 - reflectance)

        if max([sublist[1] for sublist in final_reflectance]) > 0.1:
            ylabel = "Reflektancja"
            title = "Reflektancja - model macierzowy"
            ct_prem = str(datetime.datetime.now().timestamp()).replace(".", "_")
            display_data([el[0] for el in final_reflectance], [el[1] for el in final_reflectance],
                         ylabel, title, ct_prem, True, False)
            display_sections_data(np.linspace(0, M - 1, M), X_z_all_sections,
                                  "X_z", "X(z) dla kolejnych sekcji", ct_prem, True)
            display_sections_data(np.linspace(0, M - 1, M), period_all_sections,
                                  "okres siatki", "Okresy siatki dla kolejnych sekcji", ct_prem, True)
            display_sections_data(np.linspace(0, M - 1, M), delta_n_eff_all_sections,
                                  "delta_n_eff", "Delta_n_eff dla kolejnych sekcji", ct_prem, True)
            display_sections_data(np.linspace(0, M - 1, M), n_eff_all_sections,
                                  "n_eff", "n_eff dla kolejnych sekcji", ct_prem, True)

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
        #     print(f"Done example {example_index} on chunk {i}")

    # with open(f"model_input_wavelengths_{example_index}.json", "w") as outfile:
    #     json.dump(all_examples, outfile, indent=4)
    # all_examples = []
    np.save(f"./results/model_input_wavelengths_chunk_34{i}.npy", np.array(all_examples_wavelengths))
    np.save(f"./results/model_input_reflectances_chunk_34{i}.npy", np.array(all_examples_reflectances))
    np.save(f"./results/model_input_delta_n_eff_chunk_34{i}.npy", np.array(all_examples_delta_n_eff))
    np.save(f"./results/model_input_period_chunk_34{i}.npy", np.array(all_examples_period))
    np.save(f"./results/model_input_n_eff_chunk_34{i}.npy", np.array(all_examples_n_eff))
    np.save(f"./results/model_input_X_z_chunk_34{i}.npy", np.array(all_examples_X_z))

    # all_examples_dict = {index: value for index, value in enumerate(all_examples)}
    # # TO SAVE CALCULATIONS AS JSON
    # with open(f"data_model_input_last.json", "w") as outfile:
    #     json.dump(all_examples_dict, outfile, indent=4)

    # ylabel = "Transmittance"
    # title = "Transmittance - piecewise model"
    # display_data(wavelengths, final_transmittance, delta_n_eff, n_eff, period, ylabel, title)


# dane sekcji sa wybierane jako czesc rozkladu gaussa
def gauss_calculate(i):
    all_examples_wavelengths = []
    all_examples_reflectances = []
    all_examples_delta_n_eff = []
    all_examples_n_eff = []
    all_examples_period = []
    all_examples_X_z = []
    n_effs = np.load('./input_data_generated/gauss_600_n_eff.npy')
    delta_n_effs = np.load('./input_data_generated/gauss_600_delta_n_eff.npy')
    periods = np.load('./input_data_generated/gauss_600_period.npy')
    Xzs = np.load('./input_data_generated/gauss_600_X_z.npy')

    for example_index in range(N):
        R_0 = 1  # R - the forward propagating wave
        S_0 = 0  # S - the backward propagating wave

        # indeks danego elementu = indeks sekcji
        # ustalamy parametry per sekcja
        temp_ct = int(str(datetime.datetime.now().timestamp()).replace(".", ""))
        random.seed(random.gauss() + i + M + N + temp_ct)
        random_value_distr = random.randint(0, 11)
        random_value_array = random.randint(0, 4980)
        n_eff_all_sections = n_effs[random_value_distr][random_value_array:random_value_array + M]
        random_value_distr = random.randint(0, 11)
        random_value_array = random.randint(0, 4980)
        delta_n_eff_all_sections = delta_n_effs[random_value_distr][random_value_array:random_value_array + M]
        random_value_distr = random.randint(0, 11)
        random_value_array = random.randint(0, 4980)
        X_z_all_sections = Xzs[random_value_distr][random_value_array:random_value_array + M]
        random_value_distr = random.randint(0, 11)
        random_value_array = random.randint(0, 4980)
        period_all_sections = periods[random_value_distr][random_value_array:random_value_array + M]
        final_reflectance = []

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
            reflectance = reflectance if reflectance >= 0.01 else 0
            final_reflectance.append(reflectance)

        all_examples_wavelengths.append(wavelengths)
        all_examples_reflectances.append(final_reflectance)
        all_examples_delta_n_eff.append(delta_n_eff_all_sections)
        all_examples_period.append(period_all_sections)
        all_examples_n_eff.append(n_eff_all_sections)
        all_examples_X_z.append(X_z_all_sections)
        if example_index % 1000 == 0:
            print(f"Done example {example_index} on chunk {i}")

    np.save(f"./results/gauss/model_input_wavelengths_chunk_23033{i}.npy", np.array(all_examples_wavelengths))
    np.save(f"./results/gauss/model_input_reflectances_chunk_23033{i}.npy", np.array(all_examples_reflectances))
    np.save(f"./results/gauss/model_input_delta_n_eff_chunk_23033{i}.npy", np.array(all_examples_delta_n_eff))
    np.save(f"./results/gauss/model_input_period_chunk_23033{i}.npy", np.array(all_examples_period))
    np.save(f"./results/gauss/model_input_n_eff_chunk_23033{i}.npy", np.array(all_examples_n_eff))
    np.save(f"./results/gauss/model_input_X_z_chunk_23033{i}.npy", np.array(all_examples_X_z))


# dane sekcji sa wybierane jako wielomian 2 lub 3 stopnia
def polynomial_calculate(i):
    # all_examples_wavelengths = []
    all_examples_reflectances = []
    all_examples_delta_n_eff = []
    all_examples_n_eff = []
    all_examples_period = []
    all_examples_X_z = []
    n_effs = np.load('./input_data_generated/2nd_3rd_degree_n_eff.npy')
    delta_n_effs = np.load('./input_data_generated/2nd_3rd_degree_delta_n_eff.npy')
    periods = np.load('./input_data_generated/2nd_3rd_degree_period.npy')
    Xzs = np.load('./input_data_generated/2nd_3rd_degree_X_z.npy')

    for example_index in range(N):
        R_0 = 1  # R - the forward propagating wave
        # S_0 = 0  # S - the backward propagating wave

        # indeks danego elementu = indeks sekcji
        # ustalamy parametry per sekcja
        random.seed(random.gauss() + i + M + N)
        random_value_distr = random.randint(0, len(n_effs) - 1)
        n_eff_all_sections = n_effs[random_value_distr]

        random.seed(random.gauss() + i + M + N)
        random_value_distr = random.randint(0, len(delta_n_effs) - 1)
        delta_n_eff_all_sections = delta_n_effs[random_value_distr]

        random.seed(random.gauss() + i + M + N)
        random_value_distr = random.randint(0, len(Xzs) - 1)
        X_z_all_sections = Xzs[random_value_distr]

        random.seed(random.gauss() + i + M + N)
        random_value_distr = random.randint(0, len(periods) - 1)
        period_all_sections = periods[random_value_distr]
        final_reflectance = []

        # reflektancja per długość fali
        for wavelength in wavelengths:
            all_R_matrices_per_wavelength = []
            for param_index in range(0, M - 1):
                n_eff = n_eff_all_sections[param_index]
                delta_n_eff = delta_n_eff_all_sections[param_index] * 1e-4
                X_z = X_z_all_sections[param_index] * 1e-2
                grating_period = period_all_sections[param_index] * 1e-7
                bragg_wavelength = 2 * n_eff * grating_period

                sigma = 2 * cmath.pi * n_eff * (1 / wavelength - 1 / bragg_wavelength) + (
                        2 * cmath.pi / wavelength) * delta_n_eff
                # sigma = (X_z * 2 * cmath.pi * delta_n_eff) / wavelength
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
            reflectance = reflectance if reflectance >= 0.01 else 0
            final_reflectance.append(np.array([wavelength, reflectance]))

        # all_examples_wavelengths.append(wavelengths)
        if max([sublist[1] for sublist in final_reflectance]) > 0.1:
            # all_examples_reflectances.append(final_reflectance)
            # all_examples_delta_n_eff.append(delta_n_eff_all_sections)
            # all_examples_period.append(period_all_sections)
            # all_examples_n_eff.append(n_eff_all_sections)
            # all_examples_X_z.append(X_z_all_sections)
            # if example_index % 1000 == 0:
            #     print(f"Done example {example_index} on chunk {i}")
            ylabel = "Reflektancja"
            title = "Reflektancja - model macierzowy"
            display_data([el[0] for el in final_reflectance], [el[1] for el in final_reflectance],
                         ylabel, title, ct_prem + str(i), True, False)
            display_sections_data(np.linspace(0, M - 1, M), X_z_all_sections,
                                  "X_z", "X(z) dla kolejnych sekcji", "X_Z" + ct_prem + str(i), True)
            display_sections_data(np.linspace(0, M - 1, M), period_all_sections,
                                  "okres siatki", "Okresy siatki dla kolejnych sekcji", "okres" + ct_prem + str(i),
                                  True)
            display_sections_data(np.linspace(0, M - 1, M), delta_n_eff_all_sections,
                                  "delta_n_eff", "Delta_n_eff dla kolejnych sekcji", "delta_n_e" + ct_prem + str(i),
                                  True)
            display_sections_data(np.linspace(0, M - 1, M), n_eff_all_sections,
                                  "n_eff", "n_eff dla kolejnych sekcji", "n_eff" + ct_prem + str(i), True)

    # np.save(f"./results/parabol/model_input_wavelengths_chunk_2603{i}.npy", np.array(all_examples_wavelengths))
    # np.save(f"./results/polynomial/model_input_reflectances_chunk_704{i}.npy", np.array(all_examples_reflectances))
    # np.save(f"./results/polynomial/model_input_delta_n_eff_chunk_704{i}.npy", np.array(all_examples_delta_n_eff))
    # np.save(f"./results/polynomial/model_input_period_chunk_704{i}.npy", np.array(all_examples_period))
    # np.save(f"./results/polynomial/model_input_n_eff_chunk_704{i}.npy", np.array(all_examples_n_eff))
    # np.save(f"./results/polynomial/model_input_X_z_chunk_704{i}.npy", np.array(all_examples_X_z))


# do pliku zapisujemmy tylko współczynniki rownan 3 stopnia i reflektancje (x,y), nie n_eff, delta_n_eff, X_z, periods
def coefficients_calculate(i):
    all_examples_reflectances = []
    all_examples_coefficients = []

    for example_index in range(N):
        R_0 = 1  # R - the forward propagating wave
        # S_0 = 0  # S - the backward propagating wave

        # indeks danego elementu = indeks sekcji, mamy M sekcji
        x_values = np.linspace(-L / 2, L / 2, M)
        temp_coefficients = []
        # ustalamy parametry per sekcja, losujemy sobie współczynniki na podstawie wylosowanych współczynników

        # n_eff
        random.seed(random.gauss() + i + M + N)
        a = random.uniform(random.uniform(-1.5, 0.5), random.uniform(0.5, 2.0))
        b = random.uniform(random.uniform(-2, -0.25), random.uniform(i / 4, 4))
        c = random.uniform(random.uniform(-2.5, 1), random.uniform(i / 4 - 0.75, 4.5))
        d = random.uniform(random.uniform(-1.75, 0.3), random.uniform(i / 5 + 0.12, 4.44))
        n_eff_all_sections = input_data_generator.generate_polynomial([a, b, c, d], x_values, 1.44, 1.45)
        temp_coefficients.append(a)
        temp_coefficients.append(b)
        temp_coefficients.append(c)
        temp_coefficients.append(d)

        # delta_n_eff
        random.seed(random.gauss() + i + M + N)
        a = random.uniform(random.uniform(-1.75, 0.2), random.uniform(0.35, 3.0))
        b = random.uniform(random.uniform(-2.7, -0.12), random.uniform(i / 4 - 0.75, 4))
        c = random.uniform(random.uniform(-3.5, 0.1), random.uniform(i / 6 + 0.75, 4.2))
        d = random.uniform(random.uniform(-1.85, 0.11), random.uniform(i / 5 + 0.12, 3.91))
        delta_n_eff_all_sections = input_data_generator.generate_polynomial([a, b, c, d], x_values, 0.1, 1)
        temp_coefficients.append(a)
        temp_coefficients.append(b)
        temp_coefficients.append(c)
        temp_coefficients.append(d)

        # X_z
        random.seed(random.gauss() + i + M + N)
        a = random.uniform(random.uniform(-2.15, 0.65), random.uniform(0.75, 2.5))
        b = random.uniform(random.uniform(-2.9, -0.02), random.uniform(i / 4 - 0.95, 3.89))
        c = random.uniform(random.uniform(-2.75, 0.05), random.uniform(i / 6 + 0.75, 4.1))
        d = random.uniform(random.uniform(-2.15, 0.01), random.uniform(i / 5 + 0.18, 3.91))
        X_z_all_sections = input_data_generator.generate_polynomial([a, b, c, d], x_values, 0.1, 9.9)
        temp_coefficients.append(a)
        temp_coefficients.append(b)
        temp_coefficients.append(c)
        temp_coefficients.append(d)

        # periods
        random.seed(random.gauss() + i + M + N)
        a = random.uniform(random.uniform(-1.5, 0), random.uniform(0.01, 3.5))
        b = random.uniform(random.uniform(-3.2, -0.1), random.uniform(i / 4 - 1.2, 3.89))
        c = random.uniform(random.uniform(-3.15, 0.09), random.uniform(i / 6 + 1.25, 4.05))
        d = random.uniform(random.uniform(-2.87, 0.04), random.uniform(i / 5 + 1.18, 2.91))
        period_all_sections = input_data_generator.generate_polynomial([a, b, c, d], x_values, 5.350, 5.400)
        temp_coefficients.append(a)
        temp_coefficients.append(b)
        temp_coefficients.append(c)
        temp_coefficients.append(d)

        final_reflectance = []

        # reflektancja per długość fali
        for wavelength in wavelengths:
            all_R_matrices_per_wavelength = []
            for param_index in range(0, M - 1):
                n_eff = n_eff_all_sections[param_index]
                delta_n_eff = delta_n_eff_all_sections[param_index] * 1e-4
                X_z = X_z_all_sections[param_index] * 1e-2
                grating_period = period_all_sections[param_index] * 1e-7
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
            reflectance = reflectance if reflectance >= 0.0001 else 0
            final_reflectance.append(np.array([wavelength, reflectance]))

        if max([sublist[1] for sublist in final_reflectance]) > 0:
            all_examples_reflectances.append(final_reflectance)
            all_examples_coefficients.append(temp_coefficients)
            if example_index % 5000 == 0:
                print(f"Done example {example_index} on chunk {i}")

    np.save(f"./results/coefficients/model_input_reflectances_chunk_404{i}.npy", np.array(all_examples_reflectances))
    np.save(f"./results/coefficients/model_input_coefficients_chunk_404{i}.npy", np.array(all_examples_coefficients))


def positive_sin_calculate(i):
    all_examples_reflectances = []
    all_examples_delta_n_eff = []
    all_examples_n_eff = []
    all_examples_period = []
    all_examples_X_z = []
    for example_index in range(N):
        # ct = str(datetime.datetime.now().timestamp()).replace(".", "_")
        # indeks danego elementu = indeks sekcji

        R_0 = 1  # R - the forward propagating wave
        # S_0 = 0  # S - the backward propagating wave

        # ustalamy parametry per sekcja

        n = M + i - 3

        random.seed(random.gauss() + i + M + n)
        random_value_distr = random.randint(0, len(n_effs) - 1)
        n_eff_all_sections = n_effs[random_value_distr]

        random.seed(random.gauss() + i + M - n)
        random_value_distr = random.randint(0, len(delta_n_effs) - 1)
        delta_n_eff_all_sections = delta_n_effs[random_value_distr]

        random.seed(random.gauss() + i + M)
        random_value_distr = random.randint(0, len(Xzs) - 1)
        X_z_all_sections = Xzs[random_value_distr]

        random.seed(random.gauss() + i - M)
        random_value_distr = random.randint(0, len(periods) - 1)
        period_all_sections = periods[random_value_distr]

        final_reflectance = []
        # final_transmittance = []

        # reflektancja per długość fali
        for wavelength in wavelengths:

            all_R_matrices_per_wavelength = []
            for param_index in range(0, M):
                n_eff = n_eff_all_sections[param_index]
                delta_n_eff = delta_n_eff_all_sections[param_index] * 1e-4
                X_z = X_z_all_sections[param_index] * 1e-1
                grating_period = period_all_sections[param_index] * 1e-7
                # if not (1e-5 <= delta_n_eff <= 1e-4):
                #     print("Bad scaling delta_n_eff: ", delta_n_eff)
                # if not (535e-9 <= grating_period <= 540e-9):
                #     print("Bad scaling period: ", grating_period)
                # if not (0.01 <= X_z <= 0.99):
                #     print(f"Bad scaling X_z: {X_z}")

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
            final_reflectance.append([wavelength, reflectance])
            # final_transmittance.append(1 - reflectance)

        max_value = max([sublist[1] for sublist in final_reflectance])
        if max_value > 0.005:
            # all_examples_wavelengths.append(wavelengths)
            all_examples_reflectances.append(final_reflectance)
            all_examples_delta_n_eff.append(delta_n_eff_all_sections)
            all_examples_period.append(period_all_sections)
            all_examples_n_eff.append(n_eff_all_sections)
            all_examples_X_z.append(X_z_all_sections)

        # if max_value > 0.1:
        #     ylabel = "Reflektancja"
        #     title = "Reflektancja - model macierzowy"
        #     ct_prem = str(datetime.datetime.now().timestamp()).replace(".", "_")
        #     display_data([el[0] for el in final_reflectance], [el[1] for el in final_reflectance],
        #                  ylabel, title, ct_prem, True, False)
        #     display_sections_data(np.linspace(0, M - 1, M), X_z_all_sections,
        #                           "X_z", "X(z) dla kolejnych sekcji", ct_prem, True)
        #     display_sections_data(np.linspace(0, M - 1, M), period_all_sections,
        #                           "okres siatki", "Okresy siatki dla kolejnych sekcji", ct_prem, True)
        #     display_sections_data(np.linspace(0, M - 1, M), delta_n_eff_all_sections,
        #                           "delta_n_eff", "Delta_n_eff dla kolejnych sekcji", ct_prem, True)
        #     display_sections_data(np.linspace(0, M - 1, M), n_eff_all_sections,
        #                           "n_eff", "n_eff dla kolejnych sekcji", ct_prem, True)

        # all_examples.append({
        #         "wavelengths": wavelengths.tolist(),
        #         "reflectance": final_reflectance,
        #         "delta_n_eff": delta_n_eff_all_sections,
        #         "n_eff": n_eff_all_sections,
        #         "period": period_all_sections,
        #         "X_z": X_z_all_sections
        #     })

        # print(f"Done example {example_index}")
        if example_index % 1000 == 0:
            print(f"Done example {example_index} on chunk {i}")

    # np.save(f"./results/model_input_wavelengths_chunk_15{i}.npy", np.array(all_examples_wavelengths))
    np.save(f"./results/sinusoid/model_input_reflectances_chunk_2105{i}.npy", np.array(all_examples_reflectances))
    np.save(f"./results/sinusoid/model_input_delta_n_eff_chunk_2105{i}.npy", np.array(all_examples_delta_n_eff))
    np.save(f"./results/sinusoid/model_input_period_chunk_2105{i}.npy", np.array(all_examples_period))
    np.save(f"./results/sinusoid/model_input_n_eff_chunk_2105{i}.npy", np.array(all_examples_n_eff))
    np.save(f"./results/sinusoid/model_input_X_z_chunk_2105{i}.npy", np.array(all_examples_X_z))


if __name__ == '__main__':
    n_threads = 17
    threads = []
    for i in range(n_threads):
        t = Process(target=positive_sin_calculate, args=(i,))
        t.start()
        threads.append(t)

    for i in range(n_threads):
        threads[i].join()
