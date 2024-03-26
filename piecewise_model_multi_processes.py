import datetime
import json
import cmath
import random
from multiprocessing import Process
import numpy as np

with open('model_config.json', 'r') as file:
    config = json.load(file)

N = 90000  # ile przykladow (wykresow) chcemy
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
            final_reflectance.append(reflectance)
            # final_transmittance.append(1 - reflectance)

        # ylabel = "Reflectance"
        # title = "Reflectance - piecewise model"
        # display_data(wavelengths, final_reflectance, ylabel, title, ct, False, False)
        # display_sections_data(np.linspace(0, 50, 50), X_z_all_sections,
        #                       "X_z", "X(z) per section", ct, False)
        # display_sections_data(np.linspace(0, 50, 50), period_all_sections,
        #                       "grating period", "grating period per section", ct,False)
        # display_sections_data(np.linspace(0, 50, 50), delta_n_eff_all_sections,
        #                       "delta_n_eff", "delta_n_eff per section", ct,False)
        # display_sections_data(np.linspace(0, 50, 50), n_eff_all_sections,
        #                       "n_eff", "n_eff per section", ct, False)
        # all_examples.append({
        #         "wavelengths": wavelengths.tolist(),
        #         "reflectance": final_reflectance,
        #         "delta_n_eff": delta_n_eff_all_sections,
        #         "n_eff": n_eff_all_sections,
        #         "period": period_all_sections,
        #         "X_z": X_z_all_sections
        #     })
        all_examples_wavelengths.append(wavelengths)
        all_examples_reflectances.append(final_reflectance)
        all_examples_delta_n_eff.append(delta_n_eff_all_sections)
        all_examples_period.append(period_all_sections)
        all_examples_n_eff.append(n_eff_all_sections)
        all_examples_X_z.append(X_z_all_sections)
        # print(f"Done example {example_index}")
        if example_index % 1000 == 0:
            print(f"Done example {example_index} on chunk {i}")

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


# dane sekcji sa wybierane jako parabola
def parabolic_calculate(i):
    # all_examples_wavelengths = []
    all_examples_reflectances = []
    all_examples_delta_n_eff = []
    all_examples_n_eff = []
    all_examples_period = []
    all_examples_X_z = []
    n_effs = np.load('./input_data_generated/parabolic_n_eff.npy')
    delta_n_effs = np.load('./input_data_generated/parabolic_delta_n_eff.npy')
    periods = np.load('./input_data_generated/parabolic_period.npy')
    Xzs = np.load('./input_data_generated/parabolic_X_z.npy')

    for example_index in range(N):
        R_0 = 1  # R - the forward propagating wave
        S_0 = 0  # S - the backward propagating wave

        # indeks danego elementu = indeks sekcji
        # ustalamy parametry per sekcja
        random.seed(random.gauss() + i + M + N)
        random_value_distr = random.randint(0, len(n_effs))
        n_eff_all_sections = n_effs[random_value_distr]

        random.seed(random.gauss() + i + M + N)
        random_value_distr = random.randint(0, len(delta_n_effs))
        delta_n_eff_all_sections = delta_n_effs[random_value_distr]

        random.seed(random.gauss() + i + M + N)
        random_value_distr = random.randint(0, len(Xzs))
        X_z_all_sections = Xzs[random_value_distr]

        random.seed(random.gauss() + i + M + N)
        random_value_distr = random.randint(0, len(periods))
        period_all_sections = periods[random_value_distr]
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
            final_reflectance.append(np.array([wavelength, reflectance]))

        # all_examples_wavelengths.append(wavelengths)
        if max([sublist[1] for sublist in final_reflectance]) > 0:
            all_examples_reflectances.append(final_reflectance)
            all_examples_delta_n_eff.append(delta_n_eff_all_sections)
            all_examples_period.append(period_all_sections)
            all_examples_n_eff.append(n_eff_all_sections)
            all_examples_X_z.append(X_z_all_sections)
            if example_index % 1000 == 0:
                print(f"Done example {example_index} on chunk {i}")

    # np.save(f"./results/parabol/model_input_wavelengths_chunk_2603{i}.npy", np.array(all_examples_wavelengths))
    np.save(f"./results/parabol/model_input_reflectances_chunk_2603{i}.npy", np.array(all_examples_reflectances))
    np.save(f"./results/parabol/model_input_delta_n_eff_chunk_2603{i}.npy", np.array(all_examples_delta_n_eff))
    np.save(f"./results/parabol/model_input_period_chunk_2603{i}.npy", np.array(all_examples_period))
    np.save(f"./results/parabol/model_input_n_eff_chunk_2603{i}.npy", np.array(all_examples_n_eff))
    np.save(f"./results/parabol/model_input_X_z_chunk_2603{i}.npy", np.array(all_examples_X_z))


if __name__ == '__main__':
    n_threads = 16
    threads = []
    for i in range(n_threads):
        t = Process(target=parabolic_calculate, args=(i,))
        t.start()
        threads.append(t)

    for i in range(n_threads):
        threads[i].join()
