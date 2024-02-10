import json
import cmath
import numpy as np
from display_data import display_data


with open('input_test.json', 'r') as file:
    data = json.load(file)

L = 4 * 1e-3  # 1.9mm, tu w metrach
num_points = 500
start_value = 1.5e-6  # początkowy zakres fal
end_value = 1.6e-6  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)

M = 100  # na ile sekcji dzielimy siatke
r_0 = 1
s_0 = 0
delta_z = L / M  # dlugosc odcinka i-tego siatki

fringe = 1

for single_case in data:
    n_eff = single_case["n_eff"]
    period = single_case["grating_period"]
    delta_n_eff = single_case["delta_n_eff"]
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
        k = (cmath.pi / wavelength) * fringe * delta_n_eff
        for index in range(1, M):
            gamma_B = cmath.sqrt(k ** 2 - sigma ** 2)
            left_top = cmath.cosh(gamma_B * delta_z) - 1j * (sigma / gamma_B) * cmath.sinh(gamma_B * delta_z)
            left_down = 1j * (k / gamma_B) * cmath.sinh(gamma_B * delta_z)
            right_top = -1j * (k / gamma_B) * cmath.sinh(gamma_B * delta_z)
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

    ylabel = "Reflectance"
    title = "Reflectance - piecewise model"
    display_data(wavelengths, final_reflectance, delta_n_eff, n_eff, period, ylabel, title)

    ylabel = "Transmittance"
    title = "Transmittance - piecewise model"
    display_data(wavelengths, final_transmittance, delta_n_eff, n_eff, period, ylabel, title)
