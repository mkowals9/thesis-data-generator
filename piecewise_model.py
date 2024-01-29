import json
import cmath

import numpy as np
import matplotlib.pyplot as plt

with open('input.json', 'r') as file:
    data = json.load(file)

L = 19 * 1e-4  # 1.9mm, tu w metrach
num_points = 1000
start_value = 1.5e-6  # początkowy zakres fal
end_value = 1.6e-6  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)

M = 10  # na ile sekcji dzielimy siatke
r_0 = 1
s_0 = 0
delta_z = L / M  # dlugosc odcinka i-tego siatki

for single_case in data:
    k = single_case["mode_coupling_coef"] / L
    n_eff = single_case["n_eff"]
    period = single_case["grating_period"]
    bragg_wavelength = 2 * n_eff * period
    # condition = (2 * n_eff * L) // bragg_wavelength
    final_reflectance = []
    R_0 = 1  # R - the forward propagating wave
    S_0 = 0  # S - the backward propagating wave

    for wavelength in wavelengths:
        all_R_matrices_per_wavelength = []
        all_S_matrices_per_wavelength = []
        sigma = 2 * cmath.pi * n_eff * (1 / wavelength - 1 / bragg_wavelength)
        for index in range(1, M):
            gamma_B = cmath.sqrt(k ** 2 - sigma ** 2)
            left_top = cmath.cosh(gamma_B * delta_z) - 1j * (sigma / gamma_B) * cmath.sinh(gamma_B * delta_z)
            left_down = 1j * (k / gamma_B) * cmath.sinh(gamma_B * delta_z)
            right_top = -1j * (k / gamma_B) * cmath.sinh(gamma_B * delta_z)
            right_down = cmath.cosh(gamma_B * delta_z) + 1j * (sigma / gamma_B) * cmath.sinh(gamma_B * delta_z)
            matrix_f_i = np.array([[left_top, right_top], [left_down, right_down]])
            if index != 1:
                R_i = matrix_f_i * all_R_matrices_per_wavelength[
                    index - 2]  # bo indeksy tablicy mamy od 0,1 a index leci od 1
                S_i = matrix_f_i * all_S_matrices_per_wavelength[index - 2]
                all_R_matrices_per_wavelength.append(R_i)
                all_S_matrices_per_wavelength.append(S_i)
            else:
                R_i = matrix_f_i * R_0
                S_i = matrix_f_i * S_0
                all_R_matrices_per_wavelength.append(R_i)
                all_S_matrices_per_wavelength.append(S_i)
        multiplied_R_matrices = all_R_matrices_per_wavelength[0]
        multiplied_S_matrices = all_S_matrices_per_wavelength[0]
        for matrix in all_R_matrices_per_wavelength:
            multiplied_R_matrices = np.dot(multiplied_R_matrices, matrix)
        for matrix in all_S_matrices_per_wavelength[1:]:
            multiplied_S_matrices = np.dot(multiplied_S_matrices, matrix)
        final_output_R = np.array(multiplied_R_matrices) * R_0
        final_output_S = np.array(multiplied_S_matrices) * S_0
        reflectance = np.abs(final_output_R[1, 0] / final_output_R[0, 0]) ** 2
        final_reflectance.append(reflectance)

    plt.plot(wavelengths, final_reflectance)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title("Reflectance")
    stats = (f'$k$ = {k:.2f}\n'
             f'$n_eff$ = {n_eff:.2f}\n'
             f'$period$ = {period:.2e}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(1.01, 0.95, stats, fontsize=12, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    plt.grid(True)
    plt.show()
