import json
import cmath
import matplotlib.pyplot as plt
import numpy as np

with open('input_test.json', 'r') as file:
    data = json.load(file)

L = 4 * 1e-3  # 1.9mm, tu w metrach
num_points = 5000
start_value = 1.5e-6  # początkowy zakres fal
end_value = 1.6e-6  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)
fringe = 1

for single_case in data:
    #kappa = single_case["mode_coupling_coef"] // L
    n_eff = single_case["n_eff"]
    period = single_case["grating_period"]
    delta_n_eff = single_case["delta_n_eff"]
    bragg_wavelength = 2 * n_eff * period
    final_reflectance = []
    final_transmittance = []
    for index, single_point in enumerate(wavelengths):
        ro = 2 * cmath.pi * n_eff * (1 / single_point - 1 / bragg_wavelength) + (2*cmath.pi / single_point) * delta_n_eff
        kappa = ((cmath.pi / single_point) * fringe * delta_n_eff)
        gamma_b = cmath.sqrt(kappa ** 2 - ro ** 2)
        reflectance = (cmath.sinh(gamma_b * L) ** 2) / (
                cmath.cosh(gamma_b * L) ** 2 - ro**2/kappa**2)
        final_reflectance.append(reflectance)
        final_transmittance.append(1 - reflectance)

    plt.plot(wavelengths*1e9, final_reflectance)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title("Reflectance - numerical model")
    stats = (f'delta_n_eff = {delta_n_eff:.2e}\n'
             f'n_eff = {n_eff:.2f}\n'
             f'period = {period:.2e}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(0.95, 1.1, stats, fontsize=10, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    plt.grid(True)
    plt.show()

    plt.plot(wavelengths*1e9, final_transmittance)
    plt.xlabel("Wavelength")
    plt.ylabel("Transmittance")
    plt.title("Transmittance - numerical model")
    stats = (f'delta_n_eff = {delta_n_eff:.2e}\n'
             f'n_eff = {n_eff:.2f}\n'
             f'period = {period:.2e}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(0.95, 1.1, stats, fontsize=10, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    plt.grid(True)
    plt.show()
