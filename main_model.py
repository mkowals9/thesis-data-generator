import json
import cmath
import matplotlib.pyplot as plt
import numpy as np

with open('input.json', 'r') as file:
    data = json.load(file)

L = 19 * 1e-4  # 1.9mm, tu w metrach
num_points = 1000
start_value = 1.5e-6  # początkowy zakres fal
end_value = 1.6e-6  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)

for single_case in data:
    k = single_case["mode_coupling_coef"] // L
    n_eff = single_case["n_eff"]
    period = single_case["grating_period"]
    bragg_wavelength = 2 * n_eff * period
    final_reflectance = []
    final_transmittance = []
    for index, single_point in enumerate(wavelengths):
        ro = 2 * cmath.pi * n_eff * (1 / single_point - 1 / bragg_wavelength)
        gamma_b = cmath.sqrt(k ** 2 - ro ** 2) if k ** 2 > ro ** 2 else 1j * cmath.sqrt(ro ** 2 - k ** 2)
        reflectance = (k ** 2 * cmath.sinh(gamma_b * L) ** 2) / (
                ro ** 2 * cmath.sinh(gamma_b * L) ** 2 + gamma_b ** 2 * cmath.cosh(gamma_b * L) ** 2)
        final_reflectance.append(reflectance)
        final_transmittance.append(1 - reflectance)

    plt.plot(wavelengths, final_reflectance)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title("Reflectance - numerical model")
    stats = (f'$k$ = {k:.2f}\n'
             f'$n_eff$ = {n_eff:.2f}\n'
             f'$period$ = {period:.2e}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(1.01, 0.95, stats, fontsize=12, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    plt.grid(True)
    plt.show()

    plt.plot(wavelengths, final_transmittance)
    plt.xlabel("Wavelength")
    plt.ylabel("Transmittance")
    plt.title("Transmittance")
    stats = (f'$k$ = {k:.2f}\n'
             f'$n_eff$ = {n_eff:.2f}\n'
             f'$period$ = {period:.2e}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(1.01, 0.95, stats, fontsize=12, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    plt.grid(True)
    plt.show()
