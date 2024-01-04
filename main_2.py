import json
import cmath
import matplotlib.pyplot as plt
import numpy as np

with open('input.json', 'r') as file:
    data = json.load(file)

L = 9 * 1e-4  # w metrach
num_points = 1000
start_value = 1.5e-6  # początkowy zakres fal
end_value = 1.6e-6  # końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)

for single_case in data:
    k = single_case["mode_coupling_coef"]
    n_eff = single_case["n_eff"]
    period = single_case["grating_period"]
    bragg_wavelength = 2 * n_eff * period
    final = []
    for index, single_point in enumerate(wavelengths):
        ro = 2 * cmath.pi * n_eff * (1 / single_point - 1 / bragg_wavelength)
        gamma_b = cmath.sqrt(k ** 2 - ro ** 2) if k ** 2 > ro ** 2 else 1j * cmath.sqrt(ro ** 2 - k ** 2)
        reflectance = (k ** 2 * cmath.sinh(gamma_b * L) ** 2) / (
                    ro ** 2 * cmath.sinh(gamma_b * L) ** 2 + gamma_b ** 2 * cmath.cosh(gamma_b * L) ** 2)
        final.append(1-reflectance)

    plt.plot(wavelengths, final)
    plt.xlabel("Wavelength")
    plt.ylabel("Transmittance")
    plt.grid(True)
    plt.show()
