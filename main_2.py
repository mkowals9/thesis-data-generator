import json
import math
import matplotlib.pyplot as plt
import numpy as np



with open('input.json', 'r') as file:
    data = json.load(file)

L = 10 * 1e-3 # 10um w metrach
num_points = 1000
start_value = 1.35e-6 #początkowy zakres fal
end_value = 1.7e-6 #końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)

for single_case in data:
    k = single_case["mode_coupling_coef"]
    n_eff = single_case["n_eff"]
    period = single_case["grating_period"]
    bragg_wavelength = 2 * n_eff * period
    final = []
    for index, single_point in enumerate(wavelengths):
        ro = 2 * math.pi * n_eff * (1/single_point - 1/bragg_wavelength)
        k = ro + 0.0001
        gamma_b = math.sqrt(k**2 - ro**2)
        reflectance = (k**2 * math.sinh(gamma_b * L)**2) / (ro**2 * math.sinh(gamma_b * L)**2 + gamma_b**2 * math.cosh(gamma_b * L)**2)
        final.append(1 - reflectance)


    plt.plot(wavelengths, final)
    plt.xlabel("Wavelength")
    plt.ylabel("Response")
    plt.grid(True)
    plt.show()



