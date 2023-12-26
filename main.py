import json
import math
import matplotlib.pyplot as plt
import numpy as np
def case_1_calculations(phi, k_g, L, wavelengths):
    # strona 19, slajd 37, P_f(z)
    result = []
    ro = math.sqrt(phi**2 - k_g**2)
    for single_point in wavelengths:
        result.append((ro ** 2 + (k_g ** 2 * (math.sin(ro * (single_point - L))) ** 2)) / (ro ** 2 + k_g ** 2 * (math.sin(ro * L)) ** 2))
    return result

def case_2_calculations(k_g, L, wavelengths):
    #strona 19, slajd 38, P_f(z)
    result = []
    for single_point in wavelengths:
        result.append((1 + pow(k_g, 2) * (single_point - L)**2 ) / (1 + k_g**2 * L**2))
    return result

def case_3_calculations(phi, k_g, L, wavelengths):
    #strona 20, slajd 39, P_f(z)
    result = []
    alpha = math.sqrt(pow(k_g, 2) - pow(phi, 2))
    for single_point in wavelengths:
        result.append((alpha**2 + k_g**2 * math.sinh(alpha * (single_point - L))**2) / (alpha**2 + k_g**2 * math.sinh(alpha*L)**2))
    return result




with open('input.json', 'r') as file:
    data = json.load(file)

L = 10 * 1e-6 # 10um w metrach
num_points = 1000
start_value = 1.35e-6 #początkowy zakres fal
end_value = 1.7e-6 #końcowy zakres fal
wavelengths = np.linspace(start_value, end_value, num_points)

for single_case in data:
    k_g = single_case["mode_coupling_coef"]
    n_eff = single_case["n_eff"]
    phi = single_case["phi_coef"]
    period = single_case["grating_period"]
    pass_band_1_condition = (math.pi/period - k_g) / n_eff
    pass_band_2_condition = (math.pi/period + k_g) / n_eff
    case_1_condition = abs(phi) > k_g
    case_2_condition = abs(phi) == k_g
    case_3_condition = abs(phi) < k_g
    transmitance_case_1 = case_1_calculations(phi, k_g, L, wavelengths)
    #transmitance_case_2 = case_2_calculations(k_g, L, wavelengths)
    #transmitance_case_3 = case_3_calculations(phi, k_g, L, wavelengths)
    final = []
    for index, single_point in enumerate(wavelengths):
        if single_point < pass_band_1_condition or single_point < pass_band_2_condition:
            final.append(transmitance_case_1[index])
            continue
        #final.append(transmitance_case_3[index])
        final.append(0)
    plt.plot(final)
    plt.xlabel("Wavelength")
    plt.ylabel("Response")
    plt.grid(True)
    plt.show()



