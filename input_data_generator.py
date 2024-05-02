import random
import traceback

import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
import time


# n_eff
# min 1.44
# max 1.45

# delta_n_eff
# min 1e-5
# max 1e-4

# period
# min 535e-9
# max 540e-9

# X(z)
# min 0.01
# max 0.99

def generate_linear():
    other_params_points = 40

    min_n_eff = 1.440
    max_n_eff = 1.450
    n_effs = np.linspace(min_n_eff, max_n_eff, other_params_points)

    min_grating_period = 5.350
    max_grating_period = 5.400
    grating_periods = np.linspace(min_grating_period, max_grating_period, other_params_points)

    min_delta_n_eff = 0.1
    max_delta_n_eff = 1
    delta_n_effs = np.linspace(min_delta_n_eff, max_delta_n_eff, other_params_points)

    min_X_z = 0.1
    max_X_z = 9.9
    X_z_s = np.linspace(min_X_z, max_X_z, other_params_points)

    cartesian_product = itertools.product(n_effs, grating_periods, delta_n_effs, X_z_s)

    print("Saving to file")

    result = []
    for index, combination in enumerate(cartesian_product):
        n_eff, grating_period, delta_n_eff, X_z = combination
        result.append({
            "n_eff": n_eff,
            "grating_period": grating_period,
            "delta_n_eff": delta_n_eff,
            "X_z": X_z
        })

    with open("./input_data_generated/linear_input.json", "w") as outfile:
        json.dump(result, outfile, indent=4)


def generate_standard():
    other_params_points = 40

    e_n_eff = 1.445
    std_n_eff = 0.002
    n_effs = np.random.normal(loc=e_n_eff, scale=std_n_eff, size=other_params_points)

    e_grating_period = 537e-9
    std_grating_period = 1e-9
    grating_periods = np.random.normal(loc=e_grating_period, scale=std_grating_period, size=other_params_points)

    e_delta_n_eff = 5e-4
    std_delta_n_eff = 1e-4
    delta_n_effs = np.random.normal(loc=e_delta_n_eff, scale=std_delta_n_eff, size=other_params_points)

    e_X_z = 0.5
    std_X_z = 0.1
    X_z_s = np.random.normal(loc=e_X_z, scale=std_X_z, size=other_params_points)

    cartesian_product = itertools.product(n_effs, grating_periods, delta_n_effs, X_z_s)

    print("Saving to file")

    result = []
    for index, combination in enumerate(cartesian_product):
        n_eff, grating_period, delta_n_eff, X_z = combination
        result.append({
            "n_eff": n_eff,
            "grating_period": grating_period,
            "delta_n_eff": delta_n_eff,
            "X_z": X_z
        })

    with open("./input_data_generated/new_input_40_gauss.json", "w") as outfile:
        json.dump(result, outfile, indent=4)


def generate_many_gauss_distributions():
    other_params_points = 5000

    e_n_effs = [1.445, 1.442, 1.447]
    std_n_effs = [0.002, 0.001, 0.003, 0.004]
    cartesian_product_n_eff = itertools.product(e_n_effs, std_n_effs)
    n_effs = []
    for index, combination in enumerate(cartesian_product_n_eff):
        e_n_eff, std_n_eff = combination
        n_effs_distr = np.random.normal(loc=e_n_eff, scale=std_n_eff, size=other_params_points)
        plt.hist(n_effs_distr, 100)
        # plotting mean line
        plt.axvline(e_n_eff, color='k', linestyle='dashed', linewidth=2)
        plt.show()
        n_effs.append(n_effs_distr)

    e_grating_periods = [537e-9, 536e-9, 538e-9]
    std_grating_periods = [1e-9, 2e-9, 5e-9, 7e-9]
    cartesian_product_period = itertools.product(e_grating_periods, std_grating_periods)
    grating_periods = []
    for index, combination in enumerate(cartesian_product_period):
        e_period, std_period = combination
        periods_distr = np.random.normal(loc=e_period, scale=std_period, size=other_params_points)
        grating_periods.append(periods_distr)

    e_delta_n_effs = [5e-4, 6e-4, 4e-4, 8e-4]
    std_delta_n_effs = [1e-4, 5e-3, 2e-4]
    cartesian_product_delta_n_eff = itertools.product(e_delta_n_effs, std_delta_n_effs)
    delta_n_effs = []
    for index, combination in enumerate(cartesian_product_delta_n_eff):
        e_delta_n_eff, std_delta_n_eff = combination
        delta_n_effs_distr = np.random.normal(loc=e_delta_n_eff, scale=std_delta_n_eff, size=other_params_points)
        delta_n_effs.append(delta_n_effs_distr)

    e_X_zs = [0.5, 0.2, 0.75, 0.35]
    std_X_zs = [0.1, 0.05, 0.15]
    cartesian_product_X_z = itertools.product(e_X_zs, std_X_zs)
    X_zs = []
    for index, combination in enumerate(cartesian_product_X_z):
        e_X_z, std_X_z = combination
        X_z_distr = np.random.normal(loc=e_X_z, scale=std_X_z, size=other_params_points)
        X_zs.append(X_z_distr)

    print("Saving to file")
    np.save("./input_data_generated/gauss_600_n_eff.npy", np.asarray(n_effs))
    np.save("./input_data_generated/gauss_600_period.npy", np.asarray(grating_periods))
    np.save("./input_data_generated/gauss_600_delta_n_eff.npy", np.asarray(delta_n_effs))
    np.save("./input_data_generated/gauss_600_X_z.npy", np.asarray(X_zs))


def plot_normalized(x_values, normalized_values):
    plt.plot(x_values, normalized_values, marker='o', linestyle='-')
    plt.xlabel('X values')
    plt.ylabel('Normalized values')
    plt.title('Normalized Values Plot')
    plt.grid(True)
    plt.show()


def generate_polynomial(coeffs, x_values, desired_min, desired_max):
    values = np.polyval(coeffs, x_values)
    # if int(time.time()) % 3 == 0:
    #     noise_for_values = generate_noise(15)
    #     values = [value + noise_for_values[i] if i % 2 == 0 else value for i, value in enumerate(values)]
    min_value = np.min(values)
    max_value = np.max(values)
    value_ = (max_value - min_value)
    if value_ != 0:
        normalized_values = [(value - min_value) / (max_value - min_value) * (desired_max - desired_min) + desired_min
                             for value in values]
        normalized_values = [desired_min if value < desired_min else value for value in normalized_values]
        normalized_values = [desired_max if value > desired_max else value for value in normalized_values]
        # plot_normalized(x_values, normalized_values)
        return np.array(normalized_values)


def generate_parabolic_coefficients(num_coefficients, p, q, range_a, seed=None):
    rng = np.random.default_rng(seed)
    a_values = np.append(np.round(rng.uniform(range_a[0], range_a[1], num_coefficients), 4), 0)
    # b_values = np.append(np.round(np.array([-p * 2 * a + noise[a_index] if a_index % 3 == 0 else -p * 2 * a
    #                                         for a_index, a in enumerate(a_values)]), 4), 0)
    # c_values = np.append(np.round(np.unique(np.array([(4 * a * q + b_values[a_index] ** 2) / (4 * a) if a != 0 else 0
    #                                                   for a_index, a in enumerate(a_values)])), 4), 0)
    b_values = np.append(np.round(rng.uniform(range_a[0] - 0.25, range_a[1] + 0.82, num_coefficients), 4), 0)
    c_values = np.append(np.round(rng.uniform(range_a[0] - 0.17, range_a[1] + 0.29, num_coefficients), 4), 0)
    return list(zip(a_values, b_values, c_values))


def generate_3_polynomical_coefficients(num_coefficients, p, q, range_a, range_c, seed=None):
    rng = np.random.default_rng(seed)
    a_values = np.round(rng.uniform(range_a[0], range_a[1], num_coefficients), 4)
    c_values = np.round(rng.uniform(range_c[0], range_c[1], num_coefficients), 4)
    noise = generate_noise(num_coefficients)
    # b_values = np.round(np.array([-p * 3 * a + noise[a_index] if a_index % 3 == 0 else -p * 3 * a
    # for a_index, a in enumerate(a_values)]), 4)
    # d_values = np.round(np.unique(np.array([(q - (p ** 3) * a - b_values[a_index] * (p ** 2) - p * c_values[a_index])
    #          if a != 0 else 0 for a_index, a in enumerate(a_values)])), 4)
    b_values = np.round(rng.uniform(range_a[0] - 0.5, range_a[1] + 0.82, num_coefficients), 4)
    d_values = np.round(rng.uniform(range_c[0] - 0.15, range_c[1] + 0.24, num_coefficients), 4)
    return list(zip(a_values, b_values, c_values, d_values))


def generate_noise(num_points):
    return np.random.uniform(1e-2, 1e-1, size=num_points)


def generate_3rd_degree_arrays(num_points, range_a, range_c, x_values, y_max, y_min):
    temp_result = []
    vertex_values = np.linspace(y_min, y_max, num_points)
    cartesian_product = itertools.product(x_values, vertex_values)
    for index, combination in enumerate(cartesian_product):
        p, q = combination
        coefficients_polynomial = generate_3_polynomical_coefficients(2000, p, q,
                                                                      range_a, range_c, int(time.time()))
        for a, b, c, d in coefficients_polynomial:
            if (a is not None and a != 0) and (b is not None and b != 0) and (c is not None and c != 0):
                generated = np.round(generate_polynomial([a, b, c, d], x_values, y_min, y_max), 3)
            # if index % 3 == 0:
            #     generated = np.round(generated + generate_noise(num_points), 3)
            #     plot_normalized(x_values, generated)
            # if index % 5 == 0:
            #     generated = np.round(generated - generate_noise(num_points), 3)
            # generated = [x if x > 0 else (-1) * x for x in generated]

            # generated_1 = np.flipud(generated)
            # generated_2 = np.array([2 * medium - y if y > medium else (2 * medium - y) for y in generated])
            # generated_3 = np.array([2 * medium - y if y > medium else (2 * medium - y) for y in generated_1])
            # plot_normalized(x_values, generated)
            # plot_normalized(x_values, generated_1)
            # plot_normalized(x_values, generated_3)
            # plot_normalized(x_values, generated_1)
            # temp_result.append(generated_2)
            # temp_result.append(generated_3)
            # temp_result.append(np.array(generated_1))
            temp_result.append(generated)
    return np.unique(temp_result, axis=0)


def generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min):
    temp_result = []
    #medium = (y_min + y_max) / 2
    vertex_values = np.linspace(y_min, y_max, num_points)
    cartesian_product = itertools.product(x_values, vertex_values)
    for index, combination in enumerate(cartesian_product):
        p, q = combination
        coefficients_parabolic = generate_parabolic_coefficients(2000, p, q, range_a, int(time.time()))
        for a, b, c in coefficients_parabolic:
            try:
                if (a is not None and a != 0) and (b is not None and b != 0) and (c is not None and c != 0):
                    generated = generate_polynomial([a, b, c], x_values, y_min, y_max)
                    generated = np.round(generated, 3)
                    temp_result.append(generated)
            except Exception as e:
                traceback.print_exc()
                print(f"Error {a}, {b}, {c}, {generated}: {e}")
            # if index % 3 == 0:
            #     generated = np.round(generated + generate_noise(num_points), 3)
            # if index % 5 == 0:
            #     generated = np.round(generated - generate_noise(num_points), 3)
            # generated = [x if x > 0 else (-1) * x for x in generated]
            # temp_result.append(np.array(generated_1))
    return np.unique(temp_result, axis=0)


def generate_parabolic_distributions():
    with open('model_config.json', 'r') as file:
        config = json.load(file)

    L = config["L"] * 1e2
    num_points = 15
    x_min = -L / 2
    x_max = L / 2
    x_values = np.linspace(x_min, x_max, num_points)

    y_min = 1.440
    y_max = 1.450
    random.seed(int(time.time()))
    range_a = (random.uniform(-5, 0), random.uniform(0, 5))
    n_effs = generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min)
    print("Done n_effs")

    y_min = 5.350  # 535e-9
    y_max = 5.400  # 540e-9
    random.seed(int(time.time()))
    range_a = (random.uniform(-2, 0), random.uniform(0, 2))
    grating_periods = generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min)
    print("Done grating_periods")

    y_min = 0.1  # 1e-5
    y_max = 1  # 1e-4
    random.seed(int(time.time()))
    range_a = (random.uniform(-1.5, 0), random.uniform(0, 1.5))
    delta_n_effs = generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min)
    print("Done delta_n_effs")

    y_min = 0.1  # 0.01
    y_max = 9.9  # 0.99
    random.seed(int(time.time()))
    range_a = (random.uniform(-1.2, 0), random.uniform(0, 1.6))
    X_zs = generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min)
    print("Done X_z_s")

    print("Saving to file")
    np.save("./input_data_generated/parabolic_n_eff.npy", np.asarray(n_effs))
    np.save("./input_data_generated/parabolic_period.npy", np.asarray(grating_periods))
    np.save("./input_data_generated/parabolic_delta_n_eff.npy", np.asarray(delta_n_effs))
    np.save("./input_data_generated/parabolic_X_z.npy", np.asarray(X_zs))


def generate_2nd_and_3rd_degree_distributions():
    with open('model_config.json', 'r') as file:
        config = json.load(file)

    L = config["L"] * 1e3
    num_points = 15
    x_min = -L / 2
    x_max = L / 2
    x_values = np.linspace(x_min, x_max, num_points)

    y_min = 1.440
    y_max = 1.450
    random.seed(int(time.time()))
    range_a = (random.uniform(-1.5, 0), random.uniform(0.001, 0.5))
    range_c = (random.uniform(-0.3, 0), random.uniform(0.01, 1.2))
    third_n_effs = generate_3rd_degree_arrays(num_points, range_a, range_c, x_values, y_max, y_min)
    second_n_effs = generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min)
    n_effs = np.unique(np.concatenate((third_n_effs, second_n_effs), axis=0), axis=0)
    print("Done n_effs")

    y_min = 5.350  # 535e-9
    y_max = 5.400  # 540e-9
    random.seed(int(time.time()))
    range_a = (random.uniform(-2, 0), random.uniform(0, 1.2))
    range_c = (random.uniform(-1.5, 0.2), random.uniform(0.201, 1))
    third_grating_periods = generate_3rd_degree_arrays(num_points, range_a, range_c, x_values, y_max, y_min)
    second_grating_periods = generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min)
    grating_periods = np.unique(np.concatenate((third_grating_periods, second_grating_periods), axis=0), axis=0)
    print("Done grating_periods")

    y_min = 0.1  # 1e-5
    y_max = 1  # 1e-4
    random.seed(int(time.time()))
    range_a = (random.uniform(-2, 0), random.uniform(0, 1.5))
    range_c = (random.uniform(-2, 0), random.uniform(0, 1.4))
    third_delta_n_effs = generate_3rd_degree_arrays(num_points, range_a, range_c, x_values, y_max, y_min)
    second_delta_n_effs = generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min)
    delta_n_effs = np.unique(np.concatenate((third_delta_n_effs, second_delta_n_effs), axis=0), axis=0)
    print("Done delta_n_effs")

    y_min = 0.1  # 0.01
    y_max = 9.9  # 0.99
    random.seed(int(time.time()))
    range_a = (random.uniform(-1.4, 0), random.uniform(0, 0.6))
    range_c = (random.uniform(-0.6, 0), random.uniform(0, 1.3))
    third_X_zs = generate_3rd_degree_arrays(num_points, range_a, range_c, x_values, y_max, y_min)
    second_X_zs = generate_parabolas_arrays(num_points, range_a, x_values, y_max, y_min)
    X_zs = np.unique(np.concatenate((third_X_zs, second_X_zs), axis=0), axis=0)
    print("Done X_z_s")

    print("Saving to file")
    np.save("./input_data_generated/2nd_3rd_degree_n_eff.npy", np.asarray(n_effs))
    np.save("./input_data_generated/2nd_3rd_degree_period.npy", np.asarray(grating_periods))
    np.save("./input_data_generated/2nd_3rd_degree_delta_n_eff.npy", np.asarray(delta_n_effs))
    np.save("./input_data_generated/2nd_3rd_degree_X_z.npy", np.asarray(X_zs))


def generate_positive_only_sinusoid_distributions():
    with open('model_config.json', 'r') as file:
        config = json.load(file)
    L = config["L"]  # tu w metrach

    n_effs = []
    X_zs = []
    periods = []
    delta_n_effs = []

    frequencies = np.linspace(0.5, 8, 100)
    x = np.linspace(-L / 2, L / 2, 15)

    amplitudes = np.linspace(0.001, 0.005, 10)
    n_eff_y_shift = 1.44
    cartesian_product_sin = itertools.product(amplitudes, frequencies)
    for index, combination in enumerate(cartesian_product_sin):
        amplitude, frequency = combination
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * x / L) + amplitude + n_eff_y_shift
        if min(sine_wave) != max(sine_wave) and max(sine_wave) <= 1.45 and min(sine_wave) >= 1.44:
            n_effs.append(sine_wave)
        # plot_normalized(x, sine_wave)

    amplitudes = np.linspace(0.01, 0.05, 30)
    period_y_shift = 5.350
    cartesian_product_sin = itertools.product(amplitudes, frequencies)
    for index, combination in enumerate(cartesian_product_sin):
        amplitude, frequency = combination
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * x / L) + amplitude + period_y_shift
        if min(sine_wave) != max(sine_wave) and max(sine_wave) <= 5.400 and min(sine_wave) >= 5.350:
            periods.append(sine_wave)
        # plot_normalized(x, sine_wave)

    amplitudes = np.linspace(0.01, 0.9, 150)
    delta_n_eff_y_shift = 0.1
    cartesian_product_sin = itertools.product(amplitudes, frequencies)
    for index, combination in enumerate(cartesian_product_sin):
        amplitude, frequency = combination
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * x / L) + amplitude + delta_n_eff_y_shift
        if min(sine_wave) != max(sine_wave) and max(sine_wave) <= 1 and min(sine_wave) >= 0.1:
            delta_n_effs.append(sine_wave)
        # plot_normalized(x, sine_wave)

    amplitudes = np.linspace(0.5, 9.8, 150)
    X_z_shift = 0.1
    cartesian_product_sin = itertools.product(amplitudes, frequencies)
    for index, combination in enumerate(cartesian_product_sin):
        amplitude, frequency = combination
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * x / L) + amplitude + X_z_shift
        if min(sine_wave) != max(sine_wave) and max(sine_wave) <= 9.9 and min(sine_wave) >= 0.1:
            X_zs.append(sine_wave)
        # plot_normalized(x, sine_wave)

    print("Saving to file")
    np.save("./input_data_generated/sin_n_eff.npy", np.asarray(n_effs))
    np.save("./input_data_generated/sin_period.npy", np.asarray(periods))
    np.save("./input_data_generated/sin_delta_n_eff.npy", np.asarray(delta_n_effs))
    np.save("./input_data_generated/sin_X_z.npy", np.asarray(X_zs))


if __name__ == '__main__':
    generate_positive_only_sinusoid_distributions()
