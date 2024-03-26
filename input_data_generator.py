import math
import itertools
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
import time


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


def generate_many_distributions():
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


def generate_parabolas(a, b, c, x_values, desired_min, desired_max):
    values = a * x_values ** 2 + b * x_values + c
    min_value = np.min(values)
    max_value = np.max(values)
    normalized_values = (desired_min + (values - min_value) * (desired_max - desired_min) / (max_value - min_value))
    # plot_normalized(x_values, normalized_values)
    return normalized_values


def generate_coefficients(num_coefficients, p, q, range_a, seed=None):
    rng = np.random.default_rng(seed)
    a_values = np.round(rng.uniform(range_a[0], range_a[1], num_coefficients), 4)
    noise = generate_noise(num_coefficients)
    b_values = np.round(np.array([-p * 2 * a + noise[a_index] if a_index % 3 == 0 else -p * 2 * a
                                  for a_index, a in enumerate(a_values)]), 4)
    c_values = np.round(np.unique(np.array([(4 * a * q + b_values[a_index] ** 2) / (4 * a) if a != 0 else 0
                                            for a_index, a in enumerate(a_values)])), 4)
    return list(zip(a_values, b_values, c_values))


def generate_noise(num_points):
    return np.random.uniform(1e-3, 1e-4, size=num_points)


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
    vertex_values = np.linspace(y_min, y_max, num_points)
    cartesian_product = itertools.product(x_values, vertex_values)
    temp_n_effs = []
    for index, combination in enumerate(cartesian_product):
        p, q = combination
        coefficients_n_effs = (generate_coefficients(1000, p, q, (-2, 2), int(time.time())))
        for a, b, c in coefficients_n_effs:
            generated = np.round(generate_parabolas(a, b, c, x_values, y_min, y_max), 4)
            if index % 3 == 0:
                generated = np.round(generated + generate_noise(num_points), 4)
            if index % 5 == 0:
                generated = np.round(generated - generate_noise(num_points), 4)
            generated = [x if x > 0 else (-1) * x for x in generated]
            temp_n_effs.append(generated)
    n_effs = np.unique(temp_n_effs, axis=0)
    print("Done n_effs")

    y_min = 535e-9
    y_max = 540e-9
    vertex_values = np.linspace(y_min, y_max, num_points)
    cartesian_product = itertools.product(x_values, vertex_values)
    temp_periods = []
    for index, combination in enumerate(cartesian_product):
        p, q = combination
        coefficients_periods = (generate_coefficients(1000, p, q, (-1.75, 1.7), int(time.time())))
        for a, b, c in coefficients_periods:
            generated = generate_parabolas(a, b, c, x_values, y_min, y_max)
            if index % 3 == 0:
                generated = generated + generate_noise(num_points)
            if index % 5 == 0:
                generated = generated - generate_noise(num_points)
            generated = [x if x > 0 else (-1) * x for x in generated]
            temp_periods.append(generated)
    grating_periods = np.unique(temp_periods, axis=0)
    print("Done grating_periods")

    y_min = 1e-5
    y_max = 1e-4
    vertex_values = np.linspace(y_min, y_max, num_points)
    cartesian_product = itertools.product(x_values, vertex_values)
    temp_delta_n_effs = []
    for index, combination in enumerate(cartesian_product):
        p, q = combination
        coefficients_periods = (generate_coefficients(1000, p, q, (-1.65, 1.75), int(time.time())))
        for a, b, c in coefficients_periods:
            generated = generate_parabolas(a, b, c, x_values, y_min, y_max)
            if index % 3 == 0:
                generated = np.round(generated + generate_noise(num_points), 4)
            if index % 5 == 0:
                generated = np.round(generated - generate_noise(num_points), 4)
            generated = [x if x > 0 else (-1) * x for x in generated]
            temp_delta_n_effs.append(generated)

    delta_n_effs = np.unique(temp_delta_n_effs, axis=0)
    print("Done delta_n_effs")

    y_min = 0.01
    y_max = 0.99
    vertex_values = np.linspace(y_min, y_max, num_points)
    cartesian_product = itertools.product(x_values, vertex_values)
    temp_X_zs = []
    for index, combination in enumerate(cartesian_product):
        p, q = combination
        coefficients_periods = (generate_coefficients(1000, p, q, (-1.75, 1.6), int(time.time())))
        for a, b, c in coefficients_periods:
            generated = np.round(generate_parabolas(a, b, c, x_values, y_min, y_max), 4)
            if index % 3 == 0:
                generated = np.round(generated + generate_noise(num_points), 4)
            if index % 5 == 0:
                generated = np.round(generated - generate_noise(num_points), 4)
            generated = [x if x > 0 else (-1)*x for x in generated]
            temp_X_zs.append(generated)
    X_zs = np.unique(temp_X_zs, axis=0)
    print("Done X_z_s")

    print("Saving to file")
    np.save("./input_data_generated/parabolic_n_eff.npy", np.asarray(n_effs))
    np.save("./input_data_generated/parabolic_period.npy", np.asarray(grating_periods))
    np.save("./input_data_generated/parabolic_delta_n_eff.npy", np.asarray(delta_n_effs))
    np.save("./input_data_generated/parabolic_X_z.npy", np.asarray(X_zs))


if __name__ == '__main__':
    generate_parabolic_distributions()
