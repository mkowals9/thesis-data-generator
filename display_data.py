import matplotlib.pyplot as plt
import imageio
import datetime
import numpy as np
import json


def display_data(wavelengths, y_axis, delta_n_eff, n_eff, period, ylabel, title, want_save, log_scale, X_z=0):
    plt.plot(wavelengths, y_axis)
    plt.xlabel("Długość fali [m]")
    plt.ylabel(ylabel)
    plt.title(title)
    stats = (f'delta_n_eff = {delta_n_eff:.3e}\n'
             f'n_eff = {n_eff:.4f}\n'
             f'period = {period:.3}\n'
             f'X_z = {X_z:.3f}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(0.75, 0.9, stats, fontsize=10, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    if log_scale:
        plt.yscale('log')
        plt.ylim(1e-16, 1e0)
    else:
        plt.ylim(0, 0.5)
    plt.grid(True)
    if want_save:
        plt.savefig(f'./plots/plot_delta_n_eff {delta_n_eff:.2e}-n_eff {n_eff:.2f}-period {period:.2e}.png')
        plt.clf()
    else:
        plt.show()


def save_plots_to_one_gif(filenames):
    images = []
    for filename in filenames:
        images.append(imageio.imread('./plots/' + filename))
    ct = datetime.datetime.now()
    imageio.mimsave(f'./gifs/plots_{ct.timestamp()}.gif', images)


def display_data_from_both_models_on_one_plot(main_y, piecewise_y, delta_n_eff, n_eff, period, X_z, want_save):
    with open('model_config.json', 'r') as file:
        config = json.load(file)

    num_points = config["num_points"]
    start_value = config["start_value"]  # początkowy zakres fal
    end_value = config["end_value"]  # końcowy zakres fal
    wavelengths = np.linspace(start_value, end_value, num_points)
    plt.plot(wavelengths, main_y, label='Reflectance - analytical model')
    plt.plot(wavelengths, piecewise_y, label='Reflectance - piecewise model')
    plt.xlabel("Długość fali [m]")
    plt.ylim(0, 0.5)
    plt.ylabel("Reflektancja")
    plt.legend()
    plt.title("Reflektancja - model analityczny i macierzowy")
    stats = (f'delta_n_eff = {delta_n_eff:.3e}\n'
             f'n_eff = {n_eff:.4f}\n'
             f'period = {period:.3}\n'
             f'X_z = {X_z:.3f}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(0.75, 0.82, stats, fontsize=10, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    plt.grid(True)
    if want_save:
        plt.savefig(f'./plots/plot_delta_n_eff {delta_n_eff:.2e}-n_eff {n_eff:.2f}-period {period:.2e}_comparison.png')
        plt.clf()
    else:
        plt.show()


def main():
    with open('data_main_model_input.json', 'r') as file:
        analytical_model_data = json.load(file)
    with open('data_piecewise_model_input.json', 'r') as file:
        piecewise_model_data = json.load(file)
    for index, obj in enumerate(piecewise_model_data):
        display_data_from_both_models_on_one_plot(analytical_model_data[index]["reflectance"],
                                                  obj["reflectance"],
                                                  obj["delta_n_eff"],
                                                  obj["n_eff"],
                                                  obj["period"],
                                                  obj["X_z"],
                                                  False)


if __name__ == '__main__':
         main()
