import matplotlib.pyplot as plt


def display_data(wavelengths, y_axis, delta_n_eff, n_eff, period, ylabel, title):
    plt.plot(wavelengths, y_axis)
    plt.xlabel("Wavelength")
    plt.ylabel(ylabel)
    plt.title(title)
    stats = (f'delta_n_eff = {delta_n_eff:.2e}\n'
             f'n_eff = {n_eff:.2f}\n'
             f'period = {period:.2e}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(0.95, 1.1, stats, fontsize=10, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    plt.grid(True)
    plt.show()
