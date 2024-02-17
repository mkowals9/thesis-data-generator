import matplotlib.pyplot as plt
import imageio
import datetime

def display_data(wavelengths, y_axis, delta_n_eff, n_eff, X_z, period, ylabel, title):
    plt.plot(wavelengths, y_axis)
    plt.xlabel("Wavelength")
    plt.ylabel(ylabel)
    plt.title(title)
    stats = (f'delta_n_eff = {delta_n_eff:.3e}\n'
             f'n_eff = {n_eff:.4f}\n'
             f'period = {period:.3}\n'
             f'X_z = {X_z:.5f}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(0.75, 0.9, stats, fontsize=10, bbox=bbox, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')
    #plt.ylim(1e-16, 1e0)
    #plt.yscale('log')
    plt.ylim(0, 0.6)
    plt.grid(True)
    #plt.show()
    plt.savefig(f'./plots/plot_delta_n_eff {delta_n_eff:.2e}-n_eff {n_eff:.2f}-period {period:.2e}.png')
    plt.clf()


def save_plots_to_one_gif(filenames):
    images = []
    for filename in filenames:
        images.append(imageio.imread('./plots/'+filename))
    ct = datetime.datetime.now()
    imageio.mimsave(f'./gifs/plots_{ct.timestamp()}.gif', images)
