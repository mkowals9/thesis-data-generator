import matplotlib.pyplot as plt
import imageio
import datetime


def display_data(wavelengths, y_axis, ylabel, title,ct, want_save, log_scale):
    plt.plot(wavelengths, y_axis)
    plt.xlabel("Wavelength")
    plt.ylabel(ylabel)
    plt.title(title + " " + ct)
    # if log_scale:
    #     plt.yscale('log')
    #     plt.ylim(1e-16, 1e0)
    # else:
    #     plt.ylim(0, 0.6)
    plt.grid(True)
    if want_save:
        plt.savefig(f'./plots/plot_example_{ct}.png')
        plt.clf()
    else:
        plt.show()


def save_plots_to_one_gif(filenames):
    images = []
    for filename in filenames:
        images.append(imageio.imread('./plots/' + filename))
    ct = datetime.datetime.now()
    imageio.mimsave(f'./gifs/plots_{ct.timestamp()}.gif', images)
