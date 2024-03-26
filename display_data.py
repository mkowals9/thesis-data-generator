import matplotlib.pyplot as plt
import imageio
import datetime


def display_sections_data(xs, y_axis, ylabel, title,ct, want_save):
    plt.plot(xs, y_axis, drawstyle='steps-post')
    plt.xlabel("Section index")
    plt.ylabel(ylabel)
    plt.title(title + " " + ct)
    plt.grid(True)
    if want_save:
        plt.savefig(f'./plots/section_example_{ct}.png')
        plt.clf()
    else:
        plt.show()

def display_data(wavelengths, y_axis, ylabel, title,ct, want_save, log_scale):
    plt.plot(wavelengths, y_axis)
    plt.xlabel("Wavelength")
    plt.ylabel(ylabel)
    plt.title(title + " " + ct)
    if log_scale:
        plt.yscale('log')
        plt.ylim(1e-20, 1e0)
    else:
        plt.ylim(0, 1)
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
