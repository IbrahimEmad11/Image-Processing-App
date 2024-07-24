import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap

def draw_rgb_histogram(image, app):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    red_hist, red_bins = np.histogram(red_channel.flatten(), bins=256, range=[0, 256])
    green_hist, green_bins = np.histogram(green_channel.flatten(), bins=256, range=[0, 256])
    blue_hist, blue_bins = np.histogram(blue_channel.flatten(), bins=256, range=[0, 256])

    save_histogram_plot(red_hist, red_bins, 'Red', 'red_histogram.png')
    save_histogram_plot(green_hist, green_bins, 'Green', 'green_histogram.png')
    save_histogram_plot(blue_hist, blue_bins, 'Blue', 'blue_histogram.png')

    set_histogram_pixmaps(app)

def draw_rgb_disturb_curve(image, app):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    red_hist, red_bins = np.histogram(red_channel.flatten(), bins=256, range=[0, 256])
    green_hist, green_bins = np.histogram(green_channel.flatten(), bins=256, range=[0, 256])
    blue_hist, blue_bins = np.histogram(blue_channel.flatten(), bins=256, range=[0, 256])

    red_cum_hist = np.cumsum(red_hist)
    green_cum_hist = np.cumsum(green_hist)
    blue_cum_hist = np.cumsum(blue_hist)

    save_histogram_plot(red_cum_hist, red_bins, 'Red', 'red_cdf.png')
    save_histogram_plot(green_cum_hist, green_bins, 'Green', 'green_cdf.png')
    save_histogram_plot(blue_cum_hist, blue_bins, 'Blue', 'blue_cdf.png')

    set_cdf_pixmaps(app)

def save_histogram_plot(hist, bins, color, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bins[:-1], hist, color=color.lower(), label=color)
    ax.set_title(f"{color} Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"assets/graphs/{filename}")

def set_histogram_pixmaps(app):
    app.ui.Histogram_1.setPixmap(QPixmap("assets/graphs/red_histogram.png"))
    app.ui.Histogram_3.setPixmap(QPixmap("assets/graphs/green_histogram.png"))
    app.ui.Histogram_5.setPixmap(QPixmap("assets/graphs/blue_histogram.png"))

def set_cdf_pixmaps(app):
    app.ui.Histogram_2.setPixmap(QPixmap("assets/graphs/red_cdf.png"))
    app.ui.Histogram_4.setPixmap(QPixmap("assets/graphs/green_cdf.png"))
    app.ui.Histogram_6.setPixmap(QPixmap("assets/graphs/blue_cdf.png"))
