"""

# Copyright (c) Mindseye Biomedical LLC. All rights reserved.
# Distributed under the (new) CC BY-NC-SA 4.0 License. See LICENSE.txt for more info.

    Read in a data file and plot it using an algorithm.

"""
from __future__ import division, absolute_import, print_function
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import OpenEIT.reconstruction
import os

n_electrodes = 16
data_directory = "data/"
results_directory = "results/"
data_filename = data_directory + "200928lungs_working4.txt"
baseline_filename = data_directory + "200928NoWire.txt"
results_filename = "results/200928lungs_working45.gif"


def main():
    # Create a pyEIT Jacobian Reconstruction Object
    pyeit_obj = OpenEIT.reconstruction.JacReconstruction(n_electrodes)

    # Get mesh plotting variables
    electrode_indices = pyeit_obj.el_pos
    pts = pyeit_obj.mesh_obj['node']
    tri = pyeit_obj.mesh_obj['element']
    x = pts[:, 0]
    y = pts[:, 1]

    # Load Data
    data = load_data(data_filename)
    baseline_data = load_data(baseline_filename)

    # First reconstruction call sets baseline
    pyeit_obj.eit_reconstruction(baseline_data[0])

    plot_images = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for frame in data:
        # EIT Reconstruction
        eit_image = pyeit_obj.eit_reconstruction(frame)

        # Create plot of reconstruction
        plot_image = ax.tripcolor(x, y, tri, eit_image, vmin=-0, vmax=50000)
        plot_images.append([plot_image])

    # Plot images
    ax.plot(x[electrode_indices], y[electrode_indices], 'ro')
    for i, e in enumerate(electrode_indices):
        ax.text(x[e], y[e], str(i + 1), size=12)
    ax.axis('equal')
    fig.colorbar(plot_images[-1][0])

    # Save animation as gif
    writer_gif = animation.PillowWriter(fps=30)
    ani = animation.ArtistAnimation(fig, plot_images, interval=200, blit=True, repeat_delay=500)
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)
    ani.save(results_filename, writer_gif)

    plt.show()


def load_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        data.append(parse_line(line))

    return data


def parse_line(line):
    try:
        _, data = line.split(":", 1)
    except ValueError:
        return None
    items = []
    for item in data.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(float(item))
        except ValueError:
            return None
    return np.array(items)


if __name__ == "__main__":
    main()
