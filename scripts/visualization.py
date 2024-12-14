import sys 
import importlib
import subprocess

required_packages = ["json", "numpy", "matplotlib", "os", "sys"]

for package in required_packages:
    try:
        # Try to import the package
        importlib.import_module(package)
        print(f"'{package}' is already installed.")
    except ImportError:
        print(f"'{package}' is NOT installed.")
        if package in ["numpy", "matplotlib"]:  # These are not in the standard library
            print(f"Attempting to install '{package}'...")
            try:
                # Use subprocess to run pip for installation
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"'{package}' has been successfully installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install '{package}'. Error: {e}")
        else:
            print(f"'{package}' is part of Python's standard library and should not require installation.")

import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp"))


if len(sys.argv) > 1:
    filename = sys.argv[1]
    print(f"Received filename: {filename}")
else:
    print("No filename provided.")


with open("MNIST.json", "r") as file:
    data = json.load(file)

with open(filename, "r") as file:   #MNIST_Output.json
    result = json.load(file)

def plot_digit(ax, image_data):
    ax.imshow(image_data.T, cmap="binary")
    ax.axis("off")


fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])

left_grid = gs[0, 0].subgridspec(5, 5)

# Plot the first 5 x 5 samples
for idx, instance in enumerate(data["Instance"][:25]):

    sorted_keys = sorted(instance["Features"].keys(), key=int)
    sorted_features = [instance["Features"][k] for k in sorted_keys]
    
    # get the features 
    features = np.array(list(sorted_features)).reshape(28, 28)

    row, col = divmod(idx, 5)
    ax = fig.add_subplot(left_grid[row, col])
    plot_digit(ax, features)

plt.subplots_adjust(wspace=0, hspace=0)

# Plot the importance map on the right 
heatmap = np.zeros(784)
for pixel, importance in result.items():
    heatmap[int(pixel)] = importance
heatmap = heatmap.reshape((28, 28))

ax2 = fig.add_subplot(gs[0, 1])
heatmap_img = ax2.imshow(heatmap.T, cmap="hot")
cbar = fig.colorbar(ax2.imshow(heatmap.T, cmap="hot"), ax=ax2, ticks=[heatmap.min(), heatmap.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)
ax2.axis("off")

base_name, _ = os.path.splitext(filename)
plt.savefig(f"{base_name}_plot.png", dpi=300, bbox_inches='tight')