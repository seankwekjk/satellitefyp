import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.plot import show


# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min)/(array_max - array_min)


# Open the file:
outer_folder = "D:/SatelliteData/ndvi"
folder_list = os.listdir(outer_folder)
for folder in folder_list:
    parent_folder = outer_folder + '/' + folder
    image_list = os.listdir(parent_folder)
    # print(image_list)
    red_tif = rasterio.open(parent_folder + '/' + image_list[0])
    nir_tif = rasterio.open(parent_folder + '/' + image_list[1])

    # Convert to numpy arrays
    red = red_tif.read(1).astype('float64')
    # show(red)
    nir = nir_tif.read(1).astype('float64')
    # show(nir)

    # Calculate NDVI
    ndvi = np.where(
        (nir + red) == 0,
        0,
        (nir - red) / (nir + red))

    # Save img
    plt.imsave(outer_folder + '/' + folder + '_ndvi.png', ndvi)
