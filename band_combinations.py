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
outer_folder = "D:/SatelliteData/Aug-Oct_2019_Indonesia/landsat"
folder_list = os.listdir(outer_folder)
for folder in folder_list:
    parent_folder = outer_folder + '/' + folder
    image_list = os.listdir(parent_folder)
    # print(image_list)
    red_tif = rasterio.open(parent_folder + '/' + image_list[9])
    green_tif = rasterio.open(parent_folder + '/' + image_list[7])
    blue_tif = rasterio.open(parent_folder + '/' + image_list[4])
    ir_tif = rasterio.open(parent_folder + '/' + image_list[12])

    # Convert to numpy arrays
    red = red_tif.read(1)
    # show(red)
    green = green_tif.read(1)
    # show(green)
    blue = blue_tif.read(1)
    # show(blue)
    ir = ir_tif.read(1)
    # show(ir)

    # Normalize band DN
    redn = normalize(red)
    bluen = normalize(blue)
    greenn = normalize(green)
    irn = normalize(ir)

    # Stack bands
    rgb = np.dstack((redn, greenn, bluen, irn))
    plt.imsave('comp_img/' + folder + '.png', rgb)

    '''
    # Stack bands
    rgb = np.dstack((redn, greenn, bluen))

    # View the color composite
    plt.imsave('img/' + folder + '.png', rgb)
    plt.imsave('ir_img/' + folder + '.png', ir)
    '''
