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
parent_folder = "D:/SatelliteData/Aug-Oct_2019_Indonesia/Landsat8_30092019/LC08_L1TP_127060_20190930_20191001_01_RT"
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

# Stack bands
rgb = np.dstack((redn, greenn, bluen))

# View the color composite
plt.imshow(rgb)
plt.show()
