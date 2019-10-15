import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.plot import show


# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min)/(array_max - array_min)


# Open the file:
parent_folder = "D:/SatelliteData/Aug-Oct_2019_Indonesia/Landsat8_30092019/LC08_L1TP_127060_20190930_20191001_01_RT/"
red_tif = rasterio.open(parent_folder + 'band7.TIF')
green_tif = rasterio.open(parent_folder + 'band5.TIF')
blue_tif = rasterio.open(parent_folder + 'band2.TIF')

# Convert to numpy arrays
red = red_tif.read(1)
# show(red)
green = green_tif.read(1)
# show(green)
blue = blue_tif.read(1)
# show(blue)

# Normalize band DN
redn = normalize(red)
bluen = normalize(blue)
greenn = normalize(green)

# Stack bands
rgb = np.dstack((redn, greenn, bluen))

# View the color composite
plt.imshow(rgb)
plt.show()
