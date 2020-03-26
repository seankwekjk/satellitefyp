import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from rasterio.plot import show


# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min)/(array_max - array_min)


# Open the file:
folder = "D:\SatelliteData\MODIS_HDF\sumatra_2013_2019\Y2015"
# folder = "D:\SatelliteData\MODIS_HDF\kalimantan_2013_2019\Y2015"
vals = pd.read_csv("2015.csv", index_col="Day")
# print(vals)

img_list = os.listdir(folder)
for img in img_list:
    date = img[31:34]
    val = vals.loc[int(date), 'Sumatra']
    # val = vals.loc[int(date), 'Kalimantan']
    # print(val)
    # print(date)
    # print(date + '_' + str(val) + '.png')

    '''
    test = date + '_' + str(val) + '.png'
    first = test.split('_')[1]
    final = first.split('.')[0]
    print(final)
    '''

    ir_tif = rasterio.open(folder + '/' + img)
    ir = ir_tif.read(1)
    # show(ir)
    # print(ir.shape)

    '''
    # Normalize band DN
    redn = normalize(red)
    bluen = normalize(blue)
    greenn = normalize(green)
    irn = normalize(ir)
    '''

    # Stack bands
    plt.imsave('modis_img/' + date + '_' + str(val) + '.png', ir, cmap=plt.get_cmap('gray'))

    '''
    # Stack bands
    rgb = np.dstack((redn, greenn, bluen))

    # View the color composite
    plt.imsave('img/' + folder + '.png', rgb)
    plt.imsave('ir_img/' + folder + '.png', ir)
    '''
