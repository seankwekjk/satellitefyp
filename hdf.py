import gdal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
MODIS IMAGE DATA
IDENTIFY LOCATION BY TILE NUMBER
VERTICAL: 09
HORIZONTAL: 28 = SUMATRA, 29 = KALIMANTAN
'''

'''
ds = gdal.Open('HDF4_SDS:UNKNOWN:"test.hdf":0')
data = ds.ReadAsArray()
ds = None
print(data.shape)
'''

'''
# get the path for subdatasets
ds = gdal.Open('test2.hdf')
for sd, descr in ds.GetSubDatasets():
    print(sd)
    print(descr)
    print()


Emis_31 MODIS_Grid_Daily_1km_LST (8-bit unsigned integer)
Emis_32 MODIS_Grid_Daily_1km_LST (8-bit unsigned integer)
LST_Night_1km MODIS_Grid_Daily_1km_LST (16-bit unsigned integer)
LST_Day_1km MODIS_Grid_Daily_1km_LST (16-bit unsigned integer)
QC_Day MODIS_Grid_Daily_1km_LST (8-bit unsigned integer)
QC_Night MODIS_Grid_Daily_1km_LST (8-bit unsigned integer)
'''

'''
ds = gdal.Open('test.hdf')
subds = [sd for sd, descr in ds.GetSubDatasets() if descr.endswith('sur_refl_b01_1 MODIS_Grid_2D (16-bit integer)')][0]
dssub = gdal.Open(subds)
data = dssub.ReadAsArray()
dssub = None
ds = None
'''

# plt.plot(data)
# plt.show()

# fig, ax = plt.subplots(figsize=(6, 6))

# ax.imshow(data[:, :], cmap=plt.cm.Greys, vmin=1000, vmax=6000)


# plt.imsave('test.png', myarray)
