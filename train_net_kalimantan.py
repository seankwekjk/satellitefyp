import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnn_code.cnn_kalimantan import train
from keras.utils import to_categorical
from natsort import natsorted
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# settings
count = 0
default_shape = [756, 1211]
concat_shape = [1, 756, 1211]

# make labels (build into filename)
labels = np.zeros(276)

# Open the file:
# outer_folder = "D:/SatelliteData/MODIS_HDF/sumatra_training_data"
outer_folder = "D:/SatelliteData/MODIS_HDF/kalimantan_training_data"
folder_list = os.listdir(outer_folder)

for inner_folder in folder_list:
    year = inner_folder.split('Y')[1]
    vals = pd.read_csv(year + ".csv", index_col="Day")
    # print(vals)

    folder = outer_folder + '/' + inner_folder
    img_list = os.listdir(folder)
    for img in img_list:
        date = img[31:34]
        # val = vals.loc[int(date), 'Sumatra']
        val = vals.loc[int(date), 'Kalimantan']
        labels[count] = val
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
        # plt.imsave('modis_img/' + year + '_' + date + '_' + str(val) + '.png', ir, cmap=plt.get_cmap('gray'))

        if count == 0:
            images = ir
        elif count == 1:
            data = ir
            images = np.stack((images, data), axis=0)
        else:
            data = np.zeros(concat_shape)
            data[0, :ir.shape[0], :ir.shape[1]] = ir
            images = np.concatenate((images, data), axis=0)
        print(count)
        count += 1

print('shape: ' + str(images.shape))
# print('labels: ' + labels)

# data split
train_features, test_features, train_target, test_target = train_test_split(images, labels, test_size=0.3)
train_target = to_categorical(train_target)
test_target = to_categorical(test_target)
train(train_features, train_target, test_features, test_target)
