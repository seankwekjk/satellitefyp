import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnn_code.cnn_sumatra import train
from keras.utils import to_categorical
from natsort import natsorted
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split


# settings
count = 0
default_shape = [768, 1023]
concat_shape = [1, 768, 1023]

# make labels (build into filename)
labels = np.zeros(276)

# Open the file:
outer_folder = "D:/SatelliteData/MODIS_HDF/sumatra_training_data"
# outer_folder = "D:/SatelliteData/MODIS_HDF/kalimantan_training_data"
folder_list = os.listdir(outer_folder)

for inner_folder in folder_list:
    year = inner_folder.split('Y')[1]
    vals = pd.read_csv(year + ".csv", index_col="Day")
    # print(vals)

    folder = outer_folder + '/' + inner_folder
    img_list = os.listdir(folder)
    for img in img_list:
        date = img[31:34]
        val = vals.loc[int(date), 'Sumatra']
        # val = vals.loc[int(date), 'Kalimantan']
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
'''

# load images
count = 0
default_shape = [768, 1023]
concat_shape = [1, 768, 1023]
for path in file_paths:
    img = Image.open('comp_img/' + path)
    # ir_img = Image.open('ir_img/' + path)
    img.load()
    # ir_img.load()
    if count == 0:
        images = np.zeros(default_shape)
        data = np.asarray(img, dtype="int32")
        # ir_data = np.asarray(ir_img, dtype="int32")
        # data = np.concatenate((data, ir_data), axis=2)
        resized_data = resize(data, (data.shape[0] // 4, data.shape[1] // 4), anti_aliasing=True)
        images[:resized_data.shape[0], :resized_data.shape[1], :resized_data.shape[2]] = resized_data
    elif count == 1:
        data = np.zeros(default_shape)
        temp = np.asarray(img, dtype="int32")
        # ir_data = np.asarray(ir_img, dtype="int32")
        # temp = np.concatenate((temp, ir_data), axis=2)
        resized_data = resize(temp, (temp.shape[0] // 4, temp.shape[1] // 4), anti_aliasing=True)
        data[:resized_data.shape[0], :resized_data.shape[1], :resized_data.shape[2]] = resized_data
        images = np.stack((images, data), axis=0)
    else:
        data = np.zeros(concat_shape)
        temp = np.asarray(img, dtype="int32")
        # ir_data = np.asarray(ir_img, dtype="int32")
        # temp = np.concatenate((temp, ir_data), axis=2)
        resized_data = resize(temp, (temp.shape[0] // 4, temp.shape[1] // 4), anti_aliasing=True)
        data[0, :resized_data.shape[0], :resized_data.shape[1], :resized_data.shape[2]] = resized_data
        images = np.concatenate((images, data), axis=0)
    print(path + ' shape: ' + str(images.shape))
    count += 1

# Get image size
print(images.shape)
# image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
# print(image_size)

# make labels (build into filename)
labels = np.zeros(images.shape[0])
fires = [1, 2, 3, 4, 6, 8, 9, 18, 21, 23, 26, 27, 28, 36]
for i in fires:
    labels[i] = 1

# data split
train_features, test_features, train_target, test_target = train_test_split(images, labels, test_size=0.3)
train_target = to_categorical(train_target)
test_target = to_categorical(test_target)
'''
'''
svc = svm.SVC(kernel='linear', C=1.0)

# train on svm
svc.fit(train_features.reshape(len(train_features), -1), train_target)

# test results
prediction = svc.predict(test_features.reshape(len(test_features), -1))
print(classification_report(test_target, prediction))
'''
# train(train_features, train_target, test_features, test_target)
