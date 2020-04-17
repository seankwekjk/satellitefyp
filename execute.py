import os
import rasterio
import numpy as np

from keras.models import model_from_json

# load json and create sumatra model
json_file = open('sumatra_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
sumatra_model = model_from_json(loaded_model_json)
# load weights into new model
sumatra_model.load_weights("sumatra_model.h5")
print("Loaded sumatra model")

# load json and create kalimantan model
json_file = open('kalimantan_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
kalimantan_model = model_from_json(loaded_model_json)
# load weights into new model
kalimantan_model.load_weights("kalimantan_model.h5")
print("Loaded kalimantan model")

# load sumatra images
# settings
count = 0
default_shape = [768, 1023]
concat_shape = [1, 768, 1023]

# Open the file:
folder = "D:/SatelliteData/MODIS_HDF/sumatra_demo_data"
# folder = "D:/SatelliteData/MODIS_HDF/kalimantan_demo_data"
img_list = os.listdir(folder)

for img in img_list:
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
    # print(count)
    count += 1

sumatra_data = images[:, :, :, np.newaxis]

# load kalimantan images
folder = "D:/SatelliteData/MODIS_HDF/kalimantan_demo_data"
img_list = os.listdir(folder)

for img in img_list:
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
    # print(count)
    count += 1

kalimantan_data = images[:, :, :, np.newaxis]

# predictions
sumatra_predictions = sumatra_model.predict(sumatra_data)
kalimantan_predictions = kalimantan_model.predict(kalimantan_data)

# calculate psi
sumatra_hotspots = 0
kalimantan_hotspots = 0
for prediction in sumatra_predictions:
    sumatra_hotspots += prediction

for prediction in kalimantan_predictions:
    kalimantan_hotspots += prediction

same_week_sumatra_psi = 37.35782 + (sumatra_hotspots * 0.032264)
same_week_kalimantan_psi = 47.9311 + (kalimantan_hotspots * 0.012023)
next_week_sumatra_psi = 43.78655 + (sumatra_hotspots * 0.020929)
next_week_kalimantan_psi = 43.57163 + (kalimantan_hotspots * 0.013968)
same_week_final_psi = (same_week_sumatra_psi + same_week_kalimantan_psi)/2
next_week_final_psi = (next_week_sumatra_psi + next_week_kalimantan_psi)/2
print("Same Week PSI: " + str(same_week_final_psi))
print("Predicted PSI: " + str(next_week_final_psi))
