import os
import numpy as np
from natsort import natsorted
from PIL import Image


file_paths = natsorted(os.listdir('modis_img/sumatra_2015_bw'))
# print(file_paths)

# load images
default_shape = [1023, 768, 1]
concat_shape = [1, 1023, 768, 1]
for path in file_paths:
    img = Image.open('modis_img/sumatra_2015_bw/' + path)
    # ir_img = Image.open('ir_img/' + path)
    img.load()
    # ir_img.load()
    data = np.asarray(img, dtype="int32")
    print(data.shape)