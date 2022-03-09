import os
import cv2
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing import image


DATA_DIR = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\midlevel"

model = keras.models.load_model("modelo.h5")


class_names = os.listdir(DATA_DIR)

for class_name in class_names:
    file_names = os.listdir(os.path.join(DATA_DIR, class_name))
    for file in file_names:
        vidcap = cv2.VideoCapture(os.path.join(DATA_DIR, class_name, file))
        success, img = vidcap.read()
        while success:
            success, img = vidcap.read()
            xd = image.img_to_array(img)
            xd = np.expand_dims(xd, axis=-1)
            deep_features = model.predict(xd)

            print(deep_features.shape)