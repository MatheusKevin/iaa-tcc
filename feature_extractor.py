import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import Flatten

DATA_DIR = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\midlevel"
DEST_DIR = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\deep_features"

model = keras.models.load_model("modelo.h5")
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers if isinstance(layer, Flatten)]
)
print(feature_extractor.summary())

class_names = os.listdir(DATA_DIR)

for class_name in tqdm(class_names):
    if not os.path.exists(os.path.join(DEST_DIR, class_name)):
        os.makedirs(os.path.join(DEST_DIR, class_name))

    file_names = os.listdir(os.path.join(DATA_DIR, class_name))
    for file in file_names:
        x_deep = []
        vidcap = cv2.VideoCapture(os.path.join(DATA_DIR, class_name, file))
        success, img = vidcap.read()
        while success:
            img.resize((64, 64, 3))
            xd = image.img_to_array(img)
            xd = np.expand_dims(xd, axis=0)
            deep_features = feature_extractor(xd)

            x_image_aux = []
            for aux in deep_features:
                x_image_aux = np.append(x_image_aux, np.ravel(aux))

            deep_features = [i for i in x_image_aux]
            x_deep.append(deep_features)

            success, img = vidcap.read()

        df = pd.DataFrame(x_deep)
        df.to_csv(os.path.join(DEST_DIR, class_name, file + ".csv"), header=False, index=False)
        vidcap.release()
