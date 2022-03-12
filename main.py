import os
import cv2
import csv
import numpy as np

from tqdm import tqdm
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import Flatten

DATA_DIR = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\teste"

csv_header = ['class', 'video', 'frame', 'deep_features']

model = keras.models.load_model("modelo.h5")
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers if isinstance(layer, Flatten)],
)
print(feature_extractor.summary())

class_names = os.listdir(DATA_DIR)

f = open('x_deep.csv', 'w', encoding='UTF8', newline='')
writer = csv.writer(f, delimiter=';')
writer.writerow(csv_header)

for class_name in tqdm(class_names):
    file_names = os.listdir(os.path.join(DATA_DIR, class_name))
    for file in file_names:
        vidcap = cv2.VideoCapture(os.path.join(DATA_DIR, class_name, file))
        success, img = vidcap.read()
        frame = 1
        while success:
            img.resize((64, 64, 3))
            xd = image.img_to_array(img)
            xd = np.expand_dims(xd, axis=0)
            deep_features = feature_extractor(xd)

            x_image_aux = []
            for aux in deep_features:
                x_image_aux = np.append(x_image_aux, np.ravel(aux))

            deep_features = [i for i in x_image_aux]
            writer.writerow([class_name, file, frame, deep_features])

            success, img = vidcap.read()
            frame = frame + 1

f.close()