import os
import csv
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import Flatten

DEST_DIR = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\deep_features"
DATA_DIR = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\midlevel"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'validation')
exceptions = ['closing_backpack', 'looking_back_left_shoulder', 'looking_back_right_shoulder', 'opening_backpack', 'standing_by_the_door', 'opening_backpack']

model = keras.models.load_model("modelo.h5")
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers if isinstance(layer, Flatten)]
)
print(feature_extractor.summary())


def extract_features_in_x_y_format(data_path):
    part = data_path.split('\\')[-1]
    if not os.path.exists(os.path.join(DEST_DIR, part)):
        os.makedirs(os.path.join(DEST_DIR, part))

    x_file = open(os.path.join(DEST_DIR, part, 'x_deep.csv'), 'w', newline='')
    x_writer = csv.writer(x_file)
    y_file = open(os.path.join(DEST_DIR, part, 'y_labels.csv'), 'w', newline='')
    y_writer = csv.writer(y_file)

    class_names = [cl_name for cl_name in os.listdir(data_path) if cl_name not in exceptions]

    for class_num, class_name in enumerate(class_names):
        file_names = os.listdir(os.path.join(data_path, class_name))
        for file in tqdm(file_names):
            vidcap = cv2.VideoCapture(os.path.join(data_path, class_name, file))
            success, img = vidcap.read()

            while success:
                img.resize((64, 64, 3))
                xd = image.img_to_array(img)
                xd = np.expand_dims(xd, axis=0)
                deep_features = feature_extractor(xd)

                x_writer.writerow(np.ravel(deep_features))
                y_writer.writerow([class_num])

                success, img = vidcap.read()

            vidcap.release()

    x_file.close()
    y_file.close()


def extract_features_in_video_format(data_path):
    class_names = [cl_name for cl_name in os.listdir(data_path) if cl_name not in exceptions]

    for class_name in class_names:
        if not os.path.exists(os.path.join(data_path, class_name)):
            os.makedirs(os.path.join(data_path, class_name))

        file_names = os.listdir(os.path.join(data_path, class_name))
        for file in tqdm(file_names):
            x_deep = []
            vidcap = cv2.VideoCapture(os.path.join(data_path, class_name, file))
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
            df.to_csv(os.path.join(data_path, class_name, file + ".csv"), header=False, index=False)
            vidcap.release()


extract_features_in_x_y_format(TRAIN_DIR)
extract_features_in_x_y_format(TEST_DIR)
extract_features_in_video_format(VALID_DIR)
