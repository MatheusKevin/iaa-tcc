import os
import csv
import cv2
import math
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
class_ignore = ['closing_backpack', 'looking_back_left_shoulder', 'looking_back_right_shoulder', 'opening_backpack',
                'standing_by_the_door', 'opening_backpack']


model = keras.models.load_model("modelo.h5")
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers if isinstance(layer, Flatten)]
)
print(feature_extractor.summary())


def get_base_frame_metadata(data_path):
    class_names = [cl_name for cl_name in os.listdir(data_path) if cl_name not in class_ignore]
    counter = []

    for class_num, class_name in enumerate(class_names):
        file_names = os.listdir(os.path.join(data_path, class_name))
        class_frames = 0
        for file in tqdm(file_names):
            vidcap = cv2.VideoCapture(os.path.join(data_path, class_name, file))
            success, img = vidcap.read()
            while success:
                class_frames = class_frames + 1
                success, img = vidcap.read()

            vidcap.release()

        counter.append(class_frames)

    return np.array(counter)


def prepare_train_base(frame_limit=None):
    part = TRAIN_DIR.split('\\')[-1]
    if not os.path.exists(os.path.join(DEST_DIR, part)):
        os.makedirs(os.path.join(DEST_DIR, part))

    if frame_limit is None:
        frame_limit = math.trunc(np.mean(count_frame))

    print(frame_limit)

    x_file = open(os.path.join(DEST_DIR, part, 'features.csv'), 'w', newline='')
    x_writer = csv.writer(x_file)
    y_file = open(os.path.join(DEST_DIR, part, 'labels.csv'), 'w', newline='')
    y_writer = csv.writer(y_file)

    class_names = [cl_name for cl_name in os.listdir(TRAIN_DIR) if cl_name not in class_ignore]
    pd.DataFrame(class_names).to_csv(os.path.join(DEST_DIR, part, 'class_names.csv'), header=False, index=True)

    for class_num, class_name in enumerate(class_names):
        step = count_frame[class_num] // frame_limit
        step_count = step

        class_frames = 0

        file_names = os.listdir(os.path.join(TRAIN_DIR, class_name))
        for file in tqdm(file_names):
            vidcap = cv2.VideoCapture(os.path.join(TRAIN_DIR, class_name, file))
            success, img = vidcap.read()

            while success:
                if step_count == step:
                    img.resize((64, 64, 3))
                    xd = image.img_to_array(img)
                    xd = np.expand_dims(xd, axis=0)
                    deep_features = feature_extractor(xd)

                    x_writer.writerow(np.ravel(deep_features))
                    y_writer.writerow([class_num])

                    step_count = 0
                else:
                    step_count = step_count + 1

                success, img = vidcap.read()

            vidcap.release()

    x_file.close()
    y_file.close()


def prepare_test_base():
    part = TEST_DIR.split('\\')[-1]
    if not os.path.exists(os.path.join(DEST_DIR, part)):
        os.makedirs(os.path.join(DEST_DIR, part))

    x_file = open(os.path.join(DEST_DIR, part, 'features.csv'), 'w', newline='')
    x_writer = csv.writer(x_file)
    y_file = open(os.path.join(DEST_DIR, part, 'labels.csv'), 'w', newline='')
    y_writer = csv.writer(y_file)

    class_names = [cl_name for cl_name in os.listdir(TEST_DIR) if cl_name not in class_ignore]
    pd.DataFrame(class_names).to_csv(os.path.join(DEST_DIR, part, 'class_names.csv'), header=False, index=True)

    for class_num, class_name in enumerate(class_names):
        file_names = os.listdir(os.path.join(TEST_DIR, class_name))
        for file in tqdm(file_names):
            vidcap = cv2.VideoCapture(os.path.join(TEST_DIR, class_name, file))
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


def prepare_validation_base():
    part = VALID_DIR.split('\\')[-1]
    class_names = [cl_name for cl_name in os.listdir(VALID_DIR) if cl_name not in class_ignore]

    for class_name in class_names:
        if not os.path.exists(os.path.join(DEST_DIR, part, class_name)):
            os.makedirs(os.path.join(DEST_DIR, part, class_name))

        file_names = os.listdir(os.path.join(VALID_DIR, class_name))
        for file in tqdm(file_names):
            x_deep = []
            vidcap = cv2.VideoCapture(os.path.join(VALID_DIR, class_name, file))
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

            file_name = '.'.join(file.split('.')[:-1] + ['csv'])
            pd.DataFrame(x_deep).to_csv(os.path.join(DEST_DIR, part, class_name, file_name), header=False, index=False)
            vidcap.release()


print('Analisando a base de vídeos')
count_frame = get_base_frame_metadata(TRAIN_DIR)
print(count_frame)
print('Preparando a base de treino')
prepare_train_base()
print('Preparando a base de teste')
prepare_test_base()
print('Preparando a base de validação')
prepare_validation_base()
