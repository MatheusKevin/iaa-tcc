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

DIC_PATH = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\class_dictionary.csv'
DEST_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\deep_features'
DATA_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\midlevel'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'validation')

df_dict = pd.read_csv(DIC_PATH, header=0, sep=";")

model = keras.models.load_model("modelo.h5")
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers if isinstance(layer, Flatten)]
)
print(feature_extractor.summary())


def get_base_frame_metadata(data_path):
    counter = []

    for index, dict in df_dict.iterrows():
        class_frames = 0
        if os.path.exists(os.path.join(data_path, dict['class_base'])):
            file_names = os.listdir(os.path.join(data_path, dict['class_base']))
            # for file in tqdm(file_names):
            for file in file_names:
                vidcap = cv2.VideoCapture(os.path.join(data_path, dict['class_base'], file))
                success, img = vidcap.read()
                while success:
                    class_frames = class_frames + 1
                    success, img = vidcap.read()

                vidcap.release()

        print(dict['class_all_descr'], class_frames)
        counter.append(class_frames)

    return np.array(counter)


def extract_features_in_xy_format(data_path, frame_limit=None):
    count_frame = get_base_frame_metadata(data_path)
    print(count_frame)

    part = data_path.split('\\')[-1]
    if not os.path.exists(os.path.join(DEST_DIR, part)):
        os.makedirs(os.path.join(DEST_DIR, part))

    if frame_limit is None:
        frame_limit = math.trunc(np.mean(count_frame))

    print(frame_limit)

    x_file = open(os.path.join(DEST_DIR, part, 'features.csv'), 'w', newline='')
    x_writer = csv.writer(x_file)
    y_file_all = open(os.path.join(DEST_DIR, part, 'labels_all.csv'), 'w', newline='')
    y_wrt_all = csv.writer(y_file_all)
    y_file_grp = open(os.path.join(DEST_DIR, part, 'labels_group.csv'), 'w', newline='')
    y_wrt_grp = csv.writer(y_file_grp)

    for index, dict in df_dict.iterrows():
        if not os.path.exists(os.path.join(data_path, dict['class_base'])):
            continue

        file_names = os.listdir(os.path.join(data_path, dict['class_base']))

        step = count_frame[index] // frame_limit
        step_count = step
        for file in tqdm(file_names):
            vidcap = cv2.VideoCapture(os.path.join(data_path, dict['class_base'], file))
            success, img = vidcap.read()

            while success:
                if step_count == step:
                    img.resize((64, 64, 3))
                    xd = image.img_to_array(img)
                    xd = np.expand_dims(xd, axis=0)
                    deep_features = feature_extractor(xd)

                    x_writer.writerow(np.ravel(deep_features))
                    y_wrt_all.writerow([dict['class_all_id']])
                    y_wrt_grp.writerow([dict['class_group_id']])

                    step_count = 0
                else:
                    step_count = step_count + 1

                success, img = vidcap.read()

            vidcap.release()

    x_file.close()
    y_file_all.close()
    y_file_grp.close()


def prepare_validation_base():
    part = VALID_DIR.split('\\')[-1]

    for index, dict in df_dict.iterrows():
        class_name = dict['class_base']
        if not os.path.exists(os.path.join(VALID_DIR, class_name)):
            continue

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


print(math.trunc(np.mean(get_base_frame_metadata(TEST_DIR))))


# print('Preparando a base de treino')
# extract_features_in_xy_format(TRAIN_DIR)
# print('Preparando a base de teste')
# extract_features_in_xy_format(TEST_DIR)
# print('Preparando a base de validação')
# prepare_validation_base()
