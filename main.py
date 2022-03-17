import os
import pickle
import numpy as np
import pandas as pd
import random as rd

from tqdm import tqdm
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

DATA_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\deep_features'
MODEL_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA'
sample_by_class = 2000

class_names = os.listdir(DATA_DIR)

def get_random_frame_features(frames, max_frames):
    if max_frames > frames.shape[0]:
        return frames.to_numpy()

    rd_frames = rd.sample(range(max_frames), max_frames)
    return np.array(list(map(lambda frame: frames.loc[frame].values.tolist(),rd_frames)))

def generate_svm_classifier():
    class_samples = []
    x_data = []
    y_data = []

    for class_num, class_name in enumerate(class_names):
        file_names = os.listdir(os.path.join(DATA_DIR, class_name))
        num_videos = len(file_names)
        frames_by_video = sample_by_class // num_videos
        count = 0

        for file in tqdm(file_names):
            df_video = pd.read_csv(os.path.join(DATA_DIR, class_name, file), header=None)
            for frame_features in get_random_frame_features(df_video, frames_by_video):
                x_data.append(frame_features)
                y_data.append(class_num)
                count = count + 1

        class_samples.append(count)

    for class_num, class_name in enumerate(class_names):
        print(class_name + ': %d' % class_samples[class_num])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print(x_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=.4, random_state=42, stratify=y_data)

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    return clf

def determine_pred_class(y):
    return np.bincount(y).argmax()

if os.path.exists(os.path.join(MODEL_DIR, 'clf.pkl')):
    f = open(os.path.join(MODEL_DIR, 'clf.pkl'), 'rb')
    clf = pickle.load(f)
else:
    f = open(os.path.join(MODEL_DIR, 'clf.pkl'), 'wb')
    clf = generate_svm_classifier()
    pickle.dump(clf, f)

y_data = []
y_pred = []

for class_num, class_name in enumerate(class_names):
    file_names = os.listdir(os.path.join(DATA_DIR, class_name))

    for file in tqdm(file_names):
        x_data = pd.read_csv(os.path.join(DATA_DIR, class_name, file), header=None)
        x_data = x_data.to_numpy()
        y_frames_pred = clf.predict(x_data)
        y_data.append(class_num)
        y_pred.append(determine_pred_class(y_frames_pred))

print("Accuracy: ", metrics.accuracy_score(y_data, y_pred))