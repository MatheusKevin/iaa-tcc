import os
import numpy as np
import pandas as pd
import random as rd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_DIR = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\teste"
sample_by_class = 2000

class_names = os.listdir(DATA_DIR)

x_data = []
y_data = []

def get_random_frame_features(frames, max_frames):
    if max_frames > frames.shape[0]:
        return frames.to_numpy()

    rd_frames = rd.sample(range(max_frames), max_frames)
    return np.array(list(map(lambda frame: frames.loc[frame].values.tolist(),rd_frames)))

for class_num, class_name in enumerate(class_names):
    file_names = os.listdir(os.path.join(DATA_DIR, class_name))
    num_videos = len(file_names)
    frames_by_video = sample_by_class // num_videos

    for file in tqdm(file_names):
        df_video = pd.read_csv(os.path.join(DATA_DIR, class_name, file), header=None)
        for frame_features in get_random_frame_features(df_video, frames_by_video):
            x_data.append(frame_features)
            y_data.append(class_num)

x_data = np.array(x_data)
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=.4, random_state=42, stratify=y_data)