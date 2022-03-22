import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn import svm, metrics

MODEL_DIR = 'D:\\TCC_IAA'
DATA_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\teste'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'validation')

def generate_svm_classifier():
    train_samples = []
    test_samples = []
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    print('Carregando base de treino')
    for class_num, class_name in enumerate(os.listdir(TRAIN_DIR)):
        count = 0
        for file in tqdm(os.listdir(os.path.join(TRAIN_DIR, class_name))):
            x_deep = pd.read_csv(os.path.join(TRAIN_DIR, class_name, file), header=None).to_numpy()
            num_rows = np.shape(x_deep)[0]
            if x_train is None:
                x_train = x_deep
                y_train = np.array([class_num] * num_rows)
            else:
                x_train = np.concatenate((x_train, x_deep))
                y_train = np.concatenate((y_train, [class_num] * num_rows))

            count = count + num_rows

        train_samples.append(count)

    print('Carregando base de teste')
    for class_num, class_name in enumerate(os.listdir(TEST_DIR)):
        count = 0
        for file in tqdm(os.listdir(os.path.join(TEST_DIR, class_name))):
            x_deep = pd.read_csv(os.path.join(TEST_DIR, class_name, file), header=None).to_numpy()
            num_rows = np.shape(x_deep)[0]
            if x_test is None:
                x_test = x_deep
                y_test = np.array([class_num] * num_rows)
            else:
                x_test = np.concatenate((x_test, x_deep))
                y_test = np.concatenate((y_test, [class_num] * num_rows))

            count = count + num_rows

        test_samples.append(count)

    for class_num, class_name in enumerate(os.listdir(TRAIN_DIR)):
        print(class_name + ': %d, %d' % (train_samples[class_num], test_samples[class_num]))

    print(x_train.shape)
    print(x_test.shape)

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred, average='weighted'))
    print("F1 score: ", metrics.f1_score(y_test, y_pred, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_pred))

    return clf

def determine_pred_class(y):
    return np.bincount(y).argmax()

if os.path.exists(os.path.join(MODEL_DIR, 'clf.pkl')):
    f = open(os.path.join(MODEL_DIR, 'clf.pkl'), 'rb')
    clf = pickle.load(f)
else:
    clf = generate_svm_classifier()
    f = open(os.path.join(MODEL_DIR, 'clf.pkl'), 'wb')
    pickle.dump(clf, f)

# y_data = []
# y_pred = []
#
# for class_num, class_name in enumerate(class_names):
#     file_names = os.listdir(os.path.join(DATA_DIR, class_name))
#
#     for file in tqdm(file_names):
#         x_data = pd.read_csv(os.path.join(DATA_DIR, class_name, file), header=None)
#         x_data = x_data.to_numpy()
#         y_frames_pred = clf.predict(x_data)
#         y_data.append(class_num)
#         y_pred.append(determine_pred_class(y_frames_pred))
#
# print("Accuracy: ", metrics.accuracy_score(y_data, y_pred))