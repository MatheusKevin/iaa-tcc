import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn import svm, metrics

MODEL_DIR = 'D:\\TCC_IAA'
DATA_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\deep_features'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'validation')

def generate_svm_classifier():
    train_samples = []
    test_samples = []

    x_test = []
    y_test = []

    print('Carregando base de treino')
    x_train = pd.read_csv(os.path.join(TRAIN_DIR, 'x_deep.csv'), header=None).to_numpy()
    y_train = pd.read_csv(os.path.join(TRAIN_DIR, 'y_labels.csv'), header=None).to_numpy()

    print(x_train.shape)
    print(y_train.shape)

    # print('Classe | Treinamento | Teste')
    # for class_num, class_name in enumerate(os.listdir(TRAIN_DIR)):
    #     print(class_name + ' | %d | %d' % (train_samples[class_num], test_samples[class_num]))
    #
    # # print(x_train.shape)
    # # print(x_test.shape)
    # print('\nTreinamento: %d' % len(x_train))
    # print('Teste: %d' % len(x_test))
    #
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # print("\nAccuracy: %f" % metrics.accuracy_score(y_test, y_pred))
    # print("Recall: %f" % metrics.recall_score(y_test, y_pred, average='weighted'))
    # print("F1 score: %f \n" % metrics.f1_score(y_test, y_pred, average='weighted'))
    #
    # conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # file_matrix = open(os.path.join(MODEL_DIR, 'conf_matrix.pkl'), 'wb')
    # pickle.dump(conf_matrix, file_matrix)
    # print(conf_matrix)

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