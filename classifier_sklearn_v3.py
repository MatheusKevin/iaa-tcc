import os
import gc
import pickle
import numpy as np
import pandas as pd

from sklearn import svm, metrics
from collections import Counter

MODEL_DIR = 'D:\\TCC_IAA'
DATA_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\deep_features'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'validation')


def generate_svm_classifier():
    print('Carregando base de treino')
    x_train = pd.read_csv(os.path.join(TRAIN_DIR, 'features.csv'), header=None).to_numpy()
    y_train = pd.read_csv(os.path.join(TRAIN_DIR, 'labels.csv'), header=None).to_numpy()
    y_train = np.ravel(y_train)

    print('Iniciando treinamento')
    clf = svm.SVC(kernel='linear', verbose=True)
    clf.fit(x_train, y_train)
    del x_train
    gc.collect()

    print('Carregando base de teste')
    x_test = pd.read_csv(os.path.join(TEST_DIR, 'features.csv'), header=None).to_numpy()
    y_test = pd.read_csv(os.path.join(TEST_DIR, 'labels.csv'), header=None).to_numpy()
    y_test = np.ravel(y_test)

    print('Gerando relatório')
    count_test = Counter(y_test)
    count_train = Counter(y_train)
    labels = pd.read_csv(os.path.join(TEST_DIR, 'class_names.csv'), header=None).to_numpy()
    print('\nClasse | Treinamento | Teste')
    for class_num, class_name in labels:
        print(class_name + ' | %d | %d' % (count_train[class_num], count_test[class_num]))

    print('\nTreinamento: %d' % len(y_train))
    print('Teste: %d' % len(y_test))

    y_pred = clf.predict(x_test)
    print("\nAccuracy: %f" % metrics.accuracy_score(y_test, y_pred))
    print("Recall: %f" % metrics.recall_score(y_test, y_pred, average='weighted'))
    print("F1 score: %f \n" % metrics.f1_score(y_test, y_pred, average='weighted'))

    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    file_matrix = open(os.path.join(MODEL_DIR, 'conf_matrix.pkl'), 'wb')
    pickle.dump(conf_matrix, file_matrix)
    print(conf_matrix)

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