import os
import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn import svm, metrics
from collections import Counter

MODEL_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA'
DATA_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\deep_features'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'validation')

clf = None


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    # fig = plt.figure(figsize=figsize)
    fig = plt.figure()
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError('Confusion matrix values must be integers.')

    # heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    # heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), bbox_inches = "tight")

    return fig


def print_heatmap(y_test, y_pred, class_names):
    matrix = metrics.confusion_matrix(y_test, y_pred)
    row_sum = np.sum(matrix, axis=1)
    w, h = matrix.shape

    c_m = np.zeros((w, h))

    for i in range(h):
        c_m[i] = matrix[i] * 100 / row_sum[i]

    c = c_m.astype(dtype=np.uint8)

    heatmap = print_confusion_matrix(c, class_names, figsize=(18, 10), fontsize=20)


def evaluate_classifier(y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('\nAccuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    print('F1 score: %f' % f1)


def plot_class_chart_bar(train_vals, test_vals, class_names):
    width = 0.25
    n = len(train_vals)
    x = np.arange(n)

    fig = plt.figure()
    ay = fig.add_subplot(211)

    plt.xticks(x, class_names, rotation=45)

    ay.bar(x + 0.00, train_vals, width, color="b")
    ay.bar(x + 0.25, test_vals, width, color="g")
    ay.legend(labels=['Treinamento', 'Teste'])

    plt.title('Classes', size=25)
    plt.ylabel('Amostras', size=15)

    fig.savefig(os.path.join(MODEL_DIR, 'classes.png'))


def print_base_info(y_train, y_test, class_names):
    count_test = Counter(y_test)
    count_train = Counter(y_train)
    train_vals = []
    test_vals = []

    print('\nClasse | Treinamento | Teste')
    for class_num, class_name in enumerate(class_names):
        train_vals.append(count_train[class_num])
        test_vals.append(count_test[class_num])
        print(class_name + ' | %d | %d' % (count_train[class_num], count_test[class_num]))

    plot_class_chart_bar(train_vals, test_vals, class_names)


def generate_svm_classifier():
    print('Carregando base de treino')
    x_train = pd.read_csv(os.path.join(TRAIN_DIR, 'features.csv'), header=None).to_numpy()
    y_train = np.ravel(pd.read_csv(os.path.join(TRAIN_DIR, 'labels.csv'), header=None).to_numpy())

    print('Iniciando treinamento')
    svc = svm.SVC(kernel='linear', verbose=True)
    svc.fit(x_train, y_train)
    del x_train
    gc.collect()

    print('Carregando base de teste')
    x_test = pd.read_csv(os.path.join(TEST_DIR, 'features.csv'), header=None).to_numpy()
    y_test = np.ravel(pd.read_csv(os.path.join(TEST_DIR, 'labels.csv'), header=None).to_numpy())

    class_names = np.ravel(pd.read_csv(os.path.join(TEST_DIR, 'class_names.csv'), header=None).to_numpy())
    print_base_info(y_train, y_test, class_names)

    print('\nTreinamento: %d' % len(y_train))
    print('Teste: %d' % len(y_test))

    y_pred = svc.predict(x_test)
    evaluate_classifier(y_test, y_pred)
    print_heatmap(y_test, y_pred, class_names)

    return svc


def determine_pred_class(y):
    return np.bincount(y).argmax()


def predict_validation_videos(clf_model):
    y_valid = []
    y_pred = []

    class_names = os.listdir(VALID_DIR)
    for class_num, class_name in enumerate(class_names):
        file_names = os.listdir(os.path.join(VALID_DIR, class_name))
        for file in tqdm(file_names):
            x_data = pd.read_csv(os.path.join(VALID_DIR, class_name, file), header=None).to_numpy()
            y_frames_pred = clf_model.predict(x_data)
            y_valid.append(class_num)
            y_pred.append(determine_pred_class(y_frames_pred))

    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))


if os.path.exists(os.path.join(MODEL_DIR, 'clf.pkl')):
    f = open(os.path.join(MODEL_DIR, 'clf.pkl'), 'rb')
    clf = pickle.load(f)
else:
    clf = generate_svm_classifier()
    f = open(os.path.join(MODEL_DIR, 'clf.pkl'), 'wb')
    pickle.dump(clf, f)

predict_validation_videos(clf)
