import csv
import pandas as pd
import moviepy.editor as mpy
import cv2
import time
from tqdm import tqdm
import gc
import os

FILE = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\midlevel.chunks_90.csv"
DATASET_PATH = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\drive_and_act\\kinect_color"
DEST_PATH = "C:\\Users\\mathe\\Desktop\\TCC_IAA\\midlevel"

treino = [1, 10]
teste = [11, 12]
validacao = [13, 15]

def obter_parte(part_id):
    if treino[0] <= part_id <= treino[1]:
        return 'train'
    elif teste[0] <= part_id <= teste[1]:
        return 'test'
    elif validacao[0] <= part_id <= validacao[1]:
        return 'validation'
    else:
        return ''

# get anotations per video
# video, anotation id, frame start, frame end, label (activity)
# per anotation id  return first frame start, and last frame_end
def get_full_length_annotations_per_video(video_name, anotations) -> list:
    dataset = anotations[anotations['file_id'] == video_name]
    # gera lista de todos os annotations do video
    # ira retornar  file, activity, start_end ( from the whole activity), annotation_id
    annotations_ids = dataset['annotation_id'].unique()
    lista = []

    for annotation_id in annotations_ids:
        # pega o annotation_id
        temp = dataset[dataset['annotation_id'] == annotation_id]
        # print(temp)
        # ( video_name, activity, start, end, annotation_id)
        lista.append((temp['file_id'].iat[0], temp['activity'].iat[0], temp['frame_start'].iat[0],
                      temp['frame_end'].iat[-1], annotation_id, temp['participant_id'].iat[0]))
    return lista

anotations = pd.read_csv(FILE, sep=";")

for video in tqdm(anotations['file_id'].unique()):

    print("Processando VIDEO: {}".format(video))
    labels = get_full_length_annotations_per_video(video, anotations)

    if len(labels):
        label = labels.pop(0)
    else:
        raise Exception("Video sem labels")

    cap = cv2.VideoCapture(DATASET_PATH + "\\" + video + ".mp4")
    parte = obter_parte(label[5])
    if not parte:
        raise Exception('Parte não configurada')

    framecount = 0
    clip = []
    end = 9999
    while True:
        framecount += 1
        if framecount > end and len(clip) > 0:

            # troca o label que esta procurando
            # salva o clip
            # print(len(clip))
            clip_to_save = mpy.ImageSequenceClip(clip, fps=15)
            # save as vp1_activity_anotationid
            # label( video_name, activity, start, end, annotation_id)
            video = label[0].split('/')
            # path = DEST_PATH + "\\" + video[0] + f"_{video[1]}" + "_" + label[1] + "_" + f"{label[4]}.mp4"
            if not os.path.exists(DEST_PATH + "\\" + parte + "\\" + label[1]):
                os.makedirs(DEST_PATH + "\\" + parte + "\\" + label[1])

            path = DEST_PATH + "\\" + parte + "\\" + label[1] + "\\" + video[0] + f"_{video[1]}" + "_" + f"{label[4]}.mp4"
            clip_to_save.write_videofile(path, verbose=False, logger=None)
            clip = []
            if len(labels):

                label = labels.pop(0)
            else:
                break
            # nao tem mais labels nao tem pq ficar iterando o video
        activity = label[1]
        start = label[2]
        end = label[3]
        ret, frame = cap.read()
        # print(ret)
        if not ret:
            break
        # print(start,end)
        if framecount >= start and framecount <= end:
            clip.append(frame)
    cap.release()
    gc.collect()