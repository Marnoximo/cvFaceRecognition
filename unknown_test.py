# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import time
from model import create_model
from align import AlignDlib
from helper import get_aligned
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

IMAGES_PATH = './images'
UNKNOWN_PATH = './tests/unknown'
KNN_MODEL_PATH = './models/knn.dat'
RESULT_PATH = './results/results.txt'
ENCODER_PATH = './models/encoder.dat'
SMALL_SIZE = (640,480)
threshold = 0.566

#########   MAIN   ######################
model = create_model()
model.load_weights('./weights/nn4.small2.v1.h5')    
alignment = AlignDlib('./models/face_landmarks.dat')
knn = joblib.load(KNN_MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

label = ['tan', 'tien', 'vu']
wrong = [0,0,0]
cnt = 0

for i_name in os.listdir(UNKNOWN_PATH):
    frame = cv2.imread(os.path.join(UNKNOWN_PATH, i_name))
    face, aligned = get_aligned(frame, alignment)
    if face is not None:
        cnt = cnt + 1
        print(cnt)
        aligned = aligned/255.
        embedded = model.predict(np.expand_dims(aligned, axis=0))[0]
        distance, result = knn.kneighbors([embedded], return_distance=True)
        pred = knn.predict([embedded])
        #print(distance,'/',result,'/',pred)
        
        if distance < threshold:
            #print('Match: ', knn.predict([embedded]))
            l = encoder.inverse_transform(pred)[0]
            for i in range(3):
                if label[i] == l:
                    wrong[i] = wrong[i] + 1
                    
print(wrong)
print(cnt)
print((wrong[0]+ wrong[1]+wrong[2])/cnt)
