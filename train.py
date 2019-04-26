# -*- coding: utf-8 -*-

from model import create_model
from align import AlignDlib
from helper import get_aligned
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import cv2
import os
import time

############# CONSTANTS ###############
IMAGES_PATH = './images'
DLIB_LANDMARK_PATH = './models/face_landmarks.dat'
KNN_MODEL_PATH = './models/knn.dat'


############## MAIN #################

### INITIALIZING ###
nn4_sv2 = create_model()
nn4_sv2.load_weights('weights/nn4.small2.v1.h5')
alignment = AlignDlib(DLIB_LANDMARK_PATH)
print('Loaded OpenFace model and alignment')

start_time = time.time()

### EXTRACT FEATURES ###
embedded = []
names = []
for dir in os.listdir(IMAGES_PATH):
    if dir[0] == '0':   #Debugging
        continue
    for i in os.listdir(os.path.join(IMAGES_PATH, dir)):
        image = cv2.imread(os.path.join(IMAGES_PATH, dir, i))
        image = image[...,::-1]
        face, image = get_aligned(image, alignment)
        print(dir, "/", i)
        if image is None:
            print('Failed: ', i)
            continue
        else:
            image = image/255.
            embedded.append(nn4_sv2.predict(np.expand_dims(image, axis=0))[0])
            names.append(dir)
print('Got all features')

### FIND THRESHOLD - (OPTIONAL - ONLY FOR ANALYZING) ###
distances = []
identical = []
cnt = len(embedded)
thresholds = np.arange(0.2, 1.0, 0.01)

for i in range(cnt):
    for j in range(cnt):
        distances.append(np.sum(np.square(embedded[i] - embedded[j])))
        identical.append(1 if names[i]==names[j] else 0)        
distances = np.array(distances)
identical = np.array(identical)
f1scores = [f1_score(identical, distances < t) for t in thresholds]
accscores = [accuracy_score(identical, distances < t) for t in thresholds]
optimal_idx = np.argmax(f1scores)
optimal_t = thresholds[optimal_idx]
optimal_acc = accuracy_score(identical, distances < optimal_t)
print("Optimal acc= ", optimal_acc, ' Optimal threshold= ', optimal_t)

### ENCODE LABELS AND PREPARE INPUT ARRAYS
features = np.array(embedded)
names = np.array(names)
encoder = LabelEncoder()
dummy_names = encoder.fit_transform(names)
joblib.dump(encoder, './models/encoder.dat')

### SPLIT TRAIN/TEST SET
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, dummy_names, test_size=0.3)

### KNN MODEL ###
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(Xtrain, Ytrain)
acc_knn = accuracy_score(Ytest, knn.predict(Xtest))
print('KNN: ', acc_knn)
joblib.dump(knn, './models/knn.dat')

### SVC MODEL ###
#svc = LinearSVC()
#svc.fit(Xtrain, Ytrain)
#acc_svc = accuracy_score(Ytest, svc.predict(Xtest))
#joblib.dump(svc, './models/svc.dat')

print('time: ', time.time() - start_time)