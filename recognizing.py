# -*- coding: utf-8 -*-

from model import create_model
from align import AlignDlib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMAGES_PATH = './images'
TIEN_PATH = './images/Tien'
TAN_PATH = './images/Tan'
DLIB_LANDMARK_PATH = './models/face_landmarks.dat'


nn4_sv2 = create_model()
nn4_sv2.load_weights('weights/nn4.small2.v1.h5')
print('Created model and loaded weights')


alignment = AlignDlib(DLIB_LANDMARK_PATH)

def get_aligned (img, alignment):
    fbb = alignment.getLargestFaceBoundingBox(img)
    if fbb is None:
        return None
    else:
        return alignment.align(96, img, fbb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

embedded = []
names = []

for dir in os.listdir(IMAGES_PATH):
    #if dir == 'unknown':
     #   continue
    for i in os.listdir(os.path.join(IMAGES_PATH, dir)):
        image = cv2.imread(os.path.join(IMAGES_PATH, dir, i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image[:,:,0] = cv2.equalizeHist(image[:,:,0])
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        image = image[...,::-1]
        
        image = get_aligned(image, alignment)
        if image is None:
            print('Failed: ', i)
            continue
        else:
            image = image/255.
            embedded.append(nn4_sv2.predict(np.expand_dims(image, axis=0))[0])
            names.append(dir)
        
print('Get all images aligned')

##############################################################
distances = []
identical = []
cnt = len(embedded)


for i in range(cnt):
    for j in range(cnt):
        distances.append(np.sum(np.square(embedded[i] - embedded[j])))
        identical.append(1 if names[i]==names[j] else 0)        
distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.2, distances.max(), 0.01)
f1scores = [f1_score(identical, distances < t) for t in thresholds]
accscores = [accuracy_score(identical, distances < t) for t in thresholds]
optimal_idx = np.argmax(f1scores)
optimal_t = thresholds[optimal_idx]
optimal_acc = accuracy_score(identical, distances < optimal_t)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(thresholds, f1scores, label='F1')
ax.plot(thresholds, accscores, label='ACC')
print("Optimal acc= ", optimal_acc, ' Optimal threshold= ', optimal_t)
X_embedded = TSNE(n_components=2).fit_transform(embedded)
plt.figure
names = np.array(names)
for i, t in enumerate(set(names)):
    idx = np.where(names == t)
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)
plt.legend(bbox_to_anchor=(1, 1))


names = np.array(names)
encoder = LabelEncoder()
dummy_names = encoder.fit_transform(names)
features = np.array(embedded)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, dummy_names, test_size=0.3)
#Itrain, Itest = StratifiedShuffleSplit(n_splits=1, test_size=0.4).split(features, dummy_names)[0]
#Xtrain, Ytrain = features[Itrain], dummy_names[Itrain]
#Xtest, Ytest = features[Itest], dummy_names[Itest]
svc = LinearSVC()
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

knn.fit(Xtrain, Ytrain)
svc.fit(Xtrain, Ytrain)

acc_knn = accuracy_score(Ytest, knn.predict(Xtest))
acc_svc = accuracy_score(Ytest, svc.predict(Xtest))

print('KNN: ', acc_knn, ' SVC: ', acc_svc)
#######################################################################

print('Test with webcam')
tien = []
tan = []
for i in os.listdir(TAN_PATH):
    image = cv2.imread(os.path.join(TAN_PATH, i))
    image = image[...,::-1]
    image = get_aligned(image, alignment)
    if image is None:
        continue
    else:
        image = image/255.
        tan.append(nn4_sv2.predict(np.expand_dims(image, axis=0))[0])

cam = cv2.VideoCapture(0)

def predict_image (im, model):
    face = alignment.getLargestFaceBoundingBox(im)
    if face is None:
        print('Noface')
        cv2.imshow('Camfeed', im)
        return
    else:
        clone = im[face.top():face.bottom(),face.left():face.right()].copy()
        #cv2.rectangle(clone, (face.top(), face.bottom()), (face.left(), face.right()), (0,255,0), 2)
        cv2.imshow('CamFeed', clone)
        im = im[...,::-1]
        im = alignment.align(96, im, face, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        im = im/255.
        embedded_vector = model.predict(np.expand_dims(im, axis=0))[0]
        result = 0
        for i in tan:
            result = result + np.sum(np.square(embedded_vector - i))
        result = result/len(tan)
        print('Probability: ', result)
        
while True:
    ret, im = cam.read()
    predict_image(im, nn4_sv2)
    key = cv2.waitKey(0)
    if key == ord('x'):
        break

cv2.destroyAllWindows()
