# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from model import create_model
from align import AlignDlib
from helper import get_aligned, draw_box, save_results
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import datetime

### CONSTANTS ###
IMAGES_PATH = './images'
KNN_MODEL_PATH = './models/knn.dat'
RESULT_PATH = './results/results.txt'
ENCODER_PATH = './models/encoder.dat'
SMALL_SIZE = (640,480)
threshold = 0.45

### FUNCTIONS ###
""" FUNC: get_prediction()
DESCRIPTION:
    Extrace OpenFace feature from an aligned image and predict the result by KNN model
INPUT:
    aligned - An aligned image
    model - OpenFace model
    knn - KNN model
    get_distance - return the distance param or not
OUTPUT:
    (prediction) - If get_distance param = False
    (prediction, distance) - If get_distance param = True

"""
def get_prediction(aligned, model, knn, get_distance=False):
    aligned = aligned/255.
    embedded = model.predict(np.expand_dims(aligned, axis=0))[0]
    distance, result = knn.kneighbors([embedded], return_distance=True)
    pred = knn.predict([embedded])
    if get_distance:
        return (pred, distance)
    else:
        return pred

##################   MAIN   ######################

### LOAD MODELS, ALIGNMENT AND ENCODER ###
model = create_model()
model.load_weights('./weights/nn4.small2.v1.h5')    
alignment = AlignDlib('./models/face_landmarks.dat')
knn = joblib.load(KNN_MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

### INITIALIZE ###
person = []         # List hold all recognized people
timestamp = []      # List hold timestamps corresponding to the recognition
predicts = []       # List hold temporary prediction for a person
last_prediction = time.time()   # Last predicting moment

### RECOGNIZE ###
print('Getting image:')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    face, aligned = get_aligned(frame, alignment)                   # Find largest face and get image aligned
    print(predicts)                                                 # Debugging
    if face is not None:
        pred, distance = get_prediction(aligned, model, knn, True)  # Get prediction and the distance parameter
        print(distance,'/',pred)
        
        if distance < threshold:                                    # Only prediction with distance below the threshold is valid
            last_prediction = time.time()                           # Update predicting moment
            draw_box(frame, face, pred, (0,255,0))                  # Draw bounding box and label on the image
            print('Match: ', pred[0])                               # Debugging
            predicts.append(pred[0])                                # Append new temp prediction to the temp list
            if len(predicts) > 20:                                  # Save result when there is 20 temp predictions
                label = encoder.inverse_transform([np.argmax(np.bincount(np.array(predicts)))])[0]  # Get the most common prediction and transform back into label
                if label not in person:                             # Only update the result lists if that person is unavailable
                    person.append(label)
                    timestamp.append(datetime.datetime.now())
                predicts.clear()                                    # Clear the temp predicting list
        else:
            draw_box(frame, face, 'Unknown', (255,255,255))         # Draw white bounding box around an unknown face
            print('Unknown')                                        # Debugging
    else:
        print('No face')                                            # No face detected in the image
        
    if time.time() - last_prediction > 3.0:                         # Clear the temp predicting list if there no new prediction in 3.0s
        last_prediction = time.time()
        predicts.clear()
    
    cv2.imshow('detect', frame)                                     # Show image
    key = cv2.waitKey(5)                                            # Exit with key 'x'
    if key == ord('x'):
        break

### CLEAN AND SAVE TO FILE ###
cv2.destroyAllWindows()
cap.release
save_results(person, timestamp, RESULT_PATH)                        # Save result before terminate the application
