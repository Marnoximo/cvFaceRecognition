# -*- coding: utf-8 -*-
import cv2
from align import AlignDlib
from helper import draw_box
import os

### CONSTANTS ###
DLIB_LANDMARK_PATH = './models/face_landmarks.dat'
IMAGE_PATH = './images'

### INITIALIZING ###
alignment = AlignDlib(DLIB_LANDMARK_PATH)
capture = cv2.VideoCapture(0)
name = ''
valid = True

### GETTING NAME LOOP ###
while True:
    ## Enter new student name ##
    print("Enter new student's name: ")
    name  = input()
    if name == 'x':
        valid = False
        break
    elif name == '':
        continue
    elif name in os.listdir(IMAGE_PATH):
        print("This name has been available")
        continue
    
    ## Make new folder in ./images
    try:
        os.mkdir(os.path.join('images', name))
    except:
        print('Creation new image folder failed')
        continue
    break

### CAPTURING LOOP ###
img_cnt = 0
while valid:
    ## Capture images, detect faces and crop
    ret, frame  = capture.read()
    face = alignment.getLargestFaceBoundingBox(frame)
    if face is not None:
        crop = frame[max([face.top(),0]):face.bottom(), max([face.left(),0]):face.right()]
        cv2.imwrite(os.path.join(IMAGE_PATH,name , str(img_cnt)+'.jpg'), crop)
        draw_box(frame, face, name, (0, 255, 0))
        img_cnt += 1
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(50)
    if key == ord('x'):
        break
        
### CLEAR ENVIRONMENT
cv2.destroyAllWindows()
capture.release()    