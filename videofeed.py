# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dlib
import os

VIDEO_PATH = './videos'
TEST_PATH = './tests'

detector = dlib.get_frontal_face_detector();

for vid in os.listdir(VIDEO_PATH):
    cnt = 0
    print('Video name:' , vid)
    capture = cv2.VideoCapture(os.path.join(VIDEO_PATH, vid))
    while (capture.isOpened()):
        ret, frame = capture.read()
        frame = cv2.transpose(frame)
        frame = cv2.resize(frame, (540,960))
        cv2.imshow('videofeed', frame)
        faces = detector(frame, 1)
        print(cnt)
        if len(faces) > 0:
            cnt = cnt + 1
            face = faces[0]
            left = face.left()
            if left < 0:
                left = 0
            top = face.top()
            if top < 0:
                top = 0
            right = face.right()
            bottom = face.bottom()
            crop_img = frame[top:bottom, left:right]
            filename = str(cnt) + '.jpg'
            cv2.imwrite(os.path.join(TEST_PATH, vid.split('_')[0],filename), crop_img)
            
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    capture.release()
    print('Has cropped ', cnt, ' images of ', vid.split('_')[0])

cv2.destroyAllWindows()