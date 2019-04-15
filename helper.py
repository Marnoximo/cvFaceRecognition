# -*- coding: utf-8 -*-
import cv2
import os
from align import AlignDlib

def draw_box(frame, face, text, color):
    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 3)
    cv2.putText(frame, str(text), (face.left(), face.top() - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def get_aligned (frame, alignment):
    #frame = cv2.resize(frame, SMALL_SIZE)
    frame = frame[...,::-1]
    face = alignment.getLargestFaceBoundingBox(frame)
    if face is None:
        return (None, None)
    else:
        aligned = alignment.align(96, frame, face, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    return (face, aligned)

def save_results(person, dtime, path):
    f = open(path, 'w')
    for i in range(len(person)):
        line = str(person[i]) + ',' + str(dtime[i]) + '\n'
        f.write(line)
    f.close()