# -*- coding: utf-8 -*-

import cv2

img = cv2.imread('./tests/tien/55.jpg')
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
img2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('result', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
