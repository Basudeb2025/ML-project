import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 2)
offset = 20
imsize = 300

folder = "Datas/fuck"
count = 0
while True:
    succes , img = cap.read()
    hands , img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgwhite = np.ones((imsize,imsize,3),np.uint8)*255
        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgaecropshape = imgcrop.shape
        imgwhite[0:imgaecropshape[0],0:imgaecropshape[1]] = imgcrop
        expectRatio = h/w
        if expectRatio > 1:
            k = imsize/h
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop,(wcal,imsize))
            imgresizeshape = imgresize.shape
            imgwhite[0:imgresizeshape[0],0:imgresizeshape[1]] = imgresize
      #  cv2.imshow('cut',imgcrop)
        else:
            k = imsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (hcal, imsize))
            imgresizeshape = imgresize.shape
            imgwhite[0:imgresizeshape[0], 0:imgresizeshape[1]] = imgresize
        cv2.imshow('white',imgwhite)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/images_{time.time()}.jpg',imgwhite)
        print(count)


