import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
offset = 20
imsize = 300

folder = "Datas/fuck"
classifier = Classifier("models/keras_model.h5","models/labels.txt")
arr = ["A","B","C","Fuck"]

while True:
    succes , img = cap.read()
    imageoutput = img.copy()
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
            predict ,index = classifier.getPrediction(imgwhite)
      #  cv2.imshow('cut',imgcrop)
        else:
            k = imsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (hcal, imsize))
            imgresizeshape = imgresize.shape
            imgwhite[0:imgresizeshape[0], 0:imgresizeshape[1]] = imgresize
            predict ,index = classifier.getPrediction(imgwhite)
        cv2.putText(imageoutput,arr[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
        cv2.imshow('white',imgwhite)

    cv2.imshow("Image",imageoutput)
    cv2.waitKey(1)


