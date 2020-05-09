# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:34:59 2020

@author: nipat
"""
from imutils.face_utils import FaceAligner
import openface.openface.align_dlib as openface
import dlib
import numpy as np
import cv2
import os

# import face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#import face landmark
alignFile = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(alignFile)
fa = FaceAligner(predictor, desiredFaceWidth=128)
face_aligner = openface.AlignDlib(alignFile)

# read image

image = cv2.imread("./test/test3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# check for existing images
path, dirs, files = next(os.walk("./images/"))
file_count = len(files)
print(file_count)

(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence < .7:
        continue

    # find box boundary
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    #startX = left; startY = top; endX = right; endy = bottom;
    # draw box around face & probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    #cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
    #cv2.putText(image, text, (startX, y),
    #cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    dlibBox = dlib.rectangle(left = startX,top = startY,right = endX,bottom = endY)
    aligned_face = fa.align(image, gray, dlibBox)
    alignedFace = face_aligner.align(534, image, dlibBox, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

img_name = "images/opencv_frame_{}.jpeg".format(file_count)
cv2.imwrite(img_name, alignedFace) #image[startY:endY, startX:endX]
print("{} written!".format(img_name))

cv2.imshow("Frame", aligned_face)
    
cv2.destroyAllWindows()
# vs.stream.stream.release()

