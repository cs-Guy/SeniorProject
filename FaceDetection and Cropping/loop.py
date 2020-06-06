# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:06:35 2020

@author: Guy
"""
from imutils.face_utils import FaceAligner
import openface.openface.align_dlib as openface
import dlib
import numpy as np
import cv2

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
for i in range(20):
    a = i+1
    image = cv2.imread("./test/{}.png".format(a))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (150, 150)), 1.0,(150, 150), (104.0, 177.0, 123.0))
        
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
        alignedFace = face_aligner.align(534, gray, dlibBox, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        alignedFace = cv2.resize(alignedFace,(150,150),interpolation = cv2.INTER_AREA)
    img_name = "processedTest/{}.png".format(a)
    cv2.imwrite(img_name, alignedFace) #image[startY:endY, startX:endX]
    print("{} written!".format(img_name))

    
cv2.destroyAllWindows()
# vs.stream.stream.release()

