# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:51:30 2020

@author: nipat
"""
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
import numpy as np
import imutils
import dlib
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

# Initiate Webcam
vs = VideoStream(src=0).start()




while True:
    # check for existing images
    path, dirs, files = next(os.walk("./images/"))
    file_count = len(files)
    print(file_count)
    
    # read video frame
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
        confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence < .8:
            continue

		# find box boundary
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
		# draw box around face & probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
         
    k = cv2.waitKey(1)
    
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "images/opencv_frame_{}.jpeg".format(file_count)
        cv2.imwrite(img_name, frame[startY:endY, startX:endX])
        print("{} written!".format(img_name))
    
    cv2.imshow("Frame", frame)
    
cv2.destroyAllWindows()
# vs.stream.stream.release()
vs.stream.release()
# vs.stop()