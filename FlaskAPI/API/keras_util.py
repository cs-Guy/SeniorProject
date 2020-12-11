from keras.models import model_from_json, Model
import os
import io
from tensorflow.keras.preprocessing import image as im
from PIL import Image
import numpy as np
import cv2
from imutils.face_utils import FaceAligner
import imutils
import dlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
classifier_folder = os.path.join(BASE_DIR,'API/model/')
classifier_file = os.path.join(classifier_folder,'model.json')
weight_file = os.path.join(classifier_folder,'model.h5')
predictor = dlib.shape_predictor(os.path.join(classifier_folder,"shape_predictor_68_face_landmarks.dat"))
fa = FaceAligner(predictor, desiredFaceWidth=224)
configFile = os.path.join(classifier_folder,"deploy.prototxt.txt")
modelFile = os.path.join(classifier_folder,"res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

json_file = open(classifier_file, 'r')
# load model
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(weight_file)
model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

def predict(img):
    target_image = Image.open(io.BytesIO(img)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    print("loaded image Successfully")

    # get height and width of image and find face from blob
    print(cv_image.shape)
    h,w = cv_image.shape[:2]
    if(h < 300 or w < 300):
        blob = cv2.dnn.blobFromImage(cv2.resize(cv_image,(150,150)), 1.0,(150,150), (104.0, 177.0, 123.0))
    else:
        blob = cv2.dnn.blobFromImage(cv2.resize(cv_image,(300,300)), 1.0,(300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    print(detections.shape[2])
    print("Get blob from image successfully")
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < .5:
            continue
        # find box boundary
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        print(startX)
        print(startY)
        print(endX)
        print(endY)
        print("break")
        break

    dlibBox = dlib.rectangle(left = startX,top = startY,right = endX,bottom = endY)
    aligned_face = fa.align(cv_image, gray, dlibBox)
    aligned_face = cv2.resize(aligned_face, (150,150))
    (h,w) = aligned_face.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(aligned_face,(150,150)), 1.0,(150,150), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    # second loop to crop aligned face
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < .5:
            continue
        # find box boundary
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
    if(startY != -1):
        image = cv2.resize(aligned_face[startY:endY, startX:endX], (224,224))
        image_pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pixels = np.expand_dims(image_pixels, axis = 0)
    else:
        print("Alternate")
        aligned_face = cv2.resize(aligned_face, (224,224))
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        image_pixels = np.expand_dims(aligned_face, axis = 0)
    # end second loop

    image_pixels = cv2.normalize(image_pixels, image_pixels, 0, 255, cv2.NORM_MINMAX)
    print("Normallize successfully")
    result = model.predict(image_pixels)[0,:]

    return result

def euclid(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)

    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def find_dist(img1, img2):
    print("image 1")
    pred = predict(img1)
    print("image 2")
    pred2 = predict(img2)
    dist = euclid(l2_normalize(pred), l2_normalize(pred2))
    return dist

def get_matrix(path1,path2, threshold=0.7):
    temp = find_dist(path1, path2)
    result = (1 if temp < threshold else 0)
    return temp,result     