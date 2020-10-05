from django.shortcuts import render

# Create your views here.
from django.core.files.storage import FileSystemStorage
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import HttpResponse
import requests


def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    if request.method == 'POST':
        r = requests.post('http://127.0.0.1:8000/api/face_predict/', data=request.POST,files=request.FILES)
        print(request.FILES)
        print(request.POST)
        print(r)
    else:
        return HttpResponse('Please Use Post')
    if r.status_code == 200:
        return render(request,'index.html',r.json())
    return HttpResponse('Could not recieved data')

# Create your views here.
@api_view(['GET', 'POST'])
def api_add(request):
    sum = 0
    response_dict = {}
    if request.method == 'GET':
        # Do nothing
        pass
    elif request.method == 'POST':
        # Add the numbers
        data = request.data
        for key in data:
            sum += data[key]
        response_dict = {"sum": sum}
    return Response(response_dict, status=status.HTTP_201_CREATED)

class Add_Values(APIView):
    def post(self, request, format=None):
        sum = 0
        # Add the numbers
        data = request.data
        for key in data:
            sum += data[key]
        response_dict = {"sum": sum}
        return Response(response_dict, status=status.HTTP_201_CREATED)

from .apps import PredictionConfig
from tensorflow.keras.preprocessing import image
import numpy as np

def predict(img):
        target_image = image.load_img(img, target_size=(224, 224))
        img_pixels = image.img_to_array(target_image)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        result = PredictionConfig.model.predict(img_pixels)[0,:]

        return result

def euclid(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def find_dist(img1, img2):
    pred = predict(img1)
    pred2 = predict(img2)
    dist = euclid(l2_normalize(pred), l2_normalize(pred2))
    return dist

def get_matrix(path1,path2, threshold=0.7):
    temp = find_dist(path1, path2)
    result = (1 if temp < threshold else 0)
    return temp,result     

class Face_Predict(APIView):
    #permission_classes = [IsAuthenticated]
    def post(self, request, format=None):
        pic1 = request.FILES['filePath1']
        pic2 = request.FILES['filePath2']
        input_threshold = request.POST.get('Threshold', None)
        fs=FileSystemStorage()
        path1 = fs.save(pic1.name, pic1)
        path2 = fs.save(pic2.name, pic2)
        
        pathURL1 = '.'+fs.url(path1)
        pathURL2 = '.'+fs.url(path2)

        print(pathURL1)
        print(pathURL2)
        print(input_threshold)
        distance,result = get_matrix(pathURL1,pathURL2,threshold=float(input_threshold))

        context={
            'path1': fs.url(path1),
            'path2': fs.url(path2),
            'distance': distance,
            'result': result,
            'Threshold': float(input_threshold),
            
        }
        
        return Response(context, status=200)

      
    
    



