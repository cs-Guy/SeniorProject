from django.shortcuts import render

# Create your views here.

from django.core.files.storage import FileSystemStorage
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from sklearn import metrics
import numpy as np

json_file = open('./models/model.json', 'r')
# load model
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./models/model.h5')
#make prediction
def predict(img):
  target_image = image.load_img(img, target_size=(224, 224))
  img_pixels = image.img_to_array(target_image)
  img_pixels = np.expand_dims(img_pixels, axis = 0)
  img_pixels /= 255
  result = model.predict(img_pixels)[0,:]

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



# django recieve request
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    pic1 = request.FILES['filePath1']
    pic2 = request.FILES['filePath2']
    input_threshold = request.POST.get('Threshold', None)
    fs=FileSystemStorage()
    path1 = fs.save(pic1.name, pic1)
    path2 = fs.save(pic2.name, pic2)
    
    pathURL1 = '.'+fs.url(path1)
    pathURL2 = '.'+fs.url(path2)
    distance,result = get_matrix(pathURL1,pathURL2,threshold=float(input_threshold))

    context={
        'path1': fs.url(path1),
        'path2': fs.url(path2),
        'distance': distance,
        'result': result,
        'Threshold': float(input_threshold),
        
    }
    return render(request,'index.html',context)

