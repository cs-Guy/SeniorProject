from django.apps import AppConfig
from keras.models import model_from_json
import os

class PredictionConfig(AppConfig):
    name = 'prediction'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifier_folder = os.path.join(BASE_DIR,'prediction/model/')
    classifier_file = os.path.join(classifier_folder,'model.json')
    weight_file = os.path.join(classifier_folder,'model.h5')
    json_file = open(classifier_file, 'r')
    # load model
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weight_file)

    

    