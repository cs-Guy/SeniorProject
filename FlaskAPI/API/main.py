from flask import Flask, request, jsonify, render_template,url_for
import numpy as np
import os
from keras_util import get_matrix
import requests


app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/demo',methods=['GET'])
def demo():
    return render_template('./Demo/index.html',threshold=.7)

@app.route('/upload',methods=['POST'])
def upload():
    if request.method == "POST":
        threshold = request.form.get('threshold')
        print(threshold)
        print(request.files)
        file = request.files.get('id')
        print(file)
        file2 = request.files.get('face')
        print(file2)
        if threshold is None:
            threshold = 0.6
        else:
            threshold = float(threshold)
        if file is None or file.filename == "":
            return jsonify({
                'message': 'no file with label id',
                'file': file.filename
            })
        if file2 is None or file2.filename == "":
            return jsonify({
                'message': 'no file with label face',
                'file': file2.filename
            })
        if not allowed_file(file.filename):
            return jsonify({
                'message': 'format not supported for label id'
            })
        if not allowed_file(file2.filename):
            return jsonify({
                'message': 'format not supported for label face'
            })
        
        try:
            img_bytes_id = file.read()
            img_bytes_face = file2.read()
            distance, result = get_matrix(img_bytes_id,img_bytes_face,threshold)
            return render_template('./Demo/index.html',message="success",threshold=float(threshold),distance= float(distance),result=result)
        except:
            return render_template('./Demo/index.html', message="Error During prediction")



ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        threshold = request.form.get('threshold')
        print(threshold)
        print(request.files)
        file = request.files.get('id')
        print(file)
        file2 = request.files.get('face')
        print(file2)
        if threshold is None:
            threshold = 0.6
        else:
            threshold = float(threshold)
        if file is None or file.filename == "":
            return jsonify({
                'message': 'no file with label id',
                'file': file.filename
            })
        if file2 is None or file2.filename == "":
            return jsonify({
                'message': 'no file with label face',
                'file': file2.filename
            })
        if not allowed_file(file.filename):
            return jsonify({
                'message': 'format not supported for label id'
            })
        if not allowed_file(file2.filename):
            return jsonify({
                'message': 'format not supported for label face'
            })
        
        try:
            img_bytes_id = file.read()
            img_bytes_face = file2.read()
            distance, result = get_matrix(img_bytes_id,img_bytes_face,threshold)
            
            return jsonify({
                'message': 'Success',
                'predicted': result,
                'distance': float(distance),
                'threshold': float(threshold)
            })

        except:
            return jsonify({
                'message': 'error during prediction'
            })
    else :
        return jsonify({
                'message': 'Error: Use POST method'
            })

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80, debug=True)

