from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import cv2
from PIL import Image as im
import base64

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("home.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    uploaded_file=request.files['file']
    
    if not uploaded_file:
        return render_template('home.html',error_message='Please upload a image!')
    
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    grey_filter=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    invert=cv2.bitwise_not(grey_filter)
    blur_image=cv2.GaussianBlur(invert,(21,21),0)
    invertblur_image=cv2.bitwise_not(blur_image)
    pencil_sketch=cv2.divide(grey_filter,invertblur_image,scale=256.0)
    
    
    _,data = cv2.imencode('.jpg', pencil_sketch)
    base64_image = base64.b64encode(data).decode('utf-8')
    
    return render_template('result.html', image_data=base64_image)

    

if __name__ == '__main__':
    app.run(debug=True)
    