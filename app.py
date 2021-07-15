# -*- coding: utf-8 -*-


from flask import Flask,redirect, url_for, request, render_template
import os
import sys
import os
import glob
import re
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2
# Define a flask app


#UPLOAD_FOLDER ='/home/vijay/Music/dash/uploads'
#ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
name = {
            0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No Passing (for any vehicle type)', 
            10:'No passing veh over 3.5 tons', 
            11:'Priority to through-traffic at the next intersection/crossroads only', 
            12:'Priority road', 
            13:'Yield to cross traffic', 
            14:'Stop', 
            15:'No vehicles of any kind permitted', 
            16:'Veh > 3.5 tons prohibited', 
            17:'Do Not Enter', 
            18:'General caution', 
            19:'Dangerous curve to the left', 
            20:'Dangerous curve to the right', 
            21:'Double curves, first to left (Slow & stay to the right)', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Construction area', 
            26:'Traffic signals', 
            27:'Pedestrians crossing', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals possible', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right of traffic barrier/divider', 
            39:'Keep left of traffic barrier/divider', 
            40:'Roundabout(Yield to traffic in circle, signal only on exit)', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
# Model saved with Keras model.save()

#model.predict()     # Necessary
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

def final_predict(img):
    # Name extracted from https://en.wikipedia.org/wiki/Road_signs_in_Germany
    MODEL_PATH = "/home/vijay/Fedora/Music/dash/traffic_jul_1.h5"

    # Load your trained model
    model = load_model(MODEL_PATH)
    raw=[]
    
    image = Image.open(img)
    image = image.convert('L')
    image = image.resize((32,32))
    image = np.array(image)
    image = cv2.equalizeHist(image)
    raw.append(np.array(image))
    processed_image_array = np.array(raw)
    processed_image_array = processed_image_array.reshape(processed_image_array.shape[0],processed_image_array.shape[1],processed_image_array.shape[2],1)
    Y_pred = np.argmax(model.predict(processed_image_array), axis=-1)
    return Y_pred

    
    
    
    
   # image = Image.open(img)
    #image = image.asarray(image)
   # image = image.convert("RGB")
    #image = image.resize((30,30))
    #image = preprocessing(image)
    #data.append(np.array(image))
    #X_test = np.array(data)
    #X_test = X_test.reshape(1,30,30,1)
    #Y_pred = np.argmax(model.predict(X_test), axis=-1)
    #print("Predicted traffic sign is:", name[Y_pred[0]])
    #plt.imshow(image)
    #plt.show()
   

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        #Save the file to ./uploads

        file_path = secure_filename(f.filename)
        #f.save(os.path.join('UPLOAD_FOLDER',file_path))
        f.save(file_path)
        preds = final_predict(file_path)
        
        preds_num = preds[0]
        Predicted = name[preds_num]  # Convert to string
        os.remove(file_path)
        return Predicted
    return None


if __name__ == '__main__':
    app.run(debug=True)
