from flask import Flask, render_template, request

import os
import shutil
import random
import itertools

import numpy as np
import tensorflow as tf

from keras import backend
from tensorflow import keras


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.metrics import categorical_crossentropy

filename="1.jpg"

from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
# global graph, model
import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
# import pdfplumber
from io import BytesIO
# model, graph = init()
import os;
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import Image
import tweepy

app = Flask(__name__)
upload_folder="C:\\Users\\latika\\iot_app\\static"


def get_model():
    global model
    model = load_model('flood_detection_model.h5')
    print(" * Model loaded!")

def preprocess_image(file):
    img_path = 'evaluate/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    
def tweeting(labels,result,file):
    file_path = 'evaluate/'+ file
    consumer_key ="CVvVnd60bi8qtF07bPCJ7ci3V"
    consumer_secret ="3pbWzBSh1nLn3sEuMmmlGBdK7rTiWD9J5CqXsSbwCPVGgYiyo1"
    access_token ="1107236292167766019-42AEazOu8oEtoDeEo43TR5v3Hp3A9S"
    access_token_secret ="cXtwnk2QnCMM9Zmbw38XTpubDFtbTnh0frR1vvKFcXPuY"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    my_tweet ="Flood Detected!!!!"+"\n"+"Please get help in this area." 
    my_image_path =file_path
    if labels[result]=="Flooding":
        my_status = api.update_with_media(my_image_path, my_tweet) 
        print("Tweet Posted")
        str="Tweet Posted"
        return str
    elif labels[result]=="No Flooding":
        print("No Tweet Posted")
        str="NO Tweet Posted"
        return str


print(" * Loading Keras model...")
get_model()


@app.route('/',methods=['GET','POST'])
def upload_predict():
    if request.method=="POST":
        image_file=request.files["image"]
        if image_file:
            image_location=os.path.join(upload_folder,image_file.filename)
            image_file.save(image_location)
            filename = image_file.filename
            
            processed_image = preprocess_image(filename)
            prediction = model.predict(processed_image).tolist()
            result = np.argmax(prediction)
            labels = ['Flooding', 'No Flooding']
            p=labels[result]
            tweet=tweeting(labels,result,filename)
            return render_template('index1.html',prediction=p,filename=filename,result=tweet)
        return render_template('index1.html',prediction=-1)
    return render_template('index1.html',prediction=0)


if __name__ == '__main__':
    app.run(debug=True, port=8000)