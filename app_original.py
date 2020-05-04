from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from imageio import imsave, imread
from PIL import Image
import numpy as np
import keras.models
import re
import sys 
import os
import base64

sys.path.append(os.path.abspath("./model"))
from load import * 

# from scipy.misc import imsave, imread, imresize
# from imageio import imsave, imread, imresize
# import tensorflow.keras.models
# import tensorflow as tf


app = Flask(__name__)
app.config['CORS_HEADERS'] = "Content-Type"
cors = CORS(app,resources={r"/foo": {"origins": "*"}})

global graph, model , sess
model, graph , sess = init()

@app.route('/')
def index_view():
    return render_template('index.html')

def convertImage(imgData1):
    print("convertImage")
    imgreg = re.search(b'base64,(.*)',imgData1)
    imgstr = imgreg.group(1)
    imgstr_64 = base64.b64decode(imgstr)
    with open('output.png','wb') as output:
            output.write(imgstr_64)


@app.route('/predict/',methods=['GET','POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def predict():
    imgData = request.get_json().encode()
    convertImage(imgData)
    imgFromOutPut = imread('output.png' , pilmode = "L")
    imgInvert = np.invert(imgFromOutPut)
    imgResize = np.array(Image.fromarray(imgInvert).resize(size=(28,28)))
    imgReshape = imgResize.reshape(1,28,28,1)

    with graph.as_default():
        set_session(sess)
        out = model.predict(imgReshape)
        response = np.array_str(np.argmax(out,axis=1))
        print(response)
        return response	

if __name__ == '__main__':
    app.run(debug=True, port=8000)
