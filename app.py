from flask import Flask, render_template, request , jsonify
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

app = Flask(__name__)
app.config['CORS_HEADERS'] = "Content-Type"
cors = CORS(app,resources={r"/foo": {"origins": "*"}})

global graph, model , sess
model, graph , sess = init()
img_size = 256

def convertImage(imgData1):
    print("convertImage")
    imgreg = re.search(b'base64,(.*)',imgData1)
    imgstr = imgreg.group(1)
    imgstr_64 = base64.b64decode(imgstr)
    with open('output.png','wb') as output:
            output.write(imgstr_64)

@app.route('/',methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def test():
    return "Hello World. test ci cd"


@app.route('/predict/',methods=['GET','POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def predict():
    imgData = request.get_json().encode()
    convertImage(imgData)
    imgFromOutPut = imread('output.png' , pilmode = "L")

    # x = Image.open('output.png')
    # im1 = x.crop((0, 0, 280, 70)).save('output1.png')
    # im2 = x.crop((0, 71, 280, 140)).save('output2.png')
    # im3 = x.crop((0, 141, 280, 210)).save('output3.png')
    # im4 = x.crop((0, 211, 280, 280)).save('output4.png')
    
    imgResize = np.array(Image.fromarray(imgFromOutPut).resize(size=(img_size,img_size)))
    imgReshape = imgResize.reshape(1,img_size,img_size,1)
    imgReshape = imgReshape//255
    
    # imgResize = np.array(Image.fromarray(imgInvert).resize(size=(28,28)))
    # imgReshape = imgResize.reshape(1,28,28,1)
    

    with graph.as_default():
        set_session(sess)
        # out = model.predict(imgReshape)
        out = model.predict(imgReshape)
        # print("out ======")
        # print(out)
        response = get_thai_char_by_idex(out)

        return jsonify({'response': response})	

def get_thai_char_by_idex(arr_idx):
    thai = ['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ',
            'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ',
            'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'อะ',
            'อา', 'อำ', 'ฮ', 'ฯ', 'เ', 'แ', 'โ', 'ใ', 'ไ', 'ๆ']

    result = []
    idx = arr_idx[0]
    # print("idx =========")
    # print(idx)

    for i in range(len(thai)):
        result.append((thai[i],idx[i]))

    sorted_by_second = sort_tuple(result)
    sorted_by_invert = sorted_by_second[::-1][:3]

    print("sort =======")
    print(sorted_by_invert)

    t = ""
    for i in range(len(sorted_by_invert)):
        t = t + sorted_by_invert[i][0]  + ":" + str(sorted_by_invert[i][1]) + "|"

    return t


def sort_tuple(tup):  
    tup.sort(key = lambda x: x[1])  
    return tup  


if __name__ == '__main__':
    app.run(debug=True, port=8000)
