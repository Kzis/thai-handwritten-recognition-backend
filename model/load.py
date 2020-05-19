
from imageio import imsave, imread
from PIL import Image
import numpy as np

import tensorflow as tf
# tf.enable_eager_execution()

import keras
import tensorflow.keras.models
from keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
from tensorflow.python.framework import ops

session = tf.Session()

def init(): 
    # init and clear session tf keras
    init = tf.global_variables_initializer()
    session = keras.backend.get_session()
    session.run(init)

    # set default graph
    graph = tf.get_default_graph()

    # load model
    model_thw = load_model_thw()

    return model_thw,graph,session

def load_model_thw():
    json_file = open('./model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #load weights into new model
    loaded_model.load_weights("model.h5")

    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return loaded_model

