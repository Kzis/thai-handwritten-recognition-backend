import numpy as np
import keras
import tensorflow.keras.models
from keras.models import model_from_json
# from scipy.misc import imread, imresize,imshow
from imageio import imsave, imread
from PIL import Image
import tensorflow as tf

from tensorflow.python.keras.backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
session = tf.Session()

from tensorflow.python.framework import ops

def init(): 
    session = keras.backend.get_session()
    init = tf.global_variables_initializer()
    session.run(init)
    print("######### Init model #########")
    json_file = open('./model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    print("######### Load model #########")
    print(loaded_model)

    #load weights into new model
    print("######### Load weights into new model #########")
    print(loaded_model)

    loaded_model.load_weights("model.h5")
    print("Loaded Model from disk")
    print("######")

    #compile and evaluate loaded model
    print("######### Compile and evaluate loaded model #########")
    print(loaded_model)
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    # loss,accuracy = model.evaluate(X_test,y_test)
    # print('loss:', loss)
    # print('accuracy:', accuracy)
    graph = tf.get_default_graph()
    # graph = tf.compat.v1.get_default_graph()

    #graph = tf.Graph()


    # graph = ops.reset_default_graph()

    print("######################################################")
    print(loaded_model)
    print(graph)

    return loaded_model,graph,session
