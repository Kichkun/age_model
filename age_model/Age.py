import os

import numpy as np
from keras.layers import Convolution2D, Flatten, Activation
from keras.models import Model

from age_model import VGGFace


def loadModel():
    model = VGGFace.baseModel()

    # --------------------------

    classes = 101
    # base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------

    age_model = Model(inputs=model.input, outputs=base_model_output)

    age_model.load_weights(os.path.join('age_model', 'models', 'age_model_weights.h5'))

    return age_model


# --------------------------

def findApparentAge(age_predictions):
    output_indexes = np.array([i for i in range(0, 101)])
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age
