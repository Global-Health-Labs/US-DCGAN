"""
Provide feature confidence score using a pre-trained binary classifier

Copyright (c) 2023 Global Health Labs, Inc
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
import os
import time
import sys
import glob
import random
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from CNN.data_generator import DataGeneratorMemory

__all__=['positive_score']

def positive_score(val_dir,batch_size,model_path,img_dir=None):
    
    if img_dir==None:
        validation_images = glob.glob(val_dir + '*.jpg')
    else:
        img = val_dir+img_dir
        validation_images = [img,img]
    validation_labels = [1 for image in validation_images]
    validation_labels[0] = 0 # add first as dummy label
    
    IMG_SHAPE = (256, 256, 1)
    batch_size = 16
    validation_datagen = DataGeneratorMemory(validation_images,
                                        validation_labels,
                                        IMG_SHAPE,
                                        batch_size=batch_size,
                                        n_classes=2,
                                        shuffle=True, balanced=False
                                            )
    model = keras.models.load_model(model_path)
    model.training = False
    model.predict_generator(validation_datagen)
    scores = model.predict_generator(validation_datagen)
    return scores
