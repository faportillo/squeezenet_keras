from __future__ import print_function
import tensorflow as tf
import keras
import keras.backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
from keras.models import load_model
import os
import math
import numpy as np
import time
from PIL import Image
import random

import train_utils as tu

#Hyperparameters
batch_size = 1024
num_classes = 1000
epochs = 170000


def _fire(x, filters, name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
    expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
    expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    return x

def SqueezeNet(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000):

	#Train from scratch
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="valid")(x)

    x = _fire(x, (16, 64, 64), name="fire2")
    x = _fire(x, (16, 64, 64), name="fire3")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3', padding="valid")(x)

    x = _fire(x, (32, 128, 128), name="fire4")
    x = _fire(x, (32, 128, 128), name="fire5")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding="valid")(x)

    x = _fire(x, (48, 192, 192), name="fire6")
    x = _fire(x, (48, 192, 192), name="fire7")

    x = _fire(x, (64, 256, 256), name="fire8")
    x = _fire(x, (64, 256, 256), name="fire9")

    if include_top:
        x = Dropout(0.5, name='dropout9')(x)

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        x = AveragePooling2D(pool_size=(13, 13), name='avgpool10')(x)
        x = Flatten(name='flatten10')(x)
        x = Activation("softmax", name='softmax')(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D(name="avgpool10")(x)
        else:
            x = GlobalMaxPooling2D(name="maxpool10")(x)

    model = Model(img_input, x, name="squeezenet")

    if weights == 'imagenet':
        weights_path = get_file('squeezenet_weights.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')

        model.load_weights(weights_path)

    return model


'''Train the model'''
def train_model(model):
	imagenet_path = '/HD1/'
	train_img_path = os.path.join(imagenet_path,'Train')
	ts_size = tu.imagenet_size(train_img_path)
	num_batches = int(float(ts_size)/batch_size)

	wnid_labels,_ = tu.load_imagenet_meta(os.path.join(imagenet_path, 'dev_kit/ILSVRC2012_devkit_t12/data/meta.mat'))
	rmsprop = optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0)
	model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
  
        for i in range(0, epochs):
          current_index = 0
          print('Epoch: '+str(i))
          while current_index + batch_size < (ts_size):
            start_time = time.time()
        
            img, lab = tu.read_batch(batch_size, train_img_path,wnid_labels)
            img = np.stack(img,axis=0)
            lab = np.stack(lab,axis=0)
            #print("IMG SHAPE: "+str(img.shape))
            loss, accuracy = model.train_on_batch(img, lab)
            print('Loss: '+str(loss)+', Accuracy: '+str(accuracy))
            end_time = time.time()
            
        
          print('batch {}/{} loss: {} accuracy: {} time: {}ms'.format(int(current_index / batch_size), int(ts_size / batch_size), loss, accuracy, 1000 * (end_time - start_time)))
          model.save('Squeezenet_model.h5')
          current_index = 0
          loss = 0.0
          acc = 0.0
          vs_size = tu.imagenet_size(os.path.join(imagenet_path,'Val'))
          print('epoch {}/{}'.format(i, epochs))
          while current_index + batch_size < vs_size:
            val_img,val_lab = tu.read_validation_batch(batch_size,os.path.join(imagenet_path,'Val'), os.path.join(imagenet_path, 'dev_kit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
            score = model.test_on_batch(val_img,val_lab)
            print('Test batch score:' , score[0])
            print('Test batch accuracy:' ,score[1])
            loss += score[0]
            acc += score[1]
          loss = loss / int(vs_size/batch_size)
          acc = acc / int(vs_size/batch_size)
          print('Val score:',str(loss))
          print('Val accuracy:',str(acc))
          
'''
	Train model
'''
model = SqueezeNet()
train_model(model)




















