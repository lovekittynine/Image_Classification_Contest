#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:14:53 2018

@author: wsw
"""

# build model
from .resnet50 import ResNet50
from .densenet201 import DenseNet201
#from .inception_v3 import InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
# from .Xception import Xception
from keras.layers import Dense
from keras.models import Model
from keras.constraints import unitnorm
from keras.layers import Lambda
import keras.backend as K


def build_resnet_model(input_shape=[224,224,3],num_class=6,bilinear=True):
  # construct model with CBAM
  resnet50 = ResNet50(include_top=False,
                      input_shape=input_shape,
                      weights=None,
                      pooling='avg',
                      bilinear=bilinear)
  x = resnet50.output
  # add l2 norm to feature and weights
  # l2 normalization feature
  x = Lambda(lambda x:K.l2_normalize(x,axis=-1))(x)
  # add l2 normalization to weights
  x = Dense(num_class,activation=None,
            kernel_constraint=unitnorm(axis=0),
            use_bias=False,
            name='predictions')(x)
  model = Model(inputs=resnet50.input,outputs=x)
  return model


def build_densenet_model(input_shape=[224,224,3],num_class=6):
  # construct model with CBAM
  densenet201 = DenseNet201(include_top=False,
                         input_shape=input_shape,
                         weights=None,
                         pooling='avg')
  x = densenet201.output
  # add l2 norm to feature and weights
  # l2 normalization feature
  x = Lambda(lambda x:K.l2_normalize(x,axis=-1))(x)
  # add l2 normalization to weights
  x = Dense(num_class,activation=None,
            kernel_constraint=unitnorm(axis=0),
            use_bias=False,
            name='predictions')(x)
  model = Model(inputs=densenet201.input,outputs=x)
  return model


def build_inception_model(input_shape=[224,224,3],num_class=6):
  # construct model with CBAM
  inceptionV3 = InceptionV3(include_top=False,
                            input_shape=input_shape,
                            weights=None,
                            pooling='avg')
  x = inceptionV3.output
  # add l2 norm to feature and weights
  # l2 normalization feature
  x = Lambda(lambda x:K.l2_normalize(x,axis=-1))(x)
  # add l2 normalization to weights
  x = Dense(num_class,
            activation=None,
            kernel_constraint=unitnorm(axis=0),
            use_bias=False,
            name='predictions')(x)
  model = Model(inputs=inceptionV3.input,outputs=x)
  return model

def build_xception_model(input_shape=[224,224,3],num_class=6):
  # construct model with CBAM
  xception = Xception(include_top=False,
                      input_shape=input_shape,
                      weights=None,
                      pooling='avg')
  x = xception.output
  # add l2 norm to feature and weights
  # l2 normalization feature
  x = Lambda(lambda x:K.l2_normalize(x,axis=-1))(x)
  # add l2 normalization to weights
  x = Dense(num_class,
            activation=None,
            kernel_constraint=unitnorm(axis=0),
            use_bias=False,
            name='predictions')(x)
  model = Model(inputs=xception.input,outputs=x)
  return model

if __name__ == '__main__':
  model = build_resnet_model()
  print(model.output.shape)
