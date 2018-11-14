#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:49:53 2018

@author: sw
"""

# test multi-model

import pandas as pd
import numpy as np
from CNN_model.multi_model import build_siamese_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import os
import tensorflow as tf
from keras.layers import Lambda



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_nums = 2

class_dict = {0:'CITY',1:'DESERT',2:'FARMLAND',
              3:'LAKE',4:'MOUNTAIN',5:'OCEAN'}

batch_size = 32

testDir = './数据集/预赛测试集B'
submitDir = './submit'
if not os.path.exists(submitDir):
    os.makedirs(submitDir)
 

# build model
base_model = build_siamese_model(training=False)
x = base_model.output[3]
s = 10.0
x = Lambda(lambda x:tf.multiply(s,x))(x)
model = Model(inputs=base_model.input,outputs=x)

# load parameters
ckpt = os.path.join('checkpoint888','model- 1-1.000.h5')
model.load_weights(ckpt,by_name=True)

# ImageGenerator
testDataGen = ImageDataGenerator(
                                 # preprocessing_function=preprocess_input,
                                 rescale=1./255,
                                 fill_mode='wrap')
test_generator = testDataGen.flow_from_directory(testDir,
                                                 target_size=(224,224),
                                                 batch_size=batch_size,
                                                 shuffle=False)

# predict test label
def predict_label(idx=1):
    predictions = model.predict_generator(test_generator,
                                     steps=test_generator.n//batch_size+1,
                                     verbose=1,
                                     workers=3)
    # get test name
    img_paths = test_generator.filenames
    img_names = [os.path.split(name)[1] for name in img_paths]
    # get test labels
    labels = np.argmax(predictions,axis=-1)
    labels = [class_dict[label] for label in labels.tolist()]
    # generate submit file
    generate_submit(img_names,labels,idx)
    print('Predict Done!!!')


def generate_submit(names,labels,idx=1):
    names = np.expand_dims(np.array(names),axis=-1)
    labels = np.expand_dims(np.array(labels),axis=-1)
    res = np.concatenate([names,labels],axis=-1)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(submitDir,'submit-B-%s.csv'%idx),
              header=None,index=False)
    print('Generate Finished!!!')
    

if __name__ == '__main__':
    predict_label(idx='3_4')
