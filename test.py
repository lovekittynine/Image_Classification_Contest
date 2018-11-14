#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:35:21 2018

@author: ws
"""

# test
import pandas as pd
import numpy as np
from CNN_model.model import build_resnet_model
#from keras.applications.resnet50 import preprocess_input
from CNN_model.model import build_xception_model
#from keras.applications.densenet import preprocess_input
#from keras.applications.inception_v3 import preprocess_input
from CNN_model.model import build_inception_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import os
import tensorflow as tf
from keras.layers import Lambda

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class_dict = {0:'CITY',1:'DESERT',2:'FARMLAND',
              3:'LAKE',4:'MOUNTAIN',5:'OCEAN'}

testDir = './数据集/宽波段数据集-预赛测试集A1000'
submitDir = './submit'
if not os.path.exists(submitDir):
    os.makedirs(submitDir)
    
# build model
base_model = build_resnet_model()
x = base_model.output
# add scale
s = 10.0
x = Lambda(lambda x:tf.multiply(s,x))(x)
model = Model(inputs=base_model.input,outputs=x)


# load parameters
ckpt = os.path.join('checkpoint2','model-43-1.000.h5')
model.load_weights(ckpt,by_name=True)

# ImageGenerator
testDataGen = ImageDataGenerator(
                                 # preprocessing_function=preprocess_input,
                                 rescale=1./255,
                                 fill_mode='wrap')
test_generator = testDataGen.flow_from_directory(testDir,
                                                 target_size=(224,224),
                                                 shuffle=False)

# predict test label
def predict_label(idx=1):
    predictions = model.predict_generator(test_generator,
                                     steps=test_generator.n//32+1,
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
    df.to_csv(os.path.join(submitDir,'submit-%d.csv'%idx),
              header=None,index=False)
    print('Generate Finished!!!')
    

if __name__ == '__main__':
    predict_label(idx=10)
    


