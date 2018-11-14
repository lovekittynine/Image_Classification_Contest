#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:36:34 2018

@author: admin3
"""

# colabrative training 
# make test datadir


import os
import pandas as pd
import shutil

dataDir = './数据集/预赛测试集B/预赛测试集B-1000'
dataFile = './submit/submit-B-2_3-0.985.csv'

trainDir = './trainDir'
if not os.path.exists(trainDir):
  os.makedirs(trainDir)
  
def make_train_dataset():
  infos = pd.read_csv(dataFile,header=None)
  Nums = len(infos)
  for i in range(Nums):
    name = infos.iloc[i,0]
    label = infos.iloc[i,1]
    # get image path
    img_path = os.path.join(dataDir,name)
    # dir for each class
    current_dir = os.path.join(trainDir,label)
    if not os.path.exists(current_dir):
      os.makedirs(current_dir)
    # move image
    shutil.copyfile(img_path,os.path.join(current_dir,name))
    print('\rStep:%4d/Total:%4d'%(i+1,Nums),end='',flush=True)
  print('\nFinished!!!')
  
if __name__ == '__main__':
  make_train_dataset()
