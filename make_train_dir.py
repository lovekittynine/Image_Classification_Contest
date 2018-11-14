#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:29:25 2018

@author: wsw
"""

# make train dataset

import os
import pandas as pd
import shutil

dataDir = './数据集/预赛训练集-2000'
dataFile = './数据集/label.csv'

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
    shutil.move(img_path,os.path.join(current_dir,name))
    print('\rStep:%4d/Total:%4d'%(i+1,Nums),end='',flush=True)
  print('\nFinished!!!')
  
if __name__ == '__main__':
  make_train_dataset()