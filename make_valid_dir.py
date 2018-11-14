#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:46:19 2018

@author: wsw
"""

# make valid dataset
import os
import numpy as np
import skimage.io as io
import glob

validDir = './validDir'
if not os.path.exists(validDir):
  os.mkdir(validDir)
  
def make_valid_dir():
  trainDir = './trainDir'
  trainLabels = os.listdir(trainDir)
  for label in trainLabels:
    current_dir = os.path.join(validDir,label)
    if not os.path.exists(current_dir):
      os.makedirs(current_dir)
    paths = glob.glob(os.path.join(trainDir,label,'*.jpg'))
    # random choice 18% samples
    nums = int(len(paths)*0.18)
    valid_paths = np.random.choice(paths,size=nums,replace=False)
    for i,path in enumerate(valid_paths):
      image = io.imread(path)
      crop_img = random_crop(image)
      io.imsave(os.path.join(current_dir,'%d.jpg'%(i+1)),crop_img)
    print('Finish %s'%label)

    

    
def random_crop(image,height=224,width=224):
  coord = np.random.randint(0,256-224,2)
  cropped_img = image[coord[0]:coord[0]+height,coord[1]:coord[1]+width,:]
  return cropped_img


if __name__ == '__main__':
  make_valid_dir()