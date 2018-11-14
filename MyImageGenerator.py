#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:47:53 2018

@author: wsw
"""


# generate a batch pair samples

import numpy as np
from keras import backend
from keras.preprocessing.image import ImageDataGenerator,Iterator,load_img,array_to_img,img_to_array
from six.moves import range
import os
import multiprocessing.pool
from functools import partial
import warnings

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


class MyImageDataGenerator(ImageDataGenerator):
  def flow_from_directory(self, directory,
                          target_size=(256, 256), color_mode='rgb',
                          classes=None, class_mode='categorical',
                          batch_size=32, shuffle=True, seed=None,
                          save_to_dir=None,
                          save_prefix='',
                          save_format='png',
                          follow_links=False,
                          subset=None,
                          interpolation='nearest'):
    
    return MyDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)
    

class MyDirectoryIterator(Iterator):
  
  def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest'):
    if data_format is None:
        data_format = backend.image_data_format()
    self.directory = directory
    self.image_data_generator = image_data_generator
    self.target_size = tuple(target_size)
    if color_mode not in {'rgb', 'grayscale'}:
        raise ValueError('Invalid color mode:', color_mode,
                         '; expected "rgb" or "grayscale".')
    self.color_mode = color_mode
    self.data_format = data_format
    if self.color_mode == 'rgb':
        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = (3,) + self.target_size
    else:
        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
    self.classes = classes
    if class_mode not in {'categorical', 'binary', 'sparse',
                          'input', None}:
        raise ValueError('Invalid class_mode:', class_mode,
                         '; expected one of "categorical", '
                         '"binary", "sparse", "input"'
                         ' or None.')
    self.class_mode = class_mode
    self.save_to_dir = save_to_dir
    self.save_prefix = save_prefix
    self.save_format = save_format
    self.interpolation = interpolation

    if subset is not None:
        validation_split = self.image_data_generator._validation_split
        if subset == 'validation':
            split = (0, validation_split)
        elif subset == 'training':
            split = (validation_split, 1)
        else:
            raise ValueError('Invalid subset name: ', subset,
                             '; expected "training" or "validation"')
    else:
        split = None
    self.subset = subset

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                          'ppm', 'tif', 'tiff'}
    # First, count the number of samples and classes.
    self.samples = 0

    if not classes:
        classes = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                classes.append(subdir)
    self.num_classes = len(classes)
    self.class_indices = dict(zip(classes, range(len(classes))))

    pool = multiprocessing.pool.ThreadPool()
    function_partial = partial(_count_valid_files_in_directory,
                               white_list_formats=white_list_formats,
                               follow_links=follow_links,
                               split=split)
    self.samples = sum(pool.map(function_partial,
                                (os.path.join(directory, subdir)
                                 for subdir in classes)))

    print('Found %d images belonging to %d classes.' %
          (self.samples, self.num_classes))

    # Second, build an index of the images
    # in the different class subfolders.
    results = []
    self.filenames = []
    self.classes = np.zeros((self.samples,), dtype='int32')
    i = 0
    for dirpath in (os.path.join(directory, subdir) for subdir in classes):
        results.append(
            pool.apply_async(_list_valid_filenames_in_directory,
                             (dirpath, white_list_formats, split,
                              self.class_indices, follow_links)))
    for res in results:
        classes, filenames = res.get()
        self.classes[i:i + len(classes)] = classes
        self.filenames += filenames
        i += len(classes)

    pool.close()
    pool.join()
    super(MyDirectoryIterator, self).__init__(self.samples,
                                              batch_size,
                                              shuffle,
                                              seed)

  def _get_batches_of_transformed_samples(self, index_array):
      batch_x = np.zeros(
          (len(index_array),) + self.image_shape,
          dtype=backend.floatx())
      grayscale = self.color_mode == 'grayscale'
      # build batch of image data
      for i, j in enumerate(index_array):
          fname = self.filenames[j]
          img = load_img(os.path.join(self.directory, fname),
                         grayscale=grayscale,
                         target_size=self.target_size,
                         interpolation=self.interpolation)
          x = img_to_array(img, data_format=self.data_format)
          params = self.image_data_generator.get_random_transform(x.shape)
          x = self.image_data_generator.apply_transform(x, params)
          x = self.image_data_generator.standardize(x)
          batch_x[i] = x
      # optionally save augmented images to disk for debugging purposes
      if self.save_to_dir:
          for i, j in enumerate(index_array):
              img = array_to_img(batch_x[i], self.data_format, scale=True)
              fname = '{prefix}_{index}_{hash}.{format}'.format(
                  prefix=self.save_prefix,
                  index=j,
                  hash=np.random.randint(1e7),
                  format=self.save_format)
              img.save(os.path.join(self.save_to_dir, fname))
      # build batch of labels
      if self.class_mode == 'input':
          batch_y = batch_x.copy()
      elif self.class_mode == 'sparse':
          batch_y = self.classes[index_array]
      elif self.class_mode == 'binary':
          batch_y = self.classes[index_array].astype(backend.floatx())
      elif self.class_mode == 'categorical':
          batch_y = np.zeros(
              (len(batch_x), self.num_classes),
              dtype=backend.floatx())
          for i, label in enumerate(self.classes[index_array]):
              batch_y[i, label] = 1.
      else:
          return batch_x
      """
      add modify
      """

      # return batch_x, batch_y
      batch_y = sample_weighted(batch_y)
      return batch_x,[batch_y,batch_y,batch_y,batch_y]
      

  def next(self):
      """For python 2.x.

      # Returns
          The next batch.
      """
      with self.lock:
          index_array = next(self.index_generator)
      # The transformation of images is not under thread lock
      # so it can be done in parallel
      return self._get_batches_of_transformed_samples(index_array)


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.

    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean.

    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands '
                                  'will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname  
                    
                  
def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean.

    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)

    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames

 
def _count_valid_files_in_directory(directory,
                                    white_list_formats,
                                    split,
                                    follow_links):
    """Counts files with extension in `white_list_formats` contained in `directory`.

    # Arguments
        directory: absolute path to the directory
            containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        follow_links: boolean.

    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    num_files = len(list(
        _iter_valid_files(directory, white_list_formats, follow_links)))
    if split:
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
    else:
        start, stop = 0, num_files
    return stop - start
 
   
def pair_generator(batch_x,batch_y):
    # get a batch datas
    one_hot_labels = batch_y
    # split label1 and label2
    one_hot_label1,one_hot_label2 = np.split(one_hot_labels,2,axis=0)
    # split images
    # construct pair samples
    label1 = np.argmax(one_hot_label1,axis=-1)
    label2 = np.argmax(one_hot_label2,axis=-1)
    # set match labels and non-match labels
    Nums = len(label1)
    match_idx = np.where(label1==label2)[0]
    match_labels = np.zeros(shape=(Nums*2,2))
    match_labels[match_idx,1] = 1.0
    return batch_x,batch_y,match_labels



def sample_weighted(labels):
  weights = [3.0,1.0,2.5,1.5,1.5,1.5]
  Nums = len(labels)
  one_hot_labels = labels
  index = np.argmax(labels,axis=-1)
  for i in range(Nums):
    one_hot_labels[i,index[i]] = weights[index[i]]
  return one_hot_labels