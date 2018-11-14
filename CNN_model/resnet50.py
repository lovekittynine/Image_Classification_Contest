from __future__ import print_function
import os
from keras import layers
from keras.layers import Activation
from keras.layers import Conv2D,Lambda
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape
## for 2.2.4
#from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from .attention_model import attach_attention_module
from keras.regularizers import l2
from keras.layers import Multiply



import tensorflow as tf
slim = tf.contrib.slim
instance_norm = slim.instance_norm


WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# weight decay 
wd = 1e-4

def identity_block(input_tensor, kernel_size, filters, stage, block,attention='cbam_block',IN=True,idx=1):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor  #输入变量#
        kernel_size: defualt 3, the kernel size of middle conv layer at main path #卷积核的大小#
        filters: list of integers, the filterss of 3 conv layer at main path  #卷积核的数目#
        stage: integer, current stage label, used for generating layer names #当前阶段的标签#
        block: 'a','b'..., current block label, used for generating layer names #当前块的标签#
    # Returns
        Output tensor for the block.  #返回块的输出变量#
    """
    filters1, filters2, filters3 = filters  #滤波器的名称#
    if K.image_data_format() == 'channels_last':  #代表图像通道维的位置#
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1) ,name=conv_name_base + '2a',
               kernel_regularizer=l2(wd))(input_tensor)
    # weather add instance normalization
    if IN:
        # split by channel
        x1,x2 = Lambda(lambda x:tf.split(x,2,axis=-1))(x)
        in1 = Lambda(lambda x:instance_norm(x))(x1)
        bn2 = BatchNormalization(axis=bn_axis,name='bn'+str(stage)+block+'2a')(x2)
        x = layers.Concatenate(axis=-1)([in1,bn2])
    else:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)   #卷积层，BN层，激活函数#

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b',
               kernel_regularizer=l2(wd))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
               kernel_regularizer=l2(wd))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    # add attention module
    if attention:
      x = attach_attention_module(x,attention,idx=idx,model='res')
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x




def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),attention='cbam_block',IN=True,idx=1):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a',
               kernel_regularizer=l2(wd))(input_tensor)
    
    # weather add instance normalization
    if IN:
        # split by channel
        x1,x2 = Lambda(lambda x:tf.split(x,2,axis=-1))(x)
        in1 = Lambda(lambda x:instance_norm(x))(x1)
        bn2 = BatchNormalization(axis=bn_axis,name='bn'+str(stage)+block+'2a')(x2)
        x = layers.Concatenate(axis=-1)([in1,bn2])
    else:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b',
               kernel_regularizer=l2(wd))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
               kernel_regularizer=l2(wd))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    # add attention module
    if attention:
      x = attach_attention_module(x,attention,idx=idx,model='res')
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             bilinear=False):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_regularizer=l2(wd),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # stage2 output 56x56x256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),idx=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',idx=2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',idx=3)
    
    # stage3 output 28x28x512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',idx=4)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',idx=5)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',idx=6)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',idx=7)
    
    # stage4 output 14x14x1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',IN=True,idx=8)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',IN=True,idx=9)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',IN=True,idx=10)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',IN=True,idx=11)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',IN=True,idx=12)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',IN=True,idx=13)
    
    # stage5 output 7x7x2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',IN=False,idx=14)
    x1 = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',IN=False,idx=15)
    x = identity_block(x1, 3, [512, 512, 2048], stage=5, block='c',IN=False,idx=16)
    
    if bilinear:
        # add bilinear layer
        bilinear = Bilinear_layer(num_outputs=2048)
        fbp1 = bilinear(x1)
        fbp2 = bilinear(x)
        # element wise multiply
        x = Multiply()([fbp1,fbp2])
        # sign square root
        x = Sign_Square_root(x)

    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs, x, name='resnet50')
    
    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH=None,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)
    
    return model


def Bilinear_layer(num_outputs=2048):
    # initialize a conv layer to project inputs feature
    bilinear = Conv2D(filters=num_outputs,
                      kernel_size=(1,1),
                      activation='relu')
    return bilinear


def Sign_Square_root(x):
    epsilion = 1e-12
    sign_square_root = Lambda(lambda x:K.sign(x)*K.sqrt(K.abs(x)+epsilion))
    return sign_square_root(x)


if __name__ == '__main__':
  model = ResNet50(include_top=False,weights=None,
                   input_shape=[224,224,3],
                   pooling='avg',
                   bilinear=True)
  x = model.output
  print(x.shape)