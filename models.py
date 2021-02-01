################################################ Imports #################################################################
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
import tensorflow_probability as tfp

######################################## Helper Functions ##################################################################

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def normalize(x,rgb_mean=DIV2K_RGB_MEAN):
    '''This function will normalize image by substracting RGB mean from image'''
    return (x - rgb_mean) / 127.5

def denormalize(x,rgb_mean=DIV2K_RGB_MEAN):
    ''' This function will denormalize image by adding back rgb_mean'''
    return (x * 127.5 )+ rgb_mean

def shuffle_pixels(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

################################################# EDSR Architecture #################################################################
def ResBlock(x_input, num_filters, scaling):
    '''Thpis function Implementes Proposed ResBlock Architecture as per EDSR paper'''
    # proposed ResBlock ==> Conv --> Relu --> Conv --> Scaling(mul) --> Add
    x = Conv2D(num_filters, 3, padding='same', activation='relu')(x_input)
    x = Conv2D(num_filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_input, x])
    return x

def Upsampling(x, scale, num_filters):
    '''This function upsampling as mentioned in EDSR paper'''
    def upsample(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(shuffle_pixels(scale=factor))(x)

    if scale == 2:
        x = upsample(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample(x, 2, name='conv2d_1_scale_2')
        x = upsample(x, 2, name='conv2d_2_scale_2')

    return x


def EDSR(scale, num_filters=64, res_blocks=8, res_block_scaling=None):
    x_input = Input(shape=(None, None, 3))
    # Normalize input with DIV2K_RGB_MEAN
    x = Lambda(normalize)(x_input)
    
    # assign value of x to x_res block for further operations
    x = x_res_block = Conv2D(num_filters, 3, padding='same')(x)

    # Goes in number of res block
    for i in range(res_blocks):
        x_res_block = ResBlock(x_res_block, num_filters, res_block_scaling)
    # convolution
    x_res_block = Conv2D(num_filters, 3, padding='same')(x_res_block)
    # add res_block output and original normalizwd input
    x = Add()([x, x_res_block])

    # upsampling
    x = Upsampling(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
    
    # denormalize to get back original form
    x = Lambda(denormalize)(x)
    return Model(x_input, x, name="EDSR")



##################################################### WDSR Architecture ###################################################################

# Weight normalization as considered in WDSR paper
def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfp.layers.weight_norm.WeightNorm(Conv2D(filters, kernel_size, padding=padding,
                                                 activation=activation, **kwargs), 
                                             data_init=False)

def WDSR(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):

    '''scale ==> It use at the end of residual blocks for slim mapping and also use in 
    original input to make size same as residual block output and then add it'''

    '''num_filters ==> Number of filters in conv of residual block '''

    '''res_block_expansion ==> a slim identity mapping pathway with wider (2× to 4×) channels 
    before activation in each residual block.'''

    '''res_block_scaling ==> whether you can to scale residual block or not'''

    '''res_block ==> use res_block_a for WDSR-A and res_block_b for WDSR-B network'''

    x_input = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_input)

    # main branch
    main_branch = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        main_branch = res_block(main_branch, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    main_branch = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(main_branch)
    main_branch = Lambda(shuffle_pixels(scale))(main_branch)

    # skip branch
    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = Lambda(shuffle_pixels(scale))(s)

    x = Add()([main_branch, s])
    x = Lambda(denormalize)(x)

    return Model(x_input, x, name="WDSR")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    ''' This block is used in WDSR-A network'''
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    '''This block is used for WDSR-B network'''
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def WDSR_A(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    ''' This function creates WDSR_A Architecture'''
    return WDSR(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def WDSR_B(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None):
    ''' This function creates WDSR_B Architecture'''
    return WDSR(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b)
