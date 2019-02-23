# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:07:51 2019

@author: huyz
"""

''' TensorFlow基本操作 '''

import tensorflow as tf

#weight_initializer = tf.contrib.layers.xavier_initializer()
weight_initializer = tf.truncated_normal_initializer(stddev=0.02)
#weight_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
weight_regularizer = None

def conv2d(x, output_channels, kernel_size=5, strides=2, padding='same', pad=0, pad_type='zero', scope='conv'):
    with tf.variable_scope(scope):
        
        '''
        tf.pad(tensor, paddings, mode='CONSTANT')
        mode由三种模式: CONSTANT, REFLECT, SYMMETRIC
            - CONSTANT 为默认填充方式, 填充的是0
            - REFLECT 为映射填充，上下填充顺序和paddings相反，左右顺序补齐
            - SYMMETRIC 为堆成填充，上下填充顺序和paddings相同，左右对称补齐
        '''
        
        if pad == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        
        x = tf.layers.conv2d(inputs=x,
                            filters=output_channels,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None,
                            kernel_initializer=weight_initializer,
                            kernel_regularizer=weight_regularizer)
        
        return x
'''
def deconv2d(x, output_channels, kernel_size=5, strides=2, padding='same', scope='deconv'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x,
                                       filters=output_channels,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding=padding,
                                       activation=None,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02)),
                                       kernel_regularizer=weight_regularizer)
       
        return x
'''
def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def max_pooling(x, pool_size=2, strides=2, padding='same', scope='max_pooling'):
    with tf.variable_scope(scope):
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=pool_size,
                                    strides=strides,
                                    padding=padding)
        
        return x
def avg_pooling(x, pool_size=2, strides=2, padding='same', scope='avg_pooling'):
    with tf.variable_scope(scope):
        x = tf.layers.average_pooling2d(inputs=x,
                                        pool_size=pool_size,
                                        strides=strides,
                                        padding=padding)
        return x
        
def gap(x, scope='global_average_pooling'):
    ''' Global Average Pooling '''
    with tf.variable_scope(scope):
        x = tf.reduce_mean(x, axis=[1, 2])
        
        return x
        
def fully_connected(x, output_units, scope='fc'):
    with tf.variable_scope(scope):
        x = tf.layers.dense(inputs=x,
                            units=output_units,
                            activation=None,
                            kernel_initializer=weight_initializer,
                            kernel_regularizer=weight_regularizer)
        
        return x
        
def flatten(x, scope='flatten'):
   with tf.variable_scope(scope):
    x = tf.layers.flatten(inputs=x)
    
    return x
    
def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)
    
def relu(x):
    return tf.nn.relu(x)
    
def tanh(x):
    return tf.tanh(x)
    
'''
def batch_norm(x, is_training=True, scope='batch_norm'):
    x = tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5,
                                    center=True, scale=True,
                                    is_training=is_training,
                                    scope=scope)
    return x
'''

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)
    
def instance_norm(x, scope='instance_norm'):
    x = tf.contrib.layers.instance_norm(x, center=True, scale=True, epsilon=1e-05, scope=scope)
    return x
    
def layer_norm(x, scope='layer_norm'):
    x = tf.contrib.layers.layer_norm(x, center=True, scale=True, scope=scope)
    
def switch_norm(x, scope='switch_norm'):
    # Reference: https://github.com/taki0112/Switchable_Normalization-Tensorflow
    with tf.variable_scope(scope) :
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
        ins_mean, ins_var = tf.nn.moments(x, [1, 2], keep_dims=True)
        layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)

        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        mean_weight = tf.nn.softmax(tf.get_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0)))
        var_wegiht = tf.nn.softmax(tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0)))

        mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
        var = var_wegiht[0] * batch_var + var_wegiht[1] * ins_var + var_wegiht[2] * layer_var

        x = (x - mean) / (tf.sqrt(var + eps))
        x = x * gamma + beta

        return x

def discriminator_loss(real, fake):
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    d_loss = d_loss_real + d_loss_fake

    return d_loss

def generator_loss(fake):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    return loss

def l1_loss(x, y):
    l1_loss = tf.reduce_mean(tf.abs(x - y))

    return l1_loss

def l2_loss(x, y):
    l2_loss = tf.reduce_mean(tf.square(x - y))

    return l2_loss