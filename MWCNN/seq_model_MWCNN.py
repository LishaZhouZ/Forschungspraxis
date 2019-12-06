import tensorflow as tf
import pywt
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from utils_py3_tfrecord_2 import *
# network structure

class WaveletConvLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(WaveletConvLayer, self).__init__()

  def call(self, inputs):
    inputs = inputs/2
    im_c1 = inputs[:, 0::2, 0::2, :]
    im_c2 = inputs[:, 0::2, 1::2, :]
    im_c3 = inputs[:, 1::2, 0::2, :]
    im_c4 = inputs[:, 1::2, 1::2, :]

    LL = im_c1 + im_c2 + im_c3 + im_c4
    LH = -im_c1 - im_c2 + im_c3 + im_c4
    HL = -im_c1 + im_c2 - im_c3 + im_c4
    HH = im_c1 - im_c2 - im_c3 + im_c4
    result = tf.concat([LL, LH, HL, HH], 3) #(None, 96,96,12)    
    return result

class WaveletInvLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(WaveletInvLayer, self).__init__()

  def call(self, inputs):
    sz = inputs.shape
    inputs = inputs/2
    a = tf.cast(sz[3]/4, tf.int32)
    LL = inputs[:, :, :, 0:a]
    LH = inputs[:, :, :, a:2*a]
    HL = inputs[:, :, :, 2*a:3*a]
    HH = inputs[:, :, :, 3*a:]
    
    aa = LL - LH - HL + HH
    bb = LL - LH + HL + HH
    cc = LL + LH - HL - HH
    dd = LL + LH + HL + HH
    concated = tf.concat([aa, bb, cc, dd], 3)
    reconstructed = tf.nn.depth_to_space(concated, 2)
    return reconstructed

#return the average psnr for
class PSNRMetric(tf.keras.metrics.Metric):

  def __init__(self, name='psnr', **kwargs):
    super(PSNRMetric, self).__init__(name=name, **kwargs)
    self.psnr = self.add_weight(name='psnr', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')
  
  def update_state(self, y_true, y_pred):
    mse = imcpsnr(y_true, y_pred)
    self.psnr.assign_add(mse)
    self.count.assign_add(1)

  def result(self):
    return self.psnr/self.count

def build_model(filters):
  my_initial = tf.initializers.he_normal()
  my_regular = tf.keras.regularizers.l2(l=0.0001)
  kernel_size = (3,3)
  model = tf.keras.Sequential()
  model.add(WaveletConvLayer())
  model.add(layers.Conv2D(filters, kernel_size, padding = 'SAME',
              kernel_initializer=my_initial, kernel_regularizer=my_regular))
  #activation?
  for i in range(2): 
      model.add(layers.Conv2D(filters, kernel_size, padding = 'SAME',
              activation=tf.nn.relu, kernel_initializer=my_initial, kernel_regularizer=my_regular))
      model.add(layers.BatchNormalization())
      model.add(layers.ReLU())
  #    #assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
  model.add(layers.Conv2D(12, kernel_size, padding = 'SAME',
              activation=tf.nn.relu, kernel_initializer=my_initial, kernel_regularizer=my_regular))
  model.add(WaveletInvLayer())
  
  return model