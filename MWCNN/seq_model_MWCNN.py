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
    im_c1 = inputs[:, 0::2, 0::2, :] # 1
    im_c2 = inputs[:, 0::2, 1::2, :] # right up
    im_c3 = inputs[:, 1::2, 0::2, :] # left down
    im_c4 = inputs[:, 1::2, 1::2, :] # right right

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
    bb = LL - LH + HL - HH
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
    psnr1 = tf.reduce_mean( tf.image.psnr(y_true, y_pred, max_val=255))
    self.psnr.assign_add(psnr1)
    self.count.assign_add(1)

  def result(self):
    return self.psnr/self.count

class MS_SSIMMetric(tf.keras.metrics.Metric):
  def __init__(self, name='psnr', **kwargs):
    super(MS_SSIMMetric, self).__init__(name=name, **kwargs)
    self.ms_ssim = self.add_weight(name='ms_ssim', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')
  
  def update_state(self, y_true, y_pred):
    mssim = tf.reduce_mean( tf.image.ssim_multiscale(y_pred, y_true, 255))
    self.ms_ssim.assign_add(mssim) #output is 4x1 array
    self.count.assign_add(1)

  def result(self):
    return self.ms_ssim/self.count


def loss_fn(model, prediction, groundtruth):
  #inv_converted = wavelet_inverse_conversion(prediction)
  lossRGB = (1.0 /batch_size / patch_size / patch_size) * tf.nn.l2_loss(prediction - groundtruth)
  #regularization loss
  reg_losses = tf.math.add_n(model.losses)
  total_loss = lossRGB + reg_losses
  return total_loss

def build_MWCNN():
  my_initial = tf.initializers.he_normal()
  my_regular = tf.keras.regularizers.l2(l=0.0001)
  kernel_size = (3,3)
  model = tf.keras.Sequential()
  # 1.
  model.add(WaveletConvLayer())
  
  for i in range(3): 
    model.add(layers.Conv2D(160, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  #2.
  model.add(WaveletConvLayer())
  for i in range(4): 
    model.add(layers.Conv2D(256, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  #3.
  model.add(WaveletConvLayer())
  for i in range(7): 
    model.add(layers.Conv2D(256, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  #3
  model.add(WaveletInvLayer())
  for i in range(4): 
    model.add(layers.Conv2D(256, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  #2
  model.add(WaveletInvLayer())
  for i in range(3): 
    model.add(layers.Conv2D(160, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  model.add(layers.Conv2D(12, kernel_size, padding = 'SAME',
              kernel_initializer=my_initial, kernel_regularizer=my_regular))
  model.add(WaveletInvLayer())
  model.build((None, patch_size, patch_size, channel))
  return model