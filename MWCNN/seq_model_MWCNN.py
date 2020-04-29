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
    LH = -im_c1 + im_c2 - im_c3 + im_c4
    HL = -im_c1 - im_c2 + im_c3 + im_c4
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
    bb = LL + LH - HL - HH
    cc = LL - LH + HL - HH
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
    psnr1 = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255.0))
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
    mssim = tf.reduce_mean( tf.image.ssim_multiscale(y_pred, y_true, 255.0))
    self.ms_ssim.assign_add(mssim) #output is 4x1 array
    self.count.assign_add(1)

  def result(self):
    return self.ms_ssim/self.count


def loss_fn(prediction, groundtruth):
  #inv_converted = wavelet_inverse_conversion(prediction)
  frobenius_norm = tf.norm(prediction-groundtruth, ord='fro', axis=(1, 2))
  lossRGB = (1/2)*(tf.reduce_mean(frobenius_norm)**2)
  #regularization loss
  return lossRGB

class ConvConcatLayer(layers.Layer):
  def __init__(self, feature_num, kernel_size, my_initial, my_regular):
    super(ConvConcatLayer, self).__init__()
    self.conv = layers.Conv2D(feature_num, kernel_size, padding = 'SAME',
        kernel_initializer=my_initial,kernel_regularizer=my_regular)# 
    self.bn = layers.BatchNormalization()
    self.relu = layers.ReLU()
  
  def call(self, inputs):
    a = self.conv(inputs)
    b = self.bn(a)
    c = self.relu(b)
    return c

class ConvBlock(layers.Layer):
  def __init__(self, feature_num, kernel_size, my_initial, my_regular):
    super(ConvBlock, self).__init__()
    self.alpha1 = ConvConcatLayer(feature_num, kernel_size, my_initial, my_regular)
    self.alpha2 = ConvConcatLayer(feature_num, kernel_size, my_initial, my_regular)
    self.alpha3 = ConvConcatLayer(feature_num, kernel_size, my_initial, my_regular)
    self.alpha4 = ConvConcatLayer(feature_num, kernel_size, my_initial, my_regular)
  
  def call(self, inputs):
    a11 = self.alpha1(inputs)
    a12 = self.alpha2(a11)
    a13 = self.alpha3(a12)
    a14 = self.alpha4(a13)
    return a14

class ConvInvBlock(layers.Layer):
  def __init__(self, feature_num1, kernel_size, my_initial, my_regular):
    super(ConvInvBlock, self).__init__()
    self.alpha1 = ConvConcatLayer(feature_num1, kernel_size, my_initial, my_regular)
    self.alpha2 = ConvConcatLayer(feature_num1, kernel_size, my_initial, my_regular)
    self.alpha3 = ConvConcatLayer(feature_num1, kernel_size, my_initial, my_regular)
  
  def call(self, inputs):
    a11 = self.alpha1(inputs)
    a12 = self.alpha2(a11)
    a13 = self.alpha3(a12)
    return a13

class MWCNN(tf.keras.Model):
  def __init__(self):
    super(MWCNN, self).__init__()
    self.my_initial = tf.initializers.he_normal()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)
    
    self.convblock1 = ConvBlock(160, (3,3), self.my_initial, self.my_regular)
    self.convblock2 = ConvBlock(256, (3,3), self.my_initial, self.my_regular)
    self.convblock3 = ConvBlock(256, (3,3), self.my_initial, self.my_regular)

    self.invblock2 = ConvInvBlock(256, (3,3), self.my_initial, self.my_regular)
    self.invblock1 = ConvInvBlock(160, (3,3), self.my_initial, self.my_regular)
    
    self.wavelet1 = WaveletConvLayer()
    self.wavelet2 = WaveletConvLayer()
    self.wavelet3 = WaveletConvLayer()
    
    self.invwavelet1 = WaveletInvLayer()
    self.invwavelet2 = WaveletInvLayer()
    self.invwavelet3 = WaveletInvLayer()

    self.convlayer1024 = layers.Conv2D(1024, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
    self.convlayer640 = layers.Conv2D(640, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)
    self.convlayer12 = layers.Conv2D(48, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
  
  def call(self, inputs):

    input_arranged = tf.nn.space_to_depth(inputs, 2, data_format='NHWC', name=None)
    
    #former side
    wav1 = self.wavelet1(input_arranged)  #3-12
    con1 = self.convblock1(wav1)  #12-160
    
    #2
    wav2 = self.wavelet2(con1)   #160-640
    con2 = self.convblock2(wav2) #640-256

    #3
    wav3 = self.wavelet3(con2)   #256-1024
    con3 = self.convblock3(wav3)  #1024-256
    invcon3_expand = self.convlayer1024(con3) #256-1024

    invwav3 = self.invwavelet3(invcon3_expand)  #1024-256
    
    #2
    invcon2 = self.invblock2(invwav3 + con2) #256
    invcon2_expand = self.convlayer640(invcon2)#640
    invwav2 = self.invwavelet2(invcon2_expand) #160

    #1
    invcon1 =self.invblock1(invwav2 + con1) #160
    invcon1_retified = self.convlayer12(invcon1)#12
    invwav1 = self.invwavelet1(invcon1_retified) #3
    
    output = tf.nn.depth_to_space(invwav1, 2, data_format='NHWC', name=None)
    return output
    
class MWCNN_other(tf.keras.Model):
  def __init__(self):
    super(MWCNN, self).__init__()
    self.my_initial = tf.initializers.he_normal()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)
    
    self.convblock1 = ConvBlock(160, (3,3), self.my_initial, self.my_regular)
    self.convblock2 = ConvBlock(256, (3,3), self.my_initial, self.my_regular)
    self.convblock3 = ConvBlock(256, (3,3), self.my_initial, self.my_regular)

    self.invblock2 = ConvInvBlock(256, (3,3), self.my_initial, self.my_regular)
    self.invblock1 = ConvInvBlock(160, (3,3), self.my_initial, self.my_regular)
    
    self.wavelet1 = WaveletConvLayer()
    self.wavelet2 = WaveletConvLayer()
    self.wavelet3 = WaveletConvLayer()
    
    self.invwavelet1 = WaveletInvLayer()
    self.invwavelet2 = WaveletInvLayer()
    self.invwavelet3 = WaveletInvLayer()

    self.convlayer1024 = layers.Conv2D(1024, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
    self.convlayer640 = layers.Conv2D(640, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)
    self.convlayer12 = layers.Conv2D(48, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
  
  def call(self, inputs):

    #former side
    wav1 = self.wavelet1(inputs)  #
    con1 = self.convblock1(wav1)  #12-160
    
    #2
    wav2 = self.wavelet2(con1)   #160-640
    con2 = self.convblock2(wav2) #640-256

    #3
    wav3 = self.wavelet3(con2)   #256-1024
    con3 = self.convblock3(wav3)  #1024-256
    invcon3_expand = self.convlayer1024(con3) #256-1024

    invwav3 = self.invwavelet3(invcon3_expand)  #1024-256
    
    #2
    invcon2 = self.invblock2(invwav3 + con2) #256
    invcon2_expand = self.convlayer640(invcon2)#640
    invwav2 = self.invwavelet2(invcon2_expand) #160

    #1
    invcon1 =self.invblock1(invwav2 + con1) #160
    invcon1_retified = self.convlayer12(invcon1)#12
    invwav1 = self.invwavelet1(invcon1_retified) #3
    
    output = tf.nn.depth_to_space(invwav1, 2, data_format='NHWC', name=None)
    return output
    

def build_MWCNN():
  my_initial = tf.initializers.he_normal()
  my_regular = tf.keras.regularizers.l2(l=0.0001)
  kernel_size = (3,3)
  model = tf.keras.Sequential()
  # 1.
  model.add(WaveletConvLayer())
  
  for i in range(3): 
    model.add(layers.Conv2D(160, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial,kernel_regularizer=my_regular))# 
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  #2.
  model.add(WaveletConvLayer())
  for i in range(4): 
    model.add(layers.Conv2D(256, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))#
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  #3.
  model.add(WaveletConvLayer())
  for i in range(7): 
    model.add(layers.Conv2D(256, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))#
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  #3
  model.add(WaveletInvLayer())
  for i in range(4): 
    model.add(layers.Conv2D(256, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))#
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  #2
  model.add(WaveletInvLayer())
  for i in range(3): 
    model.add(layers.Conv2D(160, kernel_size, padding = 'SAME',
      kernel_initializer=my_initial, kernel_regularizer=my_regular))#
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
  
  model.add(layers.Conv2D(12, kernel_size, padding = 'SAME',
              kernel_initializer=my_initial ,kernel_regularizer=my_regular))#
  model.add(WaveletInvLayer())
  model.build((None, patch_size, patch_size, channel))

  return model