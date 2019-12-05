import time
from utils_py3_tfrecord_2 import *
from seq_model_MWCNN import *
import numpy as np
import matplotlib.pyplot as plt

#class MeanSquaredError(Loss):
#  def call(self, y_true, y_pred):
#    y_pred = ops.convert_to_tensor(y_pred)
#    y_true = math_ops.cast(y_true, y_pred.dtype)
#   return K.mean(math_ops.square(y_pred - y_true), axis=-1)


class train_MWCNN(object):
    def __init__(self, batch_size, patch_size, learning_rate, optimizer='Adam', name='MWCNN'):
        super(train_MWCNN, self).__init__()
        self.__batch_size = batch_size
        self.__patch_size = patch_size
        self.__train_summary_writer = tf.summary.create_file_writer(
            './logs/train')
        self.__test_summary_writer = tf.summary.create_file_writer(
            './logs/test')
        self.__learning_rate = learning_rate
        self.model = build_model(64)
        if optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(
                self.__learning_rate, name='AdamOptimizer')
        #SGD + momentum
        elif optimizer == 'SGD':
            #optimizer = tf.keras.optimizers.SGD(self.lr, momentum=0.9, decay=0.0001)
            #optimizer= tf.train.GradientDescentOptimizer(self.lr, name='GradientDescent')
            self.optimizer = tf.keras.optimizers.MomentumOptimizer(
                self.__learning_rate, momentum=0.9)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def loss_fn(self, prediction, groundtruth):
        #inv_converted = wavelet_inverse_conversion(prediction)
        lossRGB = (1.0 /self.__batch_size / self.__patch_size / self.__patch_size) * tf.nn.l2_loss(prediction - groundtruth)
        #regularization loss
        reg_losses = self.model.losses
        total_loss = tf.add_n([lossRGB] + reg_losses)
        return total_loss
        
    #for one batch
    @tf.function
    def train_step(self, images, labels):
        #converted = wavelet_conversion(images)
        with tf.GradientTape() as tape:
            reconstructed = self.model(images, training = True)
            #plt.imshow(images[1,:,:,:]/255)
            #plt.show()
            #plt.imshow(reconstructed[1,:,:,:].numpy()/255)
            #plt.show()
            # Compute reconstruction loss
            total_loss = self.loss_fn(reconstructed, labels)
            tape.watch(self.model.trainable_variables)
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        
        # Training loop
        # Optimize the model
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    train_dataset = read_and_decode(
        './patches/MWCNN_train_data_debug.tfrecords', 2, 192)
    train_proces = train_MWCNN(2, 192, learning_rate=0.01)
    epochs = 1
    
    # Iterate over epochs.
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        # Iterate over the batches of the dataset.
        steps = 1
        for labels, images in train_dataset:
            train_proces.train_step(images,labels)
    print(train_proces.model.summary())