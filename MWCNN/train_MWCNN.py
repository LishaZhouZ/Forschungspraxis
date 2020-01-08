import time
from utils_py3_tfrecord_2 import *
from seq_model_MWCNN import *
import numpy as np
import matplotlib.pyplot as plt

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
        self.metrics = PSNRMetric()
        if optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(
                self.__learning_rate, name='AdamOptimizer')
        #SGD + momentum
        elif optimizer == 'SGD':
            #optimizer = tf.keras.optimizers.SGD(self.lr, momentum=0.9, decay=0.0001)
            #optimizer= tf.train.GradientDescentOptimizer(self.lr, name='GradientDescent')
            self.optimizer = tf.keras.optimizers.MomentumOptimizer(
                self.__learning_rate, momentum=0.9)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)

    def loss_fn(self, prediction, groundtruth):
        #inv_converted = wavelet_inverse_conversion(prediction)
        lossRGB = (1.0 /self.__batch_size / self.__patch_size / self.__patch_size) * tf.nn.l2_loss(prediction - groundtruth)
        #regularization loss
        reg_losses = self.model.losses
        total_loss = tf.add_n([lossRGB] + reg_losses)
        return total_loss
        
    #for one batch
    @tf.function
    def train_step(self, images, labels, training):
        #converted = wavelet_conversion(images)
        with tf.GradientTape() as tape:
            reconstructed = self.model(images, training = training)
            output = images + reconstructed
            # test code
            #plt.imshow(images[1,:,:,:]/255)
            #plt.show()
            #plt.imshow(reconstructed[1,:,:,:].numpy()/255)
            #plt.show()
            #psnr = imcpsnr(images[1,:,:,:], reconstructed[1,:,:,:])
            # Compute reconstruction loss
            total_loss = self.loss_fn(output, labels)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        # Training loop
        # Optimize the model
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return total_loss

    def train_and_checkpoint(self, train_dataset, epochs, val_dataset):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for epoch in range(epochs):
            print('Start of epoch %d' % (epoch,))
            # Iterate over the batches of the dataset.
            for labels, images in train_dataset:
                ## main step
                loss = self.train_step(images, labels, training = True)
                self.ckpt.step.assign_add(1)
                if int(self.ckpt.step) % 10 == 0:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path) + "-- loss {:1.2f}".format(loss.numpy()))
        
        for label_val, images_val in val_dataset:
            predict_val = self.model(images_val, training = False)
            # Update val metrics
            self.metrics.update_state(label_val, predict_val)
        val_acc = self.metrics.result()
        self.metrics.reset_states()
        print('Validation psnr: %s' % (float(val_acc),))

if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    train_dataset = read_and_decode(
        './patches/MWCNN_train_data_debug.tfrecords', 24)
    val_dataset = read_and_decode(
        './patches/MWCNN_train_data_debug.tfrecords', 24)
    train_proces = train_MWCNN(24, 192, learning_rate=0.01)
    epochs = 1
    train_proces.train_and_checkpoint(train_dataset, epochs, val_dataset)
    


