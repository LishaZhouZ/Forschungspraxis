import time
from utils_py3_tfrecord_2 import *
from seq_model_MWCNN import *
import numpy as np
import matplotlib.pyplot as plt
from config import *
import datetime

class train_MWCNN(object):
    def __init__(self, batch_size, patch_size, optimizer='Adam', name='MWCNN'):
        super(train_MWCNN, self).__init__()
        self.__batch_size = batch_size
        self.__patch_size = patch_size
        self.__summary_writer = tf.summary.create_file_writer(
            './logs/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model = build_MWCNN()
        if optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.01, epsilon=1e-8, name='AdamOptimizer')
        #SGD + momentum
        elif optimizer == 'SGD':
            #optimizer = tf.keras.optimizers.SGD(self.lr, momentum=0.9, decay=0.0001)
            #optimizer= tf.train.GradientDescentOptimizer(self.lr, name='GradientDescent')
            self.optimizer = tf.keras.optimizers.MomentumOptimizer(
                learning_rate=0.01, momentum=0.9)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=None)


    def loss_fn(self, prediction, groundtruth):
        #inv_converted = wavelet_inverse_conversion(prediction)
        lossRGB = (1.0 /self.__batch_size / self.__patch_size / self.__patch_size) * tf.nn.l2_loss(prediction - groundtruth)
        #regularization loss
        reg_losses = self.model.losses
        total_loss = tf.add_n([lossRGB] + reg_losses)
        return total_loss

    #for one batch
    @tf.function
    def train_step(self, images, labels, training, decay_step_size):
        #converted = wavelet_conversion(images)
        with tf.GradientTape() as tape:
            reconstructed = self.model(images, training = training)
            output = images + reconstructed
            total_loss = self.loss_fn(output, labels)
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        # Training loop
        # Optimize the model
        self.optimizer.learning_rate = decay_step_size
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        psnr1 = imcpsnr(output, labels) 
        psnr2 = imcpsnr(images, labels)
        return total_loss, psnr1, psnr2
    
    def evaluate_model(self, val_dataset):
        psnr = PSNRMetric()
        epoch_loss = tf.keras.metrics.Mean()
        ms_ssim = MS_SSIMMetric()
        org_psnr = PSNRMetric()
        for label_val, images_val in val_dataset:
            predict_val = images_val + self.model.predict(images_val)
            # Update val metrics
            loss = self.loss_fn(predict_val, label_val)

            org_psnr.update_state(label_val, images_val)
            psnr.update_state(label_val, predict_val)
            epoch_loss.update_state(loss)
            ms_ssim.update_state(label_val, predict_val)
        val_psnr = psnr.result()
        val_loss = epoch_loss.result()
        ms_ssim = ms_ssim.result()
        org_psnr = org_psnr.result()

        print('Original psnr: %s' % (float(org_psnr),))
        print('Validation psnr: %s' % (float(val_psnr),))
        print('Validation loss: %s' % (float(val_loss),))
        print('Validation msssim: %s' % (float(ms_ssim),))
        return val_psnr, val_loss, ms_ssim, org_psnr

    def train_and_checkpoint(self, train_dataset, epochs, val_dataset):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
            start_epoch = self.ckpt.save_counter.numpy()+1
            
        else:
            print("Initializing from scratch.")
            start_epoch = 1
        avg_tr_psnr = tf.keras.metrics.Mean()
        avg_org_psnr = tf.keras.metrics.Mean()
        avg_loss = tf.keras.metrics.Mean()

        for epoch in range(start_epoch, epochs+1):
            train_dataset = train_dataset.shuffle(2000)
            
            print('Start of epoch %d' % (epoch,))
            # iterate over the batches of the dataset.
            for labels, images in train_dataset:
                self.ckpt.step.assign_add(1)
                ## main step
                #decay step size is an interface parameter
                train_loss, psnr_tr, psnr_org = self.train_step(images, labels, training = True, decay_step_size = decay_lr[epoch])
                
                avg_loss.update_state(train_loss)
                avg_tr_psnr.update_state(psnr_tr)
                avg_org_psnr.update_state(psnr_org)

                # show the loss in every 1000 updates, keep record of the update times
                if int(self.ckpt.step.numpy()) % record_step == 0:
                    avg_relative_psnr = avg_tr_psnr.result() - avg_org_psnr.result()
                    print("Step " + str(self.ckpt.step.numpy()) + " loss {:1.2f},".format(avg_loss.result()) 
                                                                + " train_psnr {:1.5f},".format(avg_tr_psnr.result())
                                                                + " org_psnr {:1.5f},".format(avg_org_psnr.result())
                                                                + " gain {:1.5f}".format(avg_relative_psnr))
                    with self.__summary_writer.as_default():
                        tf.summary.scalar('train_loss', avg_loss.result(), step=self.ckpt.step.numpy())
                        tf.summary.scalar('train_psnr', avg_tr_psnr.result(), step=self.ckpt.step.numpy())
                        tf.summary.scalar('original_psnr', avg_org_psnr.result(), step=self.ckpt.step.numpy())
                        tf.summary.scalar('relative_tr_psnr', avg_relative_psnr, step=self.ckpt.step.numpy())
                    
                    avg_loss.reset_states()
                    avg_tr_psnr.reset_states()
                    avg_org_psnr.reset_states()
            
            with self.__summary_writer.as_default():
                tf.summary.scalar('optimizer_lr_t', self.optimizer.learning_rate, step=epoch)
                # use validation set to get accuarcy 
                val_psnr, val_loss, ms_ssim, org_psnr = self.evaluate_model(val_dataset)
                gain = val_psnr - org_psnr
                print('Relative psnr: %s' % (float(gain),))
                tf.summary.scalar('relative_val_psnr', gain, step=epoch)
                tf.summary.scalar('validation_psnr', val_psnr, step=epoch)
                tf.summary.scalar('validation_loss', val_loss, step=epoch)
                tf.summary.scalar('validation_msssim', ms_ssim, step=epoch)
            
            # save the checkpoint in every epoch
            save_path = self.manager.save()
            print("Saved checkpoint for epoch {}: {}".format(int(epoch), save_path))

if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    train_dataset = read_and_decode(
        './patches/MWCNN_train_data.tfrecords')
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = read_and_decode(
        './patches/MWCNN_validation_data.tfrecords')
    val_dataset = val_dataset.batch(batch_size)

    train_proces = train_MWCNN(batch_size, patch_size)
    train_proces.train_and_checkpoint(train_dataset, epochs, val_dataset)
    


