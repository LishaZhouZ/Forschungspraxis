import time
from utils_py3_tfrecord_2 import *
from seq_model_MWCNN import *
import numpy as np
import matplotlib.pyplot as plt
from config import *
import datetime

#for gradient computation
@tf.function
def grad(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        output = model(images, training = True)
        reconstructed = images + output
        total_loss = loss_fn(model, reconstructed, labels)
    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return total_loss, reconstructed


def train_one_epoch(model, dataset, optimizer, writer, ckpt):
    org_psnr = PSNRMetric()
    opt_psnr = PSNRMetric()
    avg_loss = tf.keras.metrics.Mean()
    for images, labels in dataset:
        total_loss, reconstructed = grad(model, images, labels, optimizer)
        org_psnr(images, labels)
        opt_psnr(reconstructed, labels)
        avg_loss(total_loss)
        step = ckpt.step.numpy()
        if int(step) % record_step == 0:
            avg_relative_psnr = opt_psnr.result() - org_psnr.result()
            print("Step " + str(step) + " loss {:1.2f},".format(avg_loss.result()) 
                                                                + " train_psnr {:1.5f},".format(opt_psnr.result())
                                                                + " org_psnr {:1.5f},".format(org_psnr.result())
                                                                + " gain {:1.5f}".format(avg_relative_psnr))
            #for record
            with writer.as_default():
                tf.summary.scalar('optimizer_lr_t', optimizer.learning_rate, step = step)
                tf.summary.scalar('train_loss', total_loss, step = step)
                tf.summary.scalar('train_psnr', opt_psnr.result(), step = step)
                tf.summary.scalar('original_psnr', org_psnr.result(), step = step)
                tf.summary.scalar('relative_tr_psnr', avg_relative_psnr, step = step)
            
            org_psnr.reset_states()
            opt_psnr.reset_states()
            avg_loss.reset_states()

        ckpt.step.assign_add(1)


def evaluate_model(model, val_dataset, writer, epoch):
    psnr = PSNRMetric()
    epoch_loss = tf.keras.metrics.Mean()
    ms_ssim = MS_SSIMMetric()
    org_psnr = PSNRMetric()
    
    for images_val, label_val in val_dataset:
        output = model(images_val)
        predict_val = images_val + output 
        # Update val metrics
        loss = loss_fn(model, predict_val, label_val)
        #record the things
        org_psnr.update_state(label_val, images_val)
        psnr.update_state(label_val, predict_val)
        epoch_loss.update_state(loss)
        ms_ssim.update_state(label_val, predict_val)
    
    val_psnr = psnr.result()
    val_loss = epoch_loss.result()
    ms_ssim = ms_ssim.result()
    org_psnr = org_psnr.result()
    gain = val_psnr - org_psnr

    print("Epoch " + str(epoch) + " val_loss {:1.2f},".format(val_loss) 
                            + " val_psnr {:1.5f},".format(val_psnr)
                            + " org_psnr {:1.5f},".format(org_psnr)
                            + " gain {:1.5f}".format(gain)
                            + " msssim {:1.5f}".format(ms_ssim))
    with writer.as_default():
        tf.summary.scalar('relative_val_psnr', gain, step=epoch)
        tf.summary.scalar('validation_loss', val_loss, step=epoch)
        tf.summary.scalar('validation_psnr', val_psnr, step=epoch)
        tf.summary.scalar('validation_msssim', ms_ssim, step=epoch)
        
 
    


