from memory_profiler import profile
from tensorflow import keras
import tensorflow as tf
import numpy as np
#from train_MWCNN import *
from seq_model_MWCNN import *


@tf.function
def one_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        reconstructed = model(images, training = True)
        output = images + reconstructed
        total_loss = loss_fn(model, output, labels)
    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    return total_loss


@profile
def test():
    print(tf.executing_eagerly())
    optimizer = keras.optimizers.Adam(lr=1e-3)

    model = MWCNN()

    optimizer = tf.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, name='AdamOptimizer')
    
    train_dataset = read_and_decode(
        '../patches/MWCNN_train_data.tfrecords')
    
    
    
    for epoch in range(1, 2):
        run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
        print('Start of epoch %d' % (epoch,))
        # iterate over the batches of the dataset.
        
        #self.ckpt.step.assign_add(1)
        ## main step
        #decay step size is an interface parameter
        optimizer.learning_rate = decay_lr[epoch]
        ###
        # Create your tf representation of the iterator
        for step, (images, labels) in enumerate(train_dataset.take(2)):
            total_loss = one_step(model, images, labels, optimizer)
            #converted = wavelet_conversion(images)
            

        #if int(count.numpy()) % record_step == 0:
        #    with writer.as_default():
        #        tf.summary.scalar('train_loss', total_loss, step = count.numpy())
            print("Step " + str(step) + " loss {:1.2f}".format(total_loss))

test()