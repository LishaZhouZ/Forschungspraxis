import argparse
from glob import glob

import tensorflow as tf
import math
from train_MWCNN import *
from utils_py3_tfrecord_2 import *
from config import *

#weigth decay momentum optimizer
#L2 regularization
#tensorboard


if __name__ == '__main__':
    print(tf.executing_eagerly())
    physical_devices = tf.config.experimental.list_physical_devices('GPU') 
    try: 
        tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        assert tf.config.experimental.get_memory_growth(physical_devices[0]) 
    except: 
        # Invalid device or cannot modify virtual devices once initialized. 
        pass

    #read dataset
    train_dataset = read_and_decode('./patches/MWCNN_train_data.tfrecords')
    val_dataset = read_and_decode('./patches/MWCNN_validation_data.tfrecords')
    #build model
    model = build_MWCNN()
    #set up optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, name='AdamOptimizer')

    writer = tf.summary.create_file_writer('./logs/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer = optimizer, net = model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_directory, max_to_keep=None)

    #checkpoint restortion
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        start_epoch = ckpt.save_counter.numpy() + 1
    else:
        print("Initializing from scratch.")
        start_epoch = 1

    for epoch in range(start_epoch, epochs+1):
        print('Start of epoch %d' % (epoch,))
        optimizer.learning_rate = decay_lr[epoch]
        train_one_epoch(model, train_dataset, optimizer, writer, ckpt)
        evaluate_model(model, val_dataset, writer, epoch)
        # save the checkpoint in every epoch
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(int(epoch), save_path))

    print("Training saved")
