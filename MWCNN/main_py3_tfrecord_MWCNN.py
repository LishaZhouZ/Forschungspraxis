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
    use_gpu = True
    tf.config.experimental_run_functions_eagerly(True)
    if use_gpu:
        print("GPU\n") 
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpu_devices))
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        train_dataset = read_and_decode('./patches/MWCNN_train_data.tfrecords')
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = read_and_decode('./patches/MWCNN_validation_data.tfrecords')
        val_dataset = val_dataset.batch(batch_size)
        
        train_proces = train_MWCNN(batch_size, patch_size)
        train_proces.train_and_checkpoint(train_dataset, epochs, val_dataset)
        
        if not os.path.exists("saved_model"):
            os.mkdir("saved_model")

        train_proces.model.save('./saved_model/model.h5')
        train_proces.model.save('./saved_model/model', save_format='tf')
        print("model have been saved")
