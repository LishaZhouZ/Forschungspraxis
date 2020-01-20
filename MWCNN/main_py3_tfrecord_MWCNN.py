import argparse
from glob import glob

import tensorflow as tf
import math

from train_MWCNN import *
from utils_py3_tfrecord_2 import *
from config import *

#parser = argparse.ArgumentParser(description='')
#parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
#parser.add_argument('--Q', dest='quantization_step', default='10', help='quantization step ')
#parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='optimizer')
#parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
#parser.add_argument('--lr', dest='lr', type=float, default=math.pow(10,-3), help='initial learning rate for adam')
#parser.add_argument('--patch_size', dest='patch_size', type=int, default=50, help='patch size')
#parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
#parser.add_argument('--gpu', dest='num_gpu', type=str, default="0", help='choose which gpu')
#parser.add_argument('--phase', dest='phase', default='train', help='train or test')
#parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
#parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
#parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
#parser.add_argument('--eval_set', dest='eval_set', default='test', help='dataset for eval in training')
#parser.add_argument('--test_set', dest='test_set', default='test', help='dataset for testing')
#args = parser.parse_args()

#weigth decay momentum optimizer
#L2 regularization
#tensorboard

if __name__ == '__main__':
    use_gpu = True
    tf.config.experimental_run_functions_eagerly(True)
    # check the path of checkpoint, samples and test
#    if not os.path.exists(args.ckpt_dir):
#        os.makedirs(args.ckpt_dir)
#    if not os.path.exists(args.sample_dir):
#        os.makedirs(args.sample_dir)
#    if not os.path.exists(args.test_dir):
#        os.makedirs(args.test_dir)
    if use_gpu:
        print("GPU\n") 
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpu_devices))
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        train_dataset = read_and_decode(
            './patches/MWCNN_train_data.tfrecords', batch_size)
        val_dataset = read_and_decode(
            './patches/MWCNN_validation_data.tfrecords', batch_size)
        
        train_proces = train_MWCNN(batch_size, patch_size, learning_rate)
        train_proces.train_and_checkpoint(train_dataset, epochs, val_dataset)

        model = train_MWCNN(batch_size, patch_size, learning_rate)
        model.train_and_checkpoint(train_dataset, epochs, val_dataset)