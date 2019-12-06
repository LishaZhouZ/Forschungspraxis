import argparse
from glob import glob

import tensorflow as tf
import math

from train_MWCNN import *
from utils_py3_tfrecord_2 import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--Q', dest='quantization_step', default='10', help='quantization step ')
parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='optimizer')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=math.pow(10,-3), help='initial learning rate for adam')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=50, help='patch size')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu', dest='num_gpu', type=str, default="0", help='choose which gpu')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
#parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
#parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='test', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='test', help='dataset for testing')
args = parser.parse_args()

#weigth decay momentum optimizer
#L2 regularization
#tensorboard

def train(train_MWCNN, lr, eval_every_step, patch_size):
    train_dataset = read_and_decode('./patches/MWCNN_train_data_debug.tfrecords', args.batch_size, args.patch_size)

    eval_files_gt = sorted(glob('../images/{}/groundtruth/*'.format(args.eval_set)))
    eval_data_gt = load_images(eval_files_gt)
    eval_files_bl = sorted(glob(('../images/{}/compressed_Q' + args.quantization_step +'/*').format(args.eval_set)))
    eval_data_bl = load_images(eval_files_bl)
    #denoiser.train(train_dataset, eval_data_gt, eval_data_bl, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, lr=lr, sample_dir=args.sample_dir, eval_every_step=eval_every_step)

    tf.config.experimental_run_functions_eagerly(True)
    train_proces.train_and_checkpoint(train_dataset, args.epochs, val_dataset)

#def test(train_MWCNN):
#    test_files_gt =  sorted(glob('../images/{}/groundtruth/*'.format(args.test_set)))
#    test_files_bl =  sorted(glob(('../images/{}/compressed_Q' + args.quantization_step +'/*').format(args.test_set)))
#    denoiser.test(test_files_gt, test_files_bl, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

#def ensemble_test(train_MWCNN):
#    test_files_gt = glob('../test/{}/groundtruth/*'.format(args.test_set))
#    test_files_bl = glob('../test/{}/compressed/*'.format(args.test_set))
#    denoiser.self_ensemble_test(test_files_gt, test_files_bl, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

if __name__ == '__main__':
    # check the path of checkpoint, samples and test
#    if not os.path.exists(args.ckpt_dir):
#        os.makedirs(args.ckpt_dir)
#    if not os.path.exists(args.sample_dir):
#        os.makedirs(args.sample_dir)
#    if not os.path.exists(args.test_dir):
#        os.makedirs(args.test_dir)
    if args.use_gpu:
        print("GPU\n") 
        os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpu_devices))
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)        

        model = train_MWCNN(args.batch_size, args.patch_size, learning_rate=0.01)
            if args.phase == 'train':
                numPatches = 0
                #TODO add control patch size
                data_set = read_and_decode('./patches/MWCNN_train_data_debug.tfrecords', None)
                numPatches = sum(1 for record in data_set)
                
                # learning rate strategy
                iter_epoch = numPatches//args.batch_size
                iter_all = args.epoch*iter_epoch
                lr = args.lr*np.ones(iter_all) 
                if args.optimizer=='Adam':
                   lr[iter_epoch*10:] = lr[0] / 2.0
                   lr[iter_epoch*20:] = lr[0] / 10.0
                   lr[iter_epoch*30:] = lr[0] / 20.0
                   lr[iter_epoch*40:] = lr[0] / 100.0

                elif args.optimizer=='SGD':
                    #decay exponentially from 1e-1 to 1e-4 for the 50 epochs.
                    for epoch in range (1, args.epoch):
                        lr[iter_epoch*epoch:] = lr[0] * math.pow(10.0,-(3.0/49.0)*epoch)

            model.train(model, lr=lr, eval_every_step=iter_epoch, patch_size=args.patch_size)
            #elif args.phase == 'test':
            #    denoiser_test(model)
            #elif args.phase == 'ensemble':
            #    ensemble_test(model)
            #else:
            #    print('[!]Unknown phase')
            #    exit(0)
    else:
        print("CPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        model = denoiser(sess, batch_size = args.batch_size, patch_size=args.patch_size)
        if args.phase == 'train':
            denoiser_train(model, lr=lr, eval_every_step=iter_epoch, patch_size=args.patch_size)
        elif args.phase == 'test':
            denoiser_test(model)
        elif args.phase == 'ensemble':
            ensemble_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)
