import argparse
from glob import glob
import iostream as os
import math

from model_MWCNN import MWCNNBlock
from utils_py3_tfrecord import *

# Tensorflow 2.0

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--Q', dest='quantization_step', default='10', help='quantization step ')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=math.pow(10,-3), help='initial learning rate for adam')
#parser.add_argument('--lr2', dest='lr2', type=float, default=5*math.pow(10,-5), help='initial learning rate for adam')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=24, help='patch size')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=0, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu', dest='num_gpu', type=str, default="0", help='choose which gpu')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='test', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='test', help='dataset for testing')
args = parser.parse_args()

#weigth decay momentum optimizer
#L2 regularization
#tensorboard

if __name__ == '__main__':
# check the path of checkpoint, samples and test
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if not args.use_gpu:
        print("CPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with tf.Session() as sess:
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
