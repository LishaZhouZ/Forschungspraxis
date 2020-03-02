import argparse
import glob
from PIL import Image
import PIL
import random
import tensorflow as tf
import time
from pathlib import Path
from utils_py3_tfrecord_2 import *
from config import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--stride', dest='stride', type=int, default=25, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step, or padding')
parser.add_argument('--augment', dest='DATA_AUG_TIMES', type=int, default=1, help='data augmentation, used to creat more data')
# check output arguments
args = parser.parse_args()

def generate_patches(dir_label, dir_input, save_dir, isDebug, tfRecord_name):
    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))

    if isDebug:
        numDebug = 50
        filepaths_label = filepaths_label[:numDebug] # take only ten images to quickly debug
        filepaths_input = filepaths_input[:numDebug]
    print("number of training images %d" % len(filepaths_label))
    count = 0 # calculate the number of patches
    for i in range(len(filepaths_label)):
        img = Image.open(filepaths_label[i])
        im_h, im_w = img.size
        for x in range(0 + args.step, (im_h - patch_size), args.stride):
            for y in range(0 + args.step, (im_w - patch_size), args.stride):
                count += 1
    origin_patch_num = count * (args.DATA_AUG_TIMES)
    if origin_patch_num % batch_size != 0:
        numPatches = int(origin_patch_num / batch_size) * batch_size
    else:
        numPatches = int(origin_patch_num)
    print("Total patches = %d , batch size = %d, total batches = %d" %(numPatches, batch_size, numPatches / batch_size))
    time.sleep(2)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    count = 0

    # generate patches
    writer = tf.io.TFRecordWriter(save_dir + '/' + tfRecord_name)
    for i in range(len(filepaths_label)):
        img = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
        
        img_s = np.array(img, dtype="uint8")
        img_s_input = np.array(img_input, dtype="uint8")
        im_h, im_w, im_c = img_s.shape
        print("The %dth image of %d training images" %(i+1, len(filepaths_label)))
        for j in range(args.DATA_AUG_TIMES):
            for x in range(0 + args.step, im_h - patch_size, args.stride):
                for y in range(0 + args.step, im_w - patch_size, args.stride):
                    if j == 0:
                        random_seed = 0
                    else:
                        random_seed = random.randint(1, 7)
                    image_label = data_augmentation(img_s[x:x + patch_size, y:y + patch_size, 0:3], random_seed) # some images have an extra blank channel 
                    image_bayer = data_augmentation(img_s_input[x:x + patch_size, y:y + patch_size, 0:3], random_seed)
                    
                    image_label = image_label.tobytes()
                    image_bayer = image_bayer.tobytes()
                    count += 1
                    example = tf.train.Example(features = tf.train.Features(feature={
                        "img_label":tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_label])),
                        'img_bayer':tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_bayer]))
                    }))
                    if count<= numPatches:
                        writer.write(example.SerializeToString())
                    else:
                        break
    writer.close()
    print("Total patches = %d , batch size = %d, total batches = %d" %(numPatches, batch_size, numPatches / batch_size))
    print("Data has been written into TFrecord.")

if __name__ == '__main__': 
    src_dir_label = Path("./images/train/groundtruth")
    src_dir_input = Path("./images/train/CompressedQ10")
    save_dir = './patches'
    tfRecord_name = 'MWCNN_train_data.tfrecords'
    print("Training data will be generated:")
    generate_patches(src_dir_label, src_dir_input, save_dir, debug_mode, tfRecord_name)

    #For validation data
    val_dir_label = Path("./images/train/validation/groundtruth")
    val_dir_input = Path("./images/train/validation/CompressedQ10")
    tfRecord_val_name = 'MWCNN_validation_data.tfrecords'
    print("Validation data will be generated:")
    generate_patches(val_dir_label, val_dir_input, save_dir, debug_mode, tfRecord_val_name)


