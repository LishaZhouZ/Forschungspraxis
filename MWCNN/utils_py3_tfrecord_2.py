import gc
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import IPython.display as display
from config import *


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def extract_fn(data_record):
    features = tf.io.parse_single_example(
        data_record, {
            'img_label': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'img_bayer': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        })
    features['img_bayer'] = tf.io.decode_raw(features['img_bayer'], tf.uint8)
    features['img_bayer'] = tf.reshape(features['img_bayer'], [patch_size, patch_size, 3])
    features['img_bayer'] = tf.cast(features['img_bayer'], tf.float32)
    features['img_bayer'] = tf.image.rgb_to_grayscale(features['img_bayer'])

    features['img_label'] = tf.io.decode_raw(features['img_label'], tf.uint8)
    features['img_label'] = tf.reshape(features['img_label'], [patch_size, patch_size, 3])
    features['img_label'] = tf.cast(features['img_label'], tf.float32)
    features['img_label'] = tf.image.rgb_to_grayscale(features['img_label'])
    return features['img_bayer'], features['img_label']


def read_and_decode(filename):
    #arg_patch_size = tf.constant(patch_size, dtype=tf.int64)
    # read from file path
    raw_image_dataset = tf.data.TFRecordDataset(filename)

    # extract the data from raw image
    #200 batch size 8 -- 101M RAM
    #100 batch sizz 8 -- 64M RAM
    raw_image_dataset = raw_image_dataset.shuffle(200)
    # order is important
    raw_image_dataset = raw_image_dataset.map(extract_fn)
    raw_image_dataset = raw_image_dataset.batch(batch_size)
    return raw_image_dataset


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # ground_truth = np.squeeze(ground_truth)
    # noisy_image = np.squeeze(noisy_image)
    # clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate(
            [ground_truth, noisy_image, clean_image], axis=1)
    im = np.around(np.clip(cat_image, 0, 255)).astype('uint8')
    im = Image.fromarray(np.squeeze(im), mode='L')
    im.save(filepath)

