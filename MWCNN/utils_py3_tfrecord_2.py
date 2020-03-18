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
    img_input_de = tf.io.decode_raw(features['img_bayer'], tf.uint8)
    img_input = tf.reshape(img_input_de, [patch_size, patch_size, 3])
    img_input = tf.cast(img_input, tf.float32)

    img_label_de = tf.io.decode_raw(features['img_label'], tf.uint8)
    img_label = tf.reshape(img_label_de, [patch_size, patch_size, 3])
    img_label = tf.cast(img_label, tf.float32)

    return img_label, img_input


def read_and_decode(filename):
    #arg_patch_size = tf.constant(patch_size, dtype=tf.int64)
    # read from file path
    raw_image_dataset = tf.data.TFRecordDataset(filename)
    # extract the data from raw image
    extracted_dataset = raw_image_dataset.map(extract_fn)
    # shuffle
    #if batch_size == None:
     #   return extracted_dataset
    #else:
    #    extracted_dataset = extracted_dataset.shuffle(1000 + 3 * batch_size)
    #    extracted_dataset = extracted_dataset.batch(batch_size)
    return extracted_dataset


def load_images(filelist):
    if not isinstance(filelist, list):
        im = Image.open(filelist)
        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    data = []
    for file in filelist:
        im = Image.open(file)
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 3))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate(
            [ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath)


def cal_psnr(im1, im2):
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    psnr = tf.image.psnr(im1, im2, 255)
    return tf.math.reduce_mean(psnr)


def imcpsnr(im1, im2, peak=255, b=0):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    im1 = im1[b:im1.shape[0]-b, b:im1.shape[1]-b, :]
    im2 = im2[b:im2.shape[0]-b, b:im2.shape[1]-b, :]
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2)
    mse = mse.mean()
    return 10 * np.log10(255 ** 2 / mse)


def impsnr(im1, im2, peak=255, b=0):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    im1 = im1[b:im1.shape[0]-b, b:im1.shape[1]-b, :]
    im2 = im2[b:im2.shape[0]-b, b:im2.shape[1]-b, :]
    csnr = np.ones(im1.shape[2])
    for i in range(im1.shape[2]):
        mse = ((im1[:, :, i].astype(np.float) -
                im2[:, :, i].astype(np.float)) ** 2).mean()
        csnr[i] = 10 * np.log10(255 ** 2 / mse)
    return csnr


def getBayer(RGB_Image):
    RGB_Image = np.squeeze(RGB_Image)
    h, w, _ = RGB_Image.shape
    Bayer = np.zeros([h, w], dtype=np.uint8)
    Bayer[0:h:2, 0:w:2] = RGB_Image[0:h:2, 0:w:2, 0]  # R
    Bayer[1:h:2, 0:w:2] = RGB_Image[1:h:2, 0:w:2, 1]  # G1
    Bayer[0:h:2, 1:w:2] = RGB_Image[0:h:2, 1:w:2, 1]  # G2
    Bayer[1:h:2, 1:w:2] = RGB_Image[1:h:2, 1:w:2, 2]  # B
    return Bayer
