from seq_model_MWCNN import *
import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np
import math

import timeit



if __name__ == "__main__":
    dir_label = Path('/home/lisha/Forschungspraxis/images/test/test/groundtruth_5')
    dir_input = Path('/home/lisha/Forschungspraxis/images/test/test/compressed_Q10_5')

    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))

    count = 0
    model = MWCNN()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
    ckpt.restore(tf.train.latest_checkpoint('/home/lisha/Downloads/Training20200430/tf_ckpts')).expect_partial()

    org_psnr = np.zeros(len(filepaths_label))
    rec_psnr = np.zeros(len(filepaths_label))
    
    org_ssim = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))
    time = np.zeros(len(filepaths_label))
    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
         
        a = np.array(img_label, dtype="float32")
        b = np.array(img_input, dtype="float32")
        img_s_label = tf.convert_to_tensor(a)
        img_s_input = tf.convert_to_tensor(b)
        
        #padding
        shape_input = tf.shape(img_s_input).numpy()
        padding_up = math.ceil(16-shape_input[0]%16/2)
        padding_down = math.floor(16-shape_input[0]%16/2)
        padding_left = math.ceil(16-shape_input[1]%16/2)
        padding_right = math.floor(16-shape_input[1]%16/2)
        paddings = tf.constant([[padding_up, padding_down,], [padding_left, padding_right], [0, 0]])

        img_s_input_padded = tf.pad(img_s_input, paddings, "REFLECT")

        img_s_input_batch = tf.expand_dims(img_s_input_padded, axis = 0)
        img_s_label_batch = tf.expand_dims(img_s_label, axis = 0)
        
        start = timeit.default_timer()
        
        output = model.predict(img_s_input_batch)
        
        stop = timeit.default_timer()
        time[i] = stop - start
        
        reconstructed = img_s_input_batch + output

        output_cut = tf.slice(reconstructed, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])


        org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
        rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
        org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
        rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
        print('Image ' + str(i) + ' org_psnr:%.4f,' % org_psnr[i] + 'after_psnr:%.4f,' % rec_psnr[i], ' org_ssim:%.4f,' % org_ssim[i] + 'after_ssim:%.4f' % rec_ssim[i])
        
        
    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
         
        a = np.array(img_label, dtype="float32")
        b = np.array(img_input, dtype="float32")
        img_s_label = tf.convert_to_tensor(a)
        img_s_input = tf.convert_to_tensor(b)
        
        #padding
        shape_input = tf.shape(img_s_input).numpy()
        padding_up = math.ceil(8-shape_input[0]%8/2)
        padding_down = math.floor(8-shape_input[0]%8/2)
        padding_left = math.ceil(8-shape_input[1]%8/2)
        padding_right = math.floor(8-shape_input[1]%8/2)
        paddings = tf.constant([[padding_up, padding_down,], [padding_left, padding_right], [0, 0]])

        img_s_input_padded = tf.pad(img_s_input, paddings, "REFLECT")

        img_s_input_batch = tf.expand_dims(img_s_input_padded, axis = 0)
        img_s_label_batch = tf.expand_dims(img_s_label, axis = 0)
        
        start = timeit.default_timer()
        
        output = model.predict(img_s_input_batch)
        
        stop = timeit.default_timer()
        time[i] = stop - start
        
        reconstructed = img_s_input_batch + output

        output_cut = tf.slice(reconstructed, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])


        org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
        rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
        org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
        rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
        print('Image ' + str(i) + ' org_psnr:%.4f,' % org_psnr[i] + 'after_psnr:%.4f,' % rec_psnr[i], ' org_ssim:%.4f,' % org_ssim[i] + 'after_ssim:%.4f' % rec_ssim[i])
        
        
        #save_images(Path('/home/lisha/Forschungspraxis/images/test/outcome/' + str(i) + '.bmp'), img_s_label, noisy_image = img_s_input.numpy(), clean_image = np.squeeze(output_cut, axis=0))
    
    print('average org_psnr:%.4f' % np.mean(org_psnr))
    print('average after_psnr:%.4f' % np.mean(rec_psnr))
    print('average org_ssim:%.4f' % np.mean(org_ssim))
    print('average after_ssim:%.4f' % np.mean(rec_ssim))
    print('time %.4f' %np.sum(time))
    print('time %.4f' %np.mean(time))
