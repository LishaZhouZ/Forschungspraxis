from seq_model_MWCNN import *
import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np

if __name__ == "__main__":
    dir_label = Path('/home/lisha/Forschungspraxis/images/test/split_label/')
    dir_input = Path('/home/lisha/Forschungspraxis/images/test/split_input/')

    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))

    count = 0
    model = build_MWCNN()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
    ckpt.restore(tf.train.latest_checkpoint('/home/lisha/Forschungspraxis/logs/Training20200413/tf_ckpts')).expect_partial()

    org_psnr = np.zeros(len(filepaths_label))
    rec_psnr = np.zeros(len(filepaths_label))
    
    org_ssim = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))

    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
        
        img_s_label = tf.convert_to_tensor(np.array(img_label, dtype="float32"))
        img_s_input = tf.convert_to_tensor(np.array(img_input, dtype="float32"))
        
        img_s_label = tf.image.rgb_to_grayscale(img_s_label[0:256,0:256,0:3])
        img_s_input = tf.image.rgb_to_grayscale(img_s_input[0:256,0:256,0:3])

        img_s_input_batch = tf.expand_dims(img_s_input, axis = 0)
        img_s_label_batch = tf.expand_dims(img_s_label, axis = 0)

        output = model.predict(img_s_input_batch)

        reconstructed = img_s_input_batch + output

        org_psnr[i] = tf.image.psnr(img_s_label_batch, img_s_input_batch, 255.0).numpy()
        rec_psnr[i] = tf.image.psnr(reconstructed, img_s_label_batch, 255.0).numpy()
        
        org_ssim[i] = tf.image.ssim(img_s_label_batch, img_s_input_batch, max_val=255.0)
        rec_ssim[i] = tf.image.ssim(reconstructed, img_s_input_batch, max_val=255.0)
        print('Image ' + str(i) + ' org_psnr:%.4f,' % org_psnr[i] + 'after_psnr:%.4f,' % rec_psnr[i] 
                + 'org_ssim:%.4f,' % org_ssim[i]+ 'after_ssim:%.4f' % rec_ssim[i])
    
    print('average org_psnr:%.4f' % np.mean(org_psnr))
    print('average after_psnr:%.4f' % np.mean(rec_psnr))
    print('average org_ssim:%.4f' % np.mean(org_ssim))
    print('average after_ssim:%.4f' % np.mean(rec_ssim))
    
