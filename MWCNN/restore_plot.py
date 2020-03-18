# this file restore the model and checkpoint from file, and to check resulting image
from seq_model_MWCNN import *
import os
from pathlib import Path
import matplotlib.pyplot as plt

checkpoint_directory = './tf_ckpts'

def restore_and_test_with_path(dir_label, dir_input):
    model = build_MWCNN()
    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))
    
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    input_data_gen = image_generator.flow_from_directory(directory=str(dir_input),
                                                     batch_size= 1,
                                                     target_size=(256, 256))

    label_data_gen = image_generator.flow_from_directory(directory=str(dir_input),
                                                     batch_size= 1,
                                                     target_size=(256, 256))
                                  




def restore_and_test_with_tfrecord(test_dataset):
    model = build_MWCNN()
    optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001, epsilon=1e-8, name='AdamOptimizer')
    
    ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer, net = model)
    name_ck = tf.train.latest_checkpoint(checkpoint_directory)
    status = ckpt.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()
    print("Restored from {}".format(name_ck))

    psnr = PSNRMetric()
    ms_ssim = MS_SSIMMetric()
    org_psnr = PSNRMetric()

    for label_val, images_val in test_dataset:
        output = model.predict(images_val)
        predict_val = images_val + output
        #plt.imshow(predict_val[1,:,:,:]/255)
        #plt.show()
        org_psnr.update_state(label_val, images_val)
        psnr.update_state(label_val, predict_val)
        ms_ssim.update_state(label_val, predict_val)

    val_psnr = psnr.result()
    ms_ssim = ms_ssim.result()
    org_psnr = org_psnr.result()
    relative_psnr = val_psnr - org_psnr
    print('Original psnr: %s' % (float(org_psnr),))
    print('Validation psnr: %s' % (float(val_psnr),))
    print('Gain: %s' % (float(relative_psnr),))
    print('Validation msssim: %s' % (float(ms_ssim),))


if __name__ == "__main__":
    src_dir_label = Path("./images/train/groundtruth")
    src_dir_input = Path("./images/train/CompressedQ10")

    tf.config.experimental_run_functions_eagerly(True)
    test_dataset = read_and_decode(
            './patches/MWCNN_validation_data.tfrecords')
    test_dataset = test_dataset.batch(batch_size)
    restore_and_test_with_tfrecord(test_dataset)
    