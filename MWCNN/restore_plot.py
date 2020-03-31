# this file restore the model and checkpoint from file, and to check resulting image
from seq_model_MWCNN import *
import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from PIL import Image

checkpoint_directory = './tf_ckpts'

def image_cutting(dir_label, dir_input):


    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))
    count = 0
    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
        
        img_s_label = np.array(img_label, dtype="uint8")
        img_s_input = np.array(img_input, dtype="uint8")

        im_h, im_w, im_c = img_s_label.shape
        print("The %dth image of %d training images" %(i+1, len(filepaths_label)))
        for x in range(0, im_h - patch_size, patch_size):
            for y in range(0, im_w - patch_size, patch_size):
                image_label = img_s_label[x:x + patch_size, y:y + patch_size, 0:3] # some images have an extra blank channel 
                image_bayer = img_s_input[x:x + patch_size, y:y + patch_size, 0:3]
                
                result_label = Image.fromarray((image_label).astype(np.uint8))
                result_label.save('./images/test/split_label/' + str(count) + '.bmp')

                result_input = Image.fromarray((image_bayer).astype(np.uint8))
                result_input.save('./images/test/split_input/' + str(count) + '.bmp')
                count += 1





if __name__ == "__main__":
    #image_cutting(Path('./images/test/live1_groundtruth'), Path('./images/test/live1_qp10'))
    #filepaths_label = sorted(Path('./images/test/split_label').glob('*'))
    #filepaths_input = sorted(Path('./images/test/split_input').glob('*'))
    path_label = Path('/home/lisha/Forschungspraxis/images/test/split_label/25.bmp')
    path_input = Path('/home/lisha/Forschungspraxis/images/test/split_input/25.bmp')

    img_label = Image.open(path_label)
    img_input = Image.open(path_input)
    img_s_label = np.array(img_label, dtype="float32")
    img_s_input = np.array(img_input, dtype="float32")
    
    img_s_label = img_s_label[0:256,0:256,0:3]
    img_s_input = img_s_input[0:256,0:256,0:3]

    im_h, im_w, im_c = img_s_label.shape
    img_s_input_batch = img_s_input.reshape(1, im_h, im_w, im_c)
    
    model = build_MWCNN()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
    ckpt.restore(tf.train.latest_checkpoint('/home/lisha/Forschungspraxis/logs/Training20200325/tf_ckpts')).expect_partial()
    output = model.predict(img_s_input_batch)
    reconstructed = img_s_input_batch + output

    reconstructed_s = reconstructed[0,:,:,:]
    print(tf.image.psnr(img_s_label, img_s_input, 255))
    print(tf.image.psnr(reconstructed, img_s_label, 255))

    f = plt.figure()
    
    f.add_subplot(1,3,1)
    plt.imshow(img_s_label/255)
    plt.axis('off')
    plt.title('original')

    f.add_subplot(1,3, 2)
    plt.imshow(reconstructed_s/255)
    plt.axis('off')
    plt.title('reconstructed')

    f.add_subplot(1,3, 3)
    plt.imshow(img_s_input/255)
    plt.axis('off')
    plt.title('input')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None, wspace = .001)
    plt.show()
    