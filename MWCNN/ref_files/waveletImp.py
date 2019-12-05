# db1 is corresponding to the haar-wavelet
import pywt
import tensorflow as tf
from utils_py3_tfrecord_2 import *
import matplotlib.pyplot as plt


def dwt(image):
    print(pywt.families())
    # for family in pywt.families():  #打印出每个小波族的每个小波函数
    # print('%s family: '%(family) + ','.join(pywt.wavelist(family)))
    db3 = pywt.Wavelet('haar')  # 创建一个小波对象

    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

    print(db3)


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.imshow(label_batch[n])
        plt.axis('off')


if __name__ == '__main__':
    train_dataset = read_and_decode('./patches/MWCNN_train_data_debug.tfrecords', 2, 192)
    img_batch, label_batch = next(iter(train_dataset))
    print(img_batch)
    for image, label in train_dataset.take(1):
        for image1 in image.take(1):
            print("Image shape: ", image1.numpy().shape)
            plt.axis("off")
            plt.imshow(image1/255)
            plt.show()
        
    #dwt(img_batch)
    print(train_dataset)
    show_batch(img_batch.numpy(), label_batch.numpy())
