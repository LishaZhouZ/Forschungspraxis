from memory_profiler import profile
from tensorflow import keras
import tensorflow as tf
import numpy as np
#from train_MWCNN import *
from seq_model_MWCNN import *
from pathlib import Path


@tf.function
def one_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        reconstructed = model(images, training = True)
        output = images + reconstructed
        total_loss = loss_fn(model, output, labels)
    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    return total_loss


@profile
def test():
    print(tf.executing_eagerly())
    optimizer = keras.optimizers.Adam(lr=1e-3)

    model = MWCNN()

    optimizer = tf.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, name='AdamOptimizer')
    
    train_dataset = read_and_decode(
        '../patches/MWCNN_train_data.tfrecords')
    
    
    
    for epoch in range(1, 2):
        run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
        print('Start of epoch %d' % (epoch,))
        # iterate over the batches of the dataset.
        
        #self.ckpt.step.assign_add(1)
        ## main step
        #decay step size is an interface parameter
        optimizer.learning_rate = decay_lr[epoch]
        ###
        # Create your tf representation of the iterator
        for step, (images, labels) in enumerate(train_dataset.take(2)):
            total_loss = one_step(model, images, labels, optimizer)
            #converted = wavelet_conversion(images)
            

        #if int(count.numpy()) % record_step == 0:
        #    with writer.as_default():
        #        tf.summary.scalar('train_loss', total_loss, step = count.numpy())
            print("Step " + str(step) + " loss {:1.2f}".format(total_loss))



if __name__ == '__main__':
    print(tf.executing_eagerly())
    physical_devices = tf.config.experimental.list_physical_devices('GPU') 
    try: 
        tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        assert tf.config.experimental.get_memory_growth(physical_devices[0]) 
    except: 
        # Invalid device or cannot modify virtual devices once initialized. 
        pass

    #model =tf.keras.Sequential()
    #model.add(WaveletConvLayer())


    #model.add()
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.summary()

    path0 = Path('/home/lisha/Forschungspraxis/images/test/test/compressed_Q10_5/airplane.jpeg')
    path1= Path('/home/lisha/Forschungspraxis/images/test/test/groundtruth_5/airplane.bmp')
    img0 = Image.open(path0)
    img1 =Image.open(path1)
    fraction0 = np.array(img0, dtype="uint8")
    fraction1 = np.array(img1, dtype="uint8")
    con2 = np.concatenate(
             [fraction1, fraction0], axis=1) 
    plt.imshow(con2)
    plt.show()
    #a=fraction[:,:,2]
    #a=np.expand_dims(a, axis=2)
    #img_label_s = np.expand_dims(fraction, axis=0)
    #output = model(img_label_s)

    # print(img_label_s-output)
    # d=np.zeros((512,512,3), dtype='uint8')
    # d = np.array(output[0,:,:,0:3]).astype('uint8')
    # plt.imshow(d)
    # plt.show()
    #temp = np.zeros((256,256,3), dtype='uint8')
    

    # a=np.zeros((256,256,3), dtype='uint8')
    # a[:,:,0] =np.clip(output[0,:,:,0], 0, 255)
    # #a[:,:,1] =np.clip(output[0,:,:,1], 0, 255)
    # #a[:,:,2] =np.clip(output[0,:,:,2], 0, 255)
    # b=np.zeros((256,256,3), dtype='uint8')
    # b[:,:,0] =output[0,:,:,3]
    # #b[:,:,1] =output[0,:,:,4]
    # #b[:,:,2] =output[0,:,:,5]
    # c=np.zeros((256,256,3), dtype='uint8')
    # c[:,:,0] = output[0,:,:,6]
    # #c[:,:,1] = output[0,:,:,7]
    # #c[:,:,2] = output[0,:,:,8]
    # d=np.zeros((256,256,3), dtype='uint8')
    # d[:,:,0] =  output[0,:,:,9]
    # #d[:,:,1] =  output[0,:,:,10]
    # #d[:,:,2] =  output[0,:,:,11]
    # con_row1= np.concatenate(
    #         [a, b], axis=1)
    # con_row2 =np.concatenate(
    #         [c, d], axis=1)
    # con2 = np.concatenate(
    #         [con_row1, con_row2], axis=0) 

    # plt.imshow(con2)
    # plt.axis('off')

    

    # plt.show()
