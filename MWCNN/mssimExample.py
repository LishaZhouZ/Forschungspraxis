from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import tensorflow as tf


tf.compat.v1.app.flags.DEFINE_string('original_image', None, 'Path to JPEG image.')
tf.compat.v1.app.flags.DEFINE_string('compared_image', None, 'Path to JPEG image.')
FLAGS = tf.compat.v1.app.flags.FLAGS



# X: (N,3,H,W) a batch of RGB images with values ranging from 0 to 255.
# Y: (N,3,H,W)  
#ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
#ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# or set 'size_average=True' to get a scalar value as loss.
#ssim_loss = ssim( X, Y, data_range=255, size_average=True) # return a scalar value
#ms_ssim_loss = ms_ssim( X, Y, data_range=255, size_average=True )

# or reuse windows with SSIM & MS_SSIM. 
#ssim_module = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)
#ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)

#ssim_loss = ssim_module(X, Y)
#ms_ssim_loss = ms_ssim_module(X, Y)

if __name__ == '__main__':
  with tf.io.gfile.GFile(name = FLAGS.original_image, mode = 'rb') as image_file:
    img1_str = image_file.read()
  with tf.io.gfile.GFile(name = FLAGS.compared_image, mode = 'rb') as image_file:
    img2_str = image_file.read()
  print(ms_ssim_loss = ms_ssim( img1_str, img2_str, data_range=255, size_average=True ))