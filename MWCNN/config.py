import numpy as np

debug_mode = False
channel = 3
batch_size = 24
patch_size = 256
epochs = 40
record_step = 1
alpha = 0.001
decay_lr = np.ones(epochs+1)
decay_lr[0:11]= alpha
decay_lr[11:21] = alpha/2
decay_lr[21:31] = alpha/5
decay_lr[31:41] = alpha/10
#decay_lr[40:50] = alpha/20
#decay_lr[50:60] = alpha/50
#decay_lr[60:epochs+1] = alpha/100

checkpoint_directory = './tf_ckpts'

if debug_mode == True:
    batch_size = 8
    epochs = 20
    record_step = 1

