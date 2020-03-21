import numpy as np

debug_mode = False
channel = 3
batch_size = 8
patch_size = 256
epochs = 40
record_step = 10
alpha = 0.01
decay_lr = np.ones(epochs+1)
decay_lr[0:int(epochs/4)]= alpha
decay_lr[int(epochs/4):int(epochs/2)] = alpha/2
decay_lr[int(epochs/2):int(epochs*3/4)] = alpha/5
decay_lr[int(epochs*3/4):epochs+1] = alpha/10

checkpoint_directory = './tf_ckpts'

if debug_mode == True:
    batch_size = 8
    epochs = 1
    record_step = 10

