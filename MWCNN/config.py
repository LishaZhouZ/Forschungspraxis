import numpy as np

debug_mode = True
channel = 3
batch_size = 24
patch_size = 256
epochs = 44
record_step = 10
decay_lr = np.ones(epochs+1)
decay_lr[0:int(epochs/4)]= 0.01
decay_lr[int(epochs/4):int(epochs/2)] = 0.01/2
decay_lr[int(epochs/2):int(epochs*3/4)] = 0.01/5
decay_lr[int(epochs*3/4):epochs+1] =0.01/10

if debug_mode == True:
    batch_size = 4

#model config
filter_num = 64