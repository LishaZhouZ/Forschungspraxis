debug_mode = True
batch_size = 24
patch_size = 256
epochs = 40

if debug_mode == True:
    batch_size = 4
    epochs = 2

#model config
filter_num = 64