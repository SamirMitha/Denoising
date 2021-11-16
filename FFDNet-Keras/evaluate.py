import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
from utils import denormalize, normalize, ffdnet_struct, img_to_patches, patches_to_img
from models import FFDNet
from imageio import imread, imwrite
import time
import glob, os

start_time = time.time()
patch_size = (50, 50, 3)
downsampled_patch_size = (25, 25, 15)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Inputs
data_path = '/media/samir/Secondary/Datasets/SIDD/SIDD_Small_sRGB_Only/Data/'
save_path = '/media/samir/Secondary/Denoising/ffdnet results/'
model_path = '/media/samir/Secondary/Denoising/ffdnet/models/models/FFDNet_Default_SIDD_20211116-044556.h5'

# Loading Model
FFDNet = FFDNet()
model = FFDNet.get_model()
model.load_weights(model_path)

# Evaluation Loop
dirs = os.listdir(data_path)
num_images = len(dirs)
for i in range(num_images):
    print("FFDNet Denoising Image:", i+1)
    folder_path = data_path + dirs[i]
    
    save_folder_path = save_path + dirs[i]
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Split Image into Patches
    noisy = normalize(imread(folder_path + '/NOISY_SRGB_010.PNG'))
    noisy_patches = img_to_patches(noisy, patch_size)

    # Downsample Patches and Noise Map
    downsampled_stack = np.zeros((noisy_patches.shape[0], *downsampled_patch_size))
    for j in range(noisy_patches.shape[0]):
        fs = ffdnet_struct(noisy_patches[j, :, :, :])
        downsampled_stack[j, :, :, :] = fs

    # Predict Output
    denoised = model.predict(downsampled_stack)
    denoised = np.squeeze(denoised)

    # Convert Patches into Image
    denoised = patches_to_img(denoised, noisy.shape)
    imwrite(save_folder_path + '/DENOISED_SRGB_010.PNG', denormalize(denoised))

print("Execution Time: %s s" % (time.time() - start_time))

# Execution Time: 1498.528359413147 s
# 3GB VRAM