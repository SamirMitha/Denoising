import numpy as np
import glob, os
from imageio import imread, imwrite
from scipy.signal.signaltools import wiener
from skimage import img_as_ubyte, img_as_float
import time

# noise based on local variance of input

start_time = time.time()

data_path = '/media/samir/Secondary/Datasets/SIDD/SIDD_Small_sRGB_Only/Data/'
save_path = '/media/samir/Secondary/Denoising/wiener results/'
kernel_size = (5, 5)

dirs = os.listdir(data_path)
num_images = len(dirs)
for i in range(num_images):
	print("Wiener Filtering Image:", i+1)
	folder_path = data_path + dirs[i]
	noisy = img_as_float(imread(folder_path + '/NOISY_SRGB_010.PNG').astype('float64'))/255
	
	save_folder_path = save_path + dirs[i]
	if(not os.path.isdir(save_folder_path)):
		os.makedirs(save_folder_path)

	denoised_r = wiener(noisy[:,:,0], kernel_size)
	denoised_g = wiener(noisy[:,:,1], kernel_size)
	denoised_b = wiener(noisy[:,:,2], kernel_size)
	denoised = np.stack((denoised_r, denoised_g, denoised_b), axis=2)

	denoised[np.isnan(denoised)] = 0
	denoised = np.clip(denoised, 0, 1)
	imwrite(save_folder_path + '/DENOISED_SRGB_010.PNG', img_as_ubyte(denoised))
print("Execution Time: %s s" % (time.time() - start_time))

# Execution Time: 2756.65567278862 s