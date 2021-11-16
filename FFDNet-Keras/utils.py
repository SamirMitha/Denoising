import numpy as np
import glob, os
import random
from imageio import imread, imwrite
import shutil
from skimage import img_as_ubyte, img_as_float
from math import ceil, sqrt, pi
from scipy.signal import convolve2d
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D

def generate_patches_pngs(noisy_image_dir, gt_image_dir, save_dir, batch_size, patch_size, total_samples):
	# for saving patches to directory
	train_size = 0.8*total_samples
	validate_size = 0.1*total_samples
	test_size = 0.1*total_samples

	dirs = os.listdir(noisy_image_dir)
	num_images = len(dirs)
	idx = list(range(num_images))
	if(not os.path.isdir(save_dir) or not os.listdir(save_dir)):
		os.makedirs(save_dir + '/train')
		os.makedirs(save_dir + '/validate')
		os.makedirs(save_dir + '/test')
		os.makedirs(save_dir + '/train/noisy')
		os.makedirs(save_dir + '/train/gt')
		os.makedirs(save_dir + '/validate/noisy')
		os.makedirs(save_dir + '/validate/gt')
		os.makedirs(save_dir + '/test/noisy')
		os.makedirs(save_dir + '/test/gt')
		os.makedirs(save_dir + '/temp_noisy')
		os.makedirs(save_dir + '/temp_gt')
	for j in range(total_samples):
		random.shuffle(idx)
		for i in range(batch_size):
			ridx = idx[i]
			img_noisy = imread(noisy_image_dir + dirs[i] + '/NOISY_SRGB_010.PNG')
			img_gt = imread(gt_image_dir + dirs[i] + '/GT_SRGB_010.PNG')
			#img_noisy = imread(noisy_image_dir + dirs[i])
			#img_gt = imread(gt_image_dir + dirs[i])
			sample_point_x = random.randint(0, img_noisy.shape[0]-patch_size[0])
			sample_point_y = random.randint(0, img_noisy.shape[1]-patch_size[1])
			patch_noisy = img_noisy[sample_point_x:sample_point_x+patch_size[0], sample_point_y:sample_point_y+patch_size[1], :]
			patch_gt = img_gt[sample_point_x:sample_point_x+patch_size[0], sample_point_y:sample_point_y+patch_size[1], :]
			imwrite(save_dir + 'temp_noisy/' + str(j*batch_size+i)+'_noisy.png', patch_noisy)
			imwrite(save_dir + 'temp_gt/' + str(j*batch_size+i)+'_gt.png', patch_gt)
			print('Generating patches:' + str(j*batch_size+i+1) + '/' + str(total_samples*batch_size))
	
	directory1 = (save_dir + 'temp_noisy/')
	directory2 = (save_dir + 'temp_gt/')
	files1 = os.listdir(directory1)
	len_files = len(files1)
	train_len = int(0.8*(len_files))
	test_len = int(0.1*(len_files))
	val_len = len_files - (train_len + test_len)

	rand_perm = np.random.permutation(files1)
	train_perm_noisy = rand_perm[:train_len]
	test_perm_noisy = rand_perm[train_len:(train_len + test_len)]
	val_perm_noisy = rand_perm[(train_len + test_len):(train_len + test_len + val_len)]

	for file in train_perm_noisy:
		orig = directory1 + file
		orig2 = directory2 + file.replace('_noisy.png', '_gt.png')
		new = save_dir + 'train/noisy/' + file
		new2 = save_dir + 'train/gt/' + file.replace('_noisy.png', '_gt.png')
		shutil.move(orig, new)
		shutil.move(orig2, new2)

	for file in test_perm_noisy:
		orig = directory1 + file
		orig2 = directory2 + file.replace('_noisy.png', '_gt.png')
		new = save_dir + 'test/noisy/' + file
		new2 = save_dir + 'test/gt/' + file.replace('_noisy.png', '_gt.png')
		shutil.move(orig, new)
		shutil.move(orig2, new2)

	for file in val_perm_noisy:
		orig = directory1 + file
		orig2 = directory2 + file.replace('_noisy.png', '_gt.png')
		new = save_dir + 'validate/noisy/' + file
		new2 = save_dir + 'validate/gt/' + file.replace('_noisy.png', '_gt.png')
		shutil.move(orig, new)
		shutil.move(orig2, new2)


def normalize(img):
	normalized = img_as_float(img).astype('float64')
	return (normalized)


def denormalize(img):
	img = np.clip(img, 0, 1)
	denormalized = img_as_ubyte(img)
	return (denormalized)


def img_to_patches(img, patch_size):
	# converts image to patches to be used during evaluation
	img_size = img.shape
	m = int(ceil(img_size[0]/patch_size[0]))
	n = int(ceil(img_size[1]/patch_size[1]))

	img_new_size = (patch_size[0] * m, patch_size[1] * n, patch_size[2])

	padding = tuple(np.subtract(img_new_size, img_size))
	padding_full = ((0, padding[0]), (0, padding[1]), (0, padding[2]))

	padded = np.pad(img, pad_width = padding_full, mode='reflect')

	num_patches = m * n

	patches = np.zeros((num_patches, *patch_size))

	for i in range(m):
		for j in range(n):
			bbox1 = i * patch_size[0]
			bbox2 = (i+1) * patch_size[0]
			bbox3 = j * patch_size[1]
			bbox4 = (j+1) * patch_size[1]
			patch = padded[bbox1:bbox2, bbox3:bbox4, :]
			patches[n * i + j, :, :, :] = patch

	return (patches)


def patches_to_img(patches, img_size):
	# converts patches to image to be used during evaluation
	patch_size = patches.shape
	patch_size = (patch_size[1], patch_size[2], patch_size[3])

	m = int(ceil(img_size[0]/patch_size[0]))
	n = int(ceil(img_size[1]/patch_size[1]))
	padded = np.zeros((m*patch_size[0], n*patch_size[1], patch_size[2]))

	for i in range(m):
		for j in range(n):
			bbox1 = i * patch_size[0]
			bbox2 = (i+1) * patch_size[0]
			bbox3 = j * patch_size[1]
			bbox4 = (j+1) * patch_size[1]
			patch = patches[n * i + j, :, :, :]
			padded[bbox1:bbox2, bbox3:bbox4, :] = patch

	img = padded[0:(img_size[0]), 0:(img_size[1]), 0:(img_size[2])]

	return (img)


def downsample_images(patch):
	# downsample original image to 4 subimages
	patch_size = patch.shape
	downsampled_shape = (patch_size[0] // 2, patch_size[1] // 2, 4*patch_size[2])
	downsampled = np.zeros((downsampled_shape))

	downsampled1 = patch[0:patch_size[0]:2, 0:patch_size[1]:2]
	downsampled2 = patch[0:patch_size[0]:2, 1:patch_size[1]:2]
	downsampled3 = patch[1:patch_size[0]:2, 0:patch_size[1]:2]
	downsampled4 = patch[1:patch_size[0]:2, 1:patch_size[1]:2]

	downsampled[:, :, 0:3] = downsampled1
	downsampled[:, :, 3:6] = downsampled2
	downsampled[:, :, 6:9] = downsampled3
	downsampled[:, :, 9:12] = downsampled4

	return (downsampled)


def upscale_images(downsampled_patches):
	downsampled_patches_size = downsampled_patches.shape
	upscaled_image_size =  (downsampled_patches_size[0] * 2, downsampled_patches_size[1] * 2, downsampled_patches_size[2] // 4)
	upscaled = np.zeros((upscaled_image_size))

	downsampled1 = downsampled_patches[:, :, 0:3]
	downsampled2 = downsampled_patches[:, :, 3:6]
	downsampled3 = downsampled_patches[:, :, 6:9]
	downsampled4 = downsampled_patches[:, :, 9:12]

	idx1 = np.arange(0, upscaled_image_size[0], 2)
	idx2 = np.arange(1, upscaled_image_size[1], 2)

	for i in range(downsampled_patches_size[1]):
		for j in range(downsampled_patches_size[2]):
			upscaled[2*i, 2*j, :] = downsampled1[i, j, :]
			upscaled[2*i, 2*j+1, :] = downsampled2[i, j, :]
			upscaled[2*i+1, 2*j, :] = downsampled3[i, j, :]
			upscaled[2*i+1, 2*j+1, :] = downsampled4[i, j, :]

	return (upscaled)


def upscale_tensor(downsampled_patches):

	# split
	downsampled = tf.split(downsampled_patches, num_or_size_splits = 4, axis=3)
	downsampled1 = downsampled[0]
	downsampled2 = downsampled[1]
	downsampled3 = downsampled[2]
	downsampled4 = downsampled[3]

	# upsample
	upsampled1 = UpSampling2D(size=(2,2))(downsampled1)
	upsampled2 = UpSampling2D()(downsampled2)
	upsampled3 = UpSampling2D()(downsampled3)
	upsampled4 = UpSampling2D()(downsampled4)

	# mask
	downsampled_patches_size = downsampled_patches.shape
	upscaled_image_size = upsampled1.shape
	upscaled_size = (upscaled_image_size[1], upscaled_image_size[2], upscaled_image_size[3])

	mask1 = np.zeros(upscaled_size)
	mask2 = np.zeros(upscaled_size)
	mask3 = np.zeros(upscaled_size)
	mask4 = np.zeros(upscaled_size)

	for i in range(downsampled_patches_size[1]):
		for j in range(downsampled_patches_size[2]):
			mask1[2*i, 2*j, :] = 1
			mask2[2*i, 2*j+1, :] = 1
			mask3[2*i+1, 2*j, :] = 1
			mask4[2*i+1, 2*j+1, :] = 1

	mask1 = tf.convert_to_tensor(mask1, dtype=tf.float32)
	mask2 = tf.convert_to_tensor(mask2, dtype=tf.float32)
	mask3 = tf.convert_to_tensor(mask3, dtype=tf.float32)
	mask4 = tf.convert_to_tensor(mask4, dtype=tf.float32)

	# multiply
	masked1 = tf.math.multiply(upsampled1, mask1)
	masked2 = tf.math.multiply(upsampled2, mask2)
	masked3 = tf.math.multiply(upsampled3, mask3)
	masked4 = tf.math.multiply(upsampled4, mask4)

	# add
	out = tf.math.add(masked1, masked2)
	out = tf.math.add(out, masked3)
	out = tf.math.add(out, masked4)

	return (out)


def noise_map(patch):
	# https://www.sciencedirect.com/science/article/abs/pii/S1077314296900600
	patch_size = patch.shape

	M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

	sigma_r = np.sum(np.sum(np.absolute(convolve2d(patch[:, :, 0], M))))
	sigma_g = np.sum(np.sum(np.absolute(convolve2d(patch[:, :, 1], M))))
	sigma_b = np.sum(np.sum(np.absolute(convolve2d(patch[:, :, 2], M))))

	sigma_r = sigma_r * sqrt(0.5 * pi) / (6 * (patch_size[0]-2) * (patch_size[1]-2))
	sigma_g = sigma_g * sqrt(0.5 * pi) / (6 * (patch_size[0]-2) * (patch_size[1]-2))
	sigma_b = sigma_b * sqrt(0.5 * pi) / (6 * (patch_size[0]-2) * (patch_size[1]-2))

	downsampled_shape = (patch_size[0] // 2, patch_size[1] // 2, patch_size[2])

	noise_map = np.ones((downsampled_shape))
	noise_map[:, :, 0] = noise_map[:, :, 0] * sigma_r
	noise_map[:, :, 1] = noise_map[:, :, 1] * sigma_g
	noise_map[:, :, 2] = noise_map[:, :, 2] * sigma_b

	return(noise_map)


def ffdnet_struct(patch):
	# combine downsampled images with noise map
	ffdnetstruct = np.zeros((25, 25, 15))

	downsampled_images = downsample_images(patch)

	noise = noise_map(patch)
	noise = noise.reshape((1, *noise.shape))

	ffdnetstruct[:, :, 0:12] = downsampled_images
	ffdnetstruct[:, :, 12:15] = noise

	return (ffdnetstruct)
