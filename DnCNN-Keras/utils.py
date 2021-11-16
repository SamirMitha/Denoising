import numpy as np
import glob, os
import random
from imageio import imread, imwrite
import shutil
from skimage import img_as_ubyte, img_as_float
from math import ceil


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
