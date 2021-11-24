import glob, os
import pandas as pd
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import img_as_ubyte, img_as_float
from imageio import imread
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(19.2, 10.8), dpi=100)

data_path = '/media/samir/Secondary/Datasets/SIDD/SIDD_Small_sRGB_Only/Data/'
results_path = '/media/samir/Secondary/Denoising/results/'

dirs = os.listdir(data_path)
num_images = len(dirs)

psnr_array = np.zeros((4, num_images))
ssim_array = np.zeros((4, num_images))
name_array = []

for i in range(num_images):
	print("Calculating Metrics for Image:", i+1)

	folder_path = data_path + dirs[i]
	clean = img_as_float(imread(folder_path + '/GT_SRGB_010.PNG'))
	noisy = img_as_float(imread(folder_path + '/NOISY_SRGB_010.PNG'))

	# Load Images
	denoised_median = img_as_float(imread(results_path + 'median results/' + dirs[i] + '/DENOISED_SRGB_010.PNG'))
	denoised_wiener = img_as_float(imread(results_path + 'wiener results/' + dirs[i] + '/DENOISED_SRGB_010.PNG'))
	denoised_dncnn = img_as_float(imread(results_path + 'dncnn results/' + dirs[i] + '/DENOISED_SRGB_010.PNG'))
	denoised_ffdnet = img_as_float(imread(results_path + 'ffdnet results/' + dirs[i] + '/DENOISED_SRGB_010.PNG'))

	# Get Difference Maps
	diff = noisy - clean
	diff1 = noisy - denoised_median
	diff2 = noisy - denoised_wiener
	diff3 = noisy - denoised_dncnn
	diff4 = noisy - denoised_ffdnet

	# Get Noise Histograms
	histogram, bin_edges = np.histogram(diff, bins=256, range=(-0.5, 127/256))
	histogram1, bin_edges = np.histogram(diff1, bins=256, range=(-0.5, 127/256))
	histogram2, bin_edges = np.histogram(diff2, bins=256, range=(-0.5, 127/256))
	histogram3, bin_edges = np.histogram(diff3, bins=256, range=(-0.5, 127/256))
	histogram4, bin_edges = np.histogram(diff4, bins=256, range=(-0.5, 127/256))
	plt.plot(bin_edges[0:-1], histogram, label='Grouth Truth')
	plt.plot(bin_edges[0:-1], histogram1, label='Median')
	plt.plot(bin_edges[0:-1], histogram2, label='Wiener')
	plt.plot(bin_edges[0:-1], histogram3, label='DnCNN')
	plt.plot(bin_edges[0:-1], histogram4, label='FFDNet')
	plt.legend()
	plt.title("Noise Level Function for " + dirs[i])
	plt.xlabel("Brightness: float32")
	plt.ylabel("Number of Pixels")
	plt.savefig(results_path + 'plots/' + dirs[i] + '.png')
	plt.clf()

	# Compute PSNR
	psnr_median = peak_signal_noise_ratio(clean, denoised_median)
	psnr_wiener = peak_signal_noise_ratio(clean, denoised_wiener)
	psnr_dncnn = peak_signal_noise_ratio(clean, denoised_dncnn)
	psnr_ffdnet = peak_signal_noise_ratio(clean, denoised_ffdnet)

	# Compute SSIM
	ssim_median = structural_similarity(clean, denoised_median, multichannel=True)
	ssim_wiener = structural_similarity(clean, denoised_wiener, multichannel=True)
	ssim_dncnn = structural_similarity(clean, denoised_dncnn, multichannel=True)
	ssim_ffdnet = structural_similarity(clean, denoised_ffdnet, multichannel=True)

	# Store PSNR and SSIM
	psnr_array[:, i] = [psnr_median, psnr_wiener, psnr_dncnn, psnr_ffdnet]
	ssim_array[:, i] = [ssim_median, ssim_wiener, ssim_dncnn, ssim_ffdnet]
	name_array.append(dirs[i])
	struct = np.concatenate((psnr_array, ssim_array), axis=0)

d = {
	'Median PSNR': struct[0, :], 
	'Wiener PSNR': struct[1, :], 
	'DnCNN PSNR': struct[2, :],
	'FFDNet PSNR': struct[3, :],
	'Median SSIM': struct[4, :],
	'Wiener SSIM': struct[5, :], 
	'DnCNN SSIM': struct[6, :], 
	'FFDNet SSIM': struct[7, :]
	}

df = pd.DataFrame(data=d, index=name_array)
df.to_excel('metrics.xlsx')