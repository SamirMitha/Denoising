from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Lambda
from tensorflow.keras.models import Model
from utils import upscale_images, upscale_tensor
import numpy as np
import tensorflow as tf

class FFDNet(Model):
	def __init__(self,
				img_size = (50, 50),
				kernel_size = (3, 3),
				ffdnet_filters = 96,
				ffdnet_layers = 12,
				ffdnet_strides = (1, 1)
				):

		# Initializations
			super(FFDNet, self).__init__()
			self.img_size = img_size
			self.kernel_size = kernel_size
			self.ffdnet_filters = ffdnet_filters
			self.ffdnet_layers = ffdnet_layers
			self.ffdnet_strides = ffdnet_strides


	def get_model(self):
		return self.__forward()


	def conv_block(self, input_tensor, kernel_size, ffdnet_filters, ffdnet_strides=1, normalization_flag=True):
		conv1 = Conv2D(filters=ffdnet_filters, kernel_size=kernel_size, padding='same')(input_tensor)
		if normalization_flag == True:
			conv1 = BatchNormalization()(conv1)
		relu1 = ReLU()(conv1)
		return (relu1)


	def ffdnet_net(self, input_tensor, kernel_size, ffdnet_filters, ffdnet_layers, ffdnet_strides=1):
		conv1 = self.conv_block(input_tensor, kernel_size, ffdnet_filters, normalization_flag=False)
		for i in range(2, ffdnet_layers):
			conv1 = self.conv_block(conv1, kernel_size, ffdnet_filters)
		conv2 = Conv2D(filters=12, kernel_size=kernel_size, padding='same')(conv1)
		return (conv2)


	def __forward(self):
		inputs = Input(shape=(self.img_size[0] // 2, self.img_size[1] // 2, 15))
		ffdnet_network = self.ffdnet_net(inputs, self.kernel_size, self.ffdnet_filters, self.ffdnet_layers)
		out = Lambda(upscale_tensor, name='out')(ffdnet_network)
		outputs = out
		model = Model(inputs=inputs, outputs=outputs)
		return (model)


if __name__ == '__main__':
	FFDNet = FFDNet()
	model = FFDNet.get_model()
	print(model.summary())