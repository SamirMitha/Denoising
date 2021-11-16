from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Lambda
from tensorflow.keras.models import Model

class DnCNN(Model):
	def __init__(self,
				img_size = (50, 50),
				kernel_size = (3, 3),
				dncnn_filters = 64,
				dncnn_layers = 17,
				dncnn_strides = (1, 1)
				):

		# Initializations
			super(DnCNN, self).__init__()
			self.img_size = img_size
			self.kernel_size = kernel_size
			self.dncnn_filters = dncnn_filters
			self.dncnn_layers = dncnn_layers
			self.dncnn_strides = dncnn_strides


	def get_model(self):
		return self.__forward()


	def conv_block(self, input_tensor, kernel_size, dncnn_filters, dncnn_strides=1, normalization_flag=True):
		conv1 = Conv2D(filters=dncnn_filters, kernel_size=kernel_size, padding='same')(input_tensor)
		if normalization_flag == True:
			conv1 = BatchNormalization()(conv1)
		relu1 = ReLU()(conv1)
		return (relu1)


	def dncnn_net(self, input_tensor, kernel_size, dncnn_filters, dncnn_layers, dncnn_strides=1):
		conv1 = self.conv_block(input_tensor, kernel_size, dncnn_filters, normalization_flag=False)
		for i in range(2, dncnn_layers):
			conv1 = self.conv_block(conv1, kernel_size, dncnn_filters)
		conv2 = Conv2D(filters=3, kernel_size=kernel_size, padding='same')(conv1)
		return (conv2)


	def __forward(self):
		inputs = Input(shape=(None, None, 3))
		dncnn_network = self.dncnn_net(inputs, self.kernel_size, self.dncnn_filters, self.dncnn_layers)
		outputs = inputs - dncnn_network
		model = Model(inputs=inputs, outputs=outputs)
		return (model)


if __name__ == '__main__':
	DnCNN = DnCNN()
	model = DnCNN.get_model()
	print(model.summary())