import tensorflow as tf

def mse(true, pred):
	mse = tf.keras.losses.MeanSquaredError()
	mse1 = mse(true, pred)
	return (mse1)