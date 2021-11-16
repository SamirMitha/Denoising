import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
from utils import denormalize, normalize, ffdnet_struct
from models import FFDNet
from imageio import imread, imwrite


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
input_path = '/media/samir/Secondary/Denoising/dncnn/test/SIDD/train/noisy/184_noisy.png'
save_path = ''
model_path = '/media/samir/Secondary/Denoising/ffdnet/models/models/FFDNet_Default_SIDD_20211116-044556.h5'

# Loading Model
FFDNet = FFDNet()
model = FFDNet.get_model()
model.load_weights(model_path)

x = normalize(imread(input_path))
print(np.amax(x))
x = ffdnet_struct(x)
x = x.reshape(1, 25, 25, 15)
print(x.shape)
out = model.predict(x)
out = np.squeeze(out)
print(np.amax(out))
imwrite('test.png', denormalize(out))