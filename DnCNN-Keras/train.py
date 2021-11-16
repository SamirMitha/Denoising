import glob, os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
import pickle
from models import DnCNN
from losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
from data_generator import DataGenerator
import matplotlib.pyplot as plt

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

# Defaults
n_channels = 3
batch_size = 128
epochs = 50
learning_rate = 1e-5
model_loss = 'mse'
monitor = 'val_loss'
train_split = 0.8
validation_split = 0.1
test_split = 0.1
checkpoint = 00

# Directories
noisy_train_path = '/media/samir/Secondary/Denoising/dncnn/test/SIDD/train/noisy/'
gt_train_path = '/media/samir/Secondary/Denoising/dncnn/test/SIDD/train/gt/'
noisy_val_path = '/media/samir/Secondary/Denoising/dncnn/test/SIDD/val/noisy/'
gt_val_path = '/media/samir/Secondary/Denoising/dncnn/test/SIDD/val/gt/'
model_path = '/media/samir/Secondary/Denoising/dncnn/models/'
model_name = 'DnCNN_Default_SIDD_'

# Create output directories
if(not os.path.isdir(model_path) or not os.listdir(model_path)):
    os.makedirs(model_path + '/logs')
    os.makedirs(model_path + '/models')
    os.makedirs(model_path + '/history')
    os.makedirs(model_path + '/figures')
    os.makedirs(model_path + '/params')
    os.makedirs(model_path + '/checkpoints')

# Create train list
train_names = glob.glob(noisy_train_path + '/*.png')
num_imgs = len(train_names)
idx = np.arange(num_imgs)
train_ids = idx

# Create validation lsit
val_names = glob.glob(noisy_val_path + '/*.png')
v_num_imgs = len(val_names)
idx = np.arange(v_num_imgs)
val_ids = idx

# Create generators
train_gen = DataGenerator(noisy_path=noisy_train_path, gt_path=gt_train_path, batch_size=batch_size)
val_gen = DataGenerator(noisy_path=noisy_val_path, gt_path=gt_val_path, batch_size=batch_size)

# Model Parameters
params = dict()
params['Number of channels'] = n_channels
params['Batch Size'] = batch_size
params['Epochs'] = epochs
params['Learning rate'] = learning_rate
params['Training split'] = train_split
params['Validation split'] = validation_split
params['Testing split'] = test_split

print(['Model Parameters'])
print('------------')
for key in params.keys():
    print(key + ':', params[key])

# Create Model
DnCNN = DnCNN()
model = DnCNN.get_model()

# Model Summary
print(model.summary())

# Compile Model
model.compile(optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0001),
                loss='mean_squared_error')
callbacks = []

# CSV Logger
callbacks.append(CSVLogger(model_path + '/logs/' + model_name + '.csv'))

# Model Checkpoints
callbacks.append(ModelCheckpoint(model_path + '/checkpoints/' + 'epoch-{epoch:02d}/' + model_name + '.h5', monitor=monitor, save_freq=100))

# Stop on NaN
callbacks.append(TerminateOnNaN())

# Fit model
start_time = time.time()

print("Starting Training...")
model_history = model.fit(train_gen, 
                                    steps_per_epoch=len(train_ids)//batch_size,
                                    validation_data=val_gen, 
                                    validation_steps=len(val_ids)//batch_size,
                                    verbose=1, epochs=epochs, callbacks=callbacks)
print("...Finished Training")

elapsed_time = time.time() - start_time

# Save history
with open(model_path + '/history/' + model_name, 'wb') as fp:
    pickle.dump(model_history.history, fp)

# Save parameters
params['Training Times'] = elapsed_time
f = open(model_path + '/params/' + model_name + '.txt', 'w')
f.write('[Model Parameters]' + '\n')
f.write('------------' + '\n')
for k, v in params.items():
    f.write(str(k) + ': '+ str(v) + '\n')
f.close()

timestr = time.strftime('%Y%m%d-%H%M%S.h5')
model.save(model_path + '/models/' + model_name + timestr)
print('Model saved successfully.')

# Display loss curves
fig, ax = plt.subplots(1, 1)
ax.plot(model_history.history['loss'], color='blue', label='Training Loss')
ax.plot(model_history.history['val_loss'], color='orange', label='Validation Loss')
ax.set_title('Loss Curves')
ax.set_ylabel(model_loss)
ax.set_xlabel('Epochs')
plt.legend()

# Save figure
plt.savefig(model_path + '/figures/' + model_name + '.png')
print('Loss figure saved successfully.')