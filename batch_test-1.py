import numpy as np
import scipy as sp
import scipy.signal as sig
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers, callbacks
from data_wrangling.datamanager import DataLoader
import pickle
import sys
from datetime import datetime

inputs = sys.argv[1:]
depth_pre  = int(inputs[0])
width_pre  = int(inputs[1])
width_mid  = int(inputs[2])
depth_post = int(inputs[3])
width_post = int(inputs[4])
dropout    = float(inputs[5])
identifier = int(inputs[6])

# Load data
X1 = np.load('X1.npy')
X = np.load('X.npy')
Y = np.load('Y.npy')
idxs_train_test_split = np.load('idxs_train_test_split.npy')

# dl = DataLoader()
# data = dl.get_fcx2(['s1'])

# Basic preprocessing
# X1 = data['s1']['data']
# mn = X1.mean(0).reshape((1, -1))
# st = X1.std (0).reshape((1, -1))
# X1 = (X1 - mn) / st
# X1.shape

# Data organization
fs = 1000
# idxs_train_test_split = int(len(X1) * 0.7)
p_in  = 100
p_out = 10
step  = 15

idxs_train = np.arange(0, idxs_train_test_split - p_out - p_in, step)

# X = np.stack([
#     X1[idx : idx + p_in]
#     for idx in idxs_train
# ])

# Y = np.stack([
#     X1[idx + p_in : idx + p_in + p_out]
#     for idx in idxs_train
# ])

# Model building
model = keras.Sequential([
    layers.Input(X.shape[1:])
])

# Pre
for _ in range(depth_pre - 1):
    model.add(layers.Bidirectional(
        layers.LSTM(
            width_pre, 
            return_sequences = True, 
            dropout = dropout
        )
    ))
model.add(layers.Bidirectional(
    layers.LSTM(
        width_pre, 
        dropout = dropout
    )
))

# Mid
model.add(layers.RepeatVector(Y.shape[1]))
model.add(layers.LSTM(
    width_mid, 
    return_sequences = True, 
    dropout = dropout
))

# Post
for _ in range(depth_pre - 1):
    model.add(layers.Bidirectional(
        layers.LSTM(
            width_pre, 
            return_sequences = True, 
            dropout = dropout
        )
    ))
model.add(layers.Bidirectional(
    layers.LSTM(
        width_pre, 
        return_sequences = True, 
        dropout = dropout
    )
))
model.add(layers.Dense(Y.shape[-1]))

# Compile
model.compile(loss = 'mse', optimizer = 'adam')

start = datetime.now()
# Coarse training
hist1 = model.fit(
    X,
    Y,
    epochs = 100,
    batch_size = 2048,
    validation_split = 0.2,
    callbacks = [callbacks.EarlyStopping(min_delta = 1e-3, patience = 3)]
)

# Fins training
hist2 = model.fit(
    X,
    Y,
    epochs = 100,
    batch_size = 128,
    validation_split = 0.3,
    callbacks = [callbacks.EarlyStopping(min_delta = 5e-4, patience = 5)]
)
time_took = (datetime.now() - start).seconds

pred = []
real = []
for idx in range(idxs_train_test_split, len(X1) - p_out - p_in, p_out):
    x = X1[idx : idx + p_in].reshape((1, p_in, -1))
    y = X1[idx + p_in : idx + p_in + p_out]
    pred.append(x)
    real.append(y)
    
per_stride = 2048
strides = 1 + len(pred) // per_stride
pred = np.concatenate([
    model(
        np.concatenate(pred[stride * per_stride : (stride + 1) * per_stride])
    ).numpy()
    for stride in range(strides)
])

real = np.stack(real)

mse = ((pred - real) ** 2).mean()

with open(f'results_batch_test_1/res-{identifier}', 'wb') as fp:
    pickle.dump({
        'mse': mse,
        'hists': [hist1.history, hist2.history],
        'n_params': model.count_params(),
        'time': time_took
    }, fp)