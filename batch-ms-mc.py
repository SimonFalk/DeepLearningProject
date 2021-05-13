import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_wrangling.datamanager import DataLoader as DL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import pickle

t_in  = int(sys.argv[1])
t_out = int(sys.argv[2])
architecture = int(sys.argv[3])

print(f'\n\n\n{t_in} - {t_out} - {architecture}\n\n\n')

dl = DL()
sessions = ['s1', 's4', 's5', 's6', 's7', 's10']
data = dl.get_fcx2(sessions)

# Retrieve arrays, normalize
def get_as_array(session):
    arrs = data[session]['data']
    mn = arrs.mean(0).reshape((1, -1))
    st = arrs.std (0).reshape((1, -1))
    arrs -= mn
    arrs /= st
    return arrs

Xs = [
    get_as_array(s)
    for s in sessions
]

# Pad data with appropriate amount of zeros
data = 0
n_c_max = max(Xs, key = lambda X: X.shape[1]).shape[1]
n_channelss = [X.shape[1] for X in Xs]

def pad_with_zeros(X):
    zeros = np.zeros((X.shape[0], n_c_max - X.shape[1]))
    out = np.concatenate([X, zeros], 1)
    return out

Xs = [pad_with_zeros(X) for X in Xs]

# Indexing/setup

train_frac = 0.5
val_frac   = 0.2

train_ranges = np.array([
    (
        0,
        int(X.shape[0] * train_frac) - t_in - t_out
    )
    for X in Xs
])

val_ranges = np.array([
    (
        train_max + t_in + t_out, 
        train_max + int(X.shape[0] * val_frac) - t_in - t_out
    )
    for (_, train_max), X in zip(train_ranges, Xs)
])

test_ranges = np.array([
    (
        val_max + t_in + t_out,
        len(X) - t_in - t_out
    )
    for (val_min, val_max), X in zip(val_ranges, Xs)
])

# Data generator

n_sessions = len(sessions)

def get_random_data_idxs(sess_no, n, mode = 'train'):
    if mode == 'train':
        _idxs = np.arange(*list(train_ranges[sess_no].copy()))
    elif mode == 'val':
        _idxs = np.arange(*list(val_ranges[sess_no].copy()))
    elif mode == 'test':
        _idxs = np.arange(*list(test_ranges[sess_no].copy()))
        
    np.random.shuffle(_idxs)
    return _idxs[:n]

def data_generator(
    batch_size,
    steps_per_epoch,
    epochs,
    mode = 'train'
):
    
    for _ in range(steps_per_epoch * epochs):
        
        # Select session indices
        counts = pd.Series(
            np.random.randint(n_sessions, size = batch_size)
        ).value_counts()

        # Session-wise indices
        idxs = [
            get_random_data_idxs(idx, n, mode = mode)
            for idx, n in enumerate(counts)
        ]

        # Create data
        x = np.concatenate([
            np.stack([
                np.concatenate([
                    X[idx + t], 
                    np.array([n_channels] * len(idx)).reshape((-1, 1))
                ], 1)
                for t in range(t_in) 
            ], 1)
            for n_channels, idx, X in zip(n_channelss, idxs, Xs)
        ])

        y = np.concatenate([
            np.stack([
                X[idx + t + t_in]
                for t in range(t_out) 
            ], 1)
            for idx, X in zip(idxs, Xs)
        ])
        
        yield (x, y)
        
# Model architecture
        
class Splitter(layers.Layer):
    def call(self, inputs):
        signals = inputs[:,:,:-1]
        labels  = tf.cast(inputs[:, :t_out, -1], tf.int32)
        return signals, labels
    
class Combiner(layers.Layer):
    def call(self, inputs):
        outputs, labels = inputs
        
        idxs_for_mask = tf.ones(tf.shape(outputs), dtype = tf.int32) * np.arange(outputs.shape[-1])
        lims_for_mask = tf.repeat(tf.expand_dims(labels, 2), outputs.shape[-1], 2)
        return outputs * tf.cast(idxs_for_mask < lims_for_mask, tf.float32)
    
if architecture == 1:
    
    input_layer = layers.Input((t_in, n_c_max + 1))
    signals, labels = Splitter()(input_layer)

    encoder = layers.Bidirectional(
        layers.LSTM(64)
    )(signals)
    
    repeater = layers.RepeatVector(t_out)(encoder)

    decoder1 = layers.LSTM(128, return_sequences = True)(repeater)
    decoder2 = layers.Bidirectional(
        layers.LSTM(64, return_sequences = True)
    )(decoder1)
    decoder3 = layers.Bidirectional(
        layers.LSTM(64, return_sequences = True)
    )(decoder2)

    regressor = layers.Dense(n_c_max)(decoder3)
    cleaner = Combiner()([regressor, labels])

    model = keras.Model(input_layer, cleaner)
    model.compile(loss = 'mse', optimizer = 'adam')
    
if architecture == 2:
    
    input_layer = layers.Input((t_in, n_c_max + 1))
    signals, labels = Splitter()(input_layer)

    encoder1 = layers.Bidirectional(
        layers.LSTM(64, dropout = 0.3, return_sequences = True),
    )(signals)
    
    encoder2 = layers.Bidirectional(
        layers.LSTM(64, dropout = 0.3, return_sequences = True),
    )(encoder1)
    
    encoder3 = layers.Bidirectional(
        layers.LSTM(64, dropout = 0.3, return_sequences = True),
    )(encoder2)
    
    slicer = layers.Lambda(lambda inputs: inputs[:, -t_out:, :])(encoder3)
    
    regressor = layers.Dense(n_c_max)(slicer)
    cleaner = Combiner()([regressor, labels])

    model = keras.Model(input_layer, cleaner)
    model.compile(loss = 'mse', optimizer = 'adam')
    
if architecture == 3:
    
    input_layer = layers.Input((t_in, n_c_max + 1))
    signals, labels = Splitter()(input_layer)

    encoder = layers.Bidirectional(
        layers.LSTM(128, dropout = 0.3, return_sequences = True),
    )(signals)
    
    slicer = layers.Lambda(lambda inputs: inputs[:, -t_out:, :])(encoder)
    
    decoder1 = layers.Bidirectional(
        layers.LSTM(128, dropout = 0.3, return_sequences = True)
    )(slicer)
    decoder2 = layers.Bidirectional(
        layers.LSTM(128, dropout = 0.3, return_sequences = True)
    )(decoder1)

    regressor = layers.Dense(n_c_max)(decoder2)
    cleaner = Combiner()([regressor, labels])

    model = keras.Model(input_layer, cleaner)
    model.compile(loss = 'mse', optimizer = 'adam')

# Training part 1

batch_size      = 1024
steps_per_epoch = 100
epochs          = 100

validation_batch_size     = batch_size
vaidation_steps_per_epoch = 30
# Necessary for function
validation_epochs         = 1

hist1 = model.fit(
    # Training data and configuration
    x = data_generator(
        batch_size, 
        steps_per_epoch, 
        epochs
    ),
    batch_size      = batch_size,
    steps_per_epoch = steps_per_epoch,
    epochs          = epochs,
    
    # Validation data
    validation_data = tf.data.Dataset.from_generator(
        lambda: data_generator(
            validation_batch_size, 
            vaidation_steps_per_epoch, 
            validation_epochs,
            mode = 'val'
        ),
        output_signature = (
            tf.TensorSpec(shape = (validation_batch_size, t_in , n_c_max + 1), dtype = tf.float64),
            tf.TensorSpec(shape = (validation_batch_size, t_out, n_c_max),     dtype = tf.float64),
        )
    ),
    validation_batch_size = validation_batch_size,
    validation_steps      = vaidation_steps_per_epoch,
    
    callbacks = [keras.callbacks.EarlyStopping(
        patience = 2,
        min_delta = 1e-3
    )]
)

# Training part 2

batch_size      = 128
steps_per_epoch = 300
epochs          = 100

hist2 = model.fit(
    # Training data and configuration
    x = data_generator(
        batch_size, 
        steps_per_epoch, 
        epochs
    ),
    batch_size      = batch_size,
    steps_per_epoch = steps_per_epoch,
    epochs          = epochs,
    
    # Validation data
    validation_data = tf.data.Dataset.from_generator(
        lambda: data_generator(
            validation_batch_size, 
            vaidation_steps_per_epoch, 
            validation_epochs,
            mode = 'val'
        ),
        output_signature = (
            tf.TensorSpec(shape = (validation_batch_size, t_in , n_c_max + 1), dtype = tf.float64),
            tf.TensorSpec(shape = (validation_batch_size, t_out, n_c_max),     dtype = tf.float64),
        )
    ),
    validation_batch_size = validation_batch_size,
    validation_steps      = vaidation_steps_per_epoch,
    
    callbacks = [keras.callbacks.EarlyStopping(
        patience = 3,
        min_delta = 5e-4
    )]
)

# Testing

mse = model.evaluate(
    x = data_generator(
        1024,
        1000,
        1,
        mode = 'test'
    )
)
    
# Saving

with open(f'results_batch_ms_mc/res-{t_in}-{t_out}-{architecture}.pickle', 'wb') as fp:
    pickle.dump({
        'hists': [hist1.history, hist2.history],
        'mse': mse
    }, fp)
    

print(f'\n\n\n{t_in} - {t_out} - {architecture}\n\n\n')