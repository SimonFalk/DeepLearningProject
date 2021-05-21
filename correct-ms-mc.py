import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_wrangling.datamanager import DataLoader as DL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal as sig
import sys
import pickle
from welford import Welford

t_in  = int(sys.argv[1])
t_out = int(sys.argv[2])
architecture = int(sys.argv[3])

print(f'\n\n\n{t_in} - {t_out} - {architecture}\n\n\n')

# 	depth_pre 	width_pre 	width_mid 	depth_post 	width_post 	dropout 	mses
#10 		2 	64 			128 		2 			64 			0.0 		0.094752


nperseg = 100
noverlap = 99
nhop = nperseg - noverlap
logfft = 8


dl = DL("../")
sessions = ['s1', 's4', 's5', 's6', 's7', 's10']
data = dl.get_fcx2(sessions)


def spectrogram_base(
        signal,
        nperseg = 100,
        noverlap = 99,
        nfft = 2 ** 8,
        f_lo=10,
        f_hi=100,
        window = 'boxcar'
    ):
        freqs, times, spect = sig.spectrogram(
            signal,
            1000,
            window,
            nperseg = nperseg,
            noverlap = noverlap,
            nfft = nfft,
            scaling = 'spectrum'
        )
        freqs_idxs = (freqs >= f_lo) & (freqs <= f_hi)
        spect = spect[freqs_idxs].T
        spect = np.sqrt(spect)

        return freqs, freqs_idxs, times, spect
    
    
def get_as_array(session):
    print("Retrieving data from session %s..."%session)
    spects = []
    arrs = data[session]['data']
    mn = arrs.mean(0).reshape((1, -1))
    st = arrs.std (0).reshape((1, -1))
    arrs -= mn
    arrs /= st
    print("...finished")
    return arrs


Xs = []
spect_params = []
for s in sessions:
    X = get_as_array(s)
    Xs.append(X)
    
    
n_channelss = [X.shape[1] for X in Xs]


data = 0
# Indexing/setup

train_frac = 0.5
val_frac   = 0.2

train_ranges = np.array([
    (
        0,
        int(X.shape[0] * train_frac) - t_in - t_out - nperseg
    )
    for X in Xs
])

val_ranges = np.array([
    (
        train_max + t_in + t_out, 
        train_max + int(X.shape[0] * val_frac) - t_in - t_out - nperseg
    )
    for (_, train_max), X in zip(train_ranges, Xs)
])

test_ranges = np.array([
    (
        val_max + t_in + t_out,
        len(X) - t_in - t_out - nperseg
    )
    for (val_min, val_max), X in zip(val_ranges, Xs)
])

np.concatenate([
    train_ranges,
    val_ranges,
    test_ranges
], 1)


n_sessions = len(sessions)

def get_random_data_idxs(sess_no, n, mode = 'train'):
    if mode == 'train':
        _idxs = np.arange(*list(train_ranges[sess_no].copy()))
    elif mode == 'val':
        _idxs = np.arange(*list(val_ranges[sess_no].copy()))
    elif mode == 'test':
        _idxs = np.arange(*list(test_ranges[sess_no].copy()))
        
    np.random.shuffle(_idxs)
    channels = np.random.choice(n_channelss[sess_no], size=n, replace=True)
    return np.array([
            _idxs[:n], 
            _idxs[:n] + t_in + t_out + nperseg - nhop, 
            channels
    ]).T


def data_generator(
    batch_size,
    steps_per_epoch,
    epochs,
    normalize = False,
    mode = 'train'
):
    
    for _ in range(steps_per_epoch * epochs):
        
        # Select session indices
        # number of samples for each session
        counts = pd.Series(
            np.random.randint(n_sessions, size = batch_size)
        ).value_counts()

        # Session-wise indices
        # [[start, end, chan] per session]
        idxs = [
            get_random_data_idxs(idx, n, mode = mode)
            for idx, n in enumerate(counts)
        ]
        
        
        samples = np.concatenate([
            np.array([
                X[lo:hi, chan] 
                for lo,hi,chan in idx
            ])
            for X, idx in zip(Xs, idxs)
        ])
        
        # Alt 1. False
        if normalize == False:
            samples = np.stack([
                spectrogram_base(
                    samp,
                    nperseg,
                    noverlap,
                    2**logfft
                )[3]
                for samp in samples
            ])
            
        # Alt 2. Mean, std tuple
        else:
            mn, st = normalize
            samples = (np.stack([
                spectrogram_base(
                    samp,
                    nperseg,
                    noverlap,
                    2**logfft
                )[3]
                for samp in samples
            ]) - mn) / st
        
        yield (samples[:, :t_in, :], samples[:, t_in:, :])
        
        
welford_from_generator = Welford()
w_epochs = 5
w_batch_size = 5000
# 5 * (t_in + t_out) * 5000 ~= 2.5M
for x, y in data_generator(w_batch_size, 1, w_epochs):
    for spects in x:
        welford_from_generator.add_all(spects)
    for spects in y:
        welford_from_generator.add_all(spects)
        
        
mn, st = welford_from_generator.mean.reshape((1, -1)), np.sqrt(welford_from_generator.var_p).reshape((1, -1))


#---------------------
#---------------------
#--ARCHITECTURES START

if architecture == 1:

    depth_pre = 2
    width_pre = 64
    width_mid = 128
    depth_post = 2
    width_post = 64
    dropout = 0.0

    # Architecture 1
    # Model building
    model = keras.Sequential([
        layers.Input((t_in, 23))
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
    model.add(layers.RepeatVector(t_out))
    model.add(layers.LSTM(
        width_mid, 
        return_sequences = True, 
        dropout = dropout
    ))

    # Post
    for _ in range(depth_post - 1):
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
    model.add(layers.Dense(23))
    model.compile(loss = 'mse', optimizer = 'adam')
    
elif architecture == 2:
    
    # ARCHITECTURE 2
    # 		depth_pre 	width_pre 	depth_post 	width_post 	dropout 	mses
    #16 	3 	 	 	64 	 	 	0 	 	 	64 	 	 	0.0 	 	0.097678

    depth_pre = 3
    width_pre = 64
    depth_post = 0
    width_post = 64
    dropout = 0.0


    # Model building
    model = keras.Sequential([
        layers.Input((t_in, 23))
    ])

    # Pre
    for _ in range(depth_pre):
        model.add(layers.Bidirectional(
            layers.LSTM(
                width_pre, 
                return_sequences = True, 
                dropout = dropout
            )
        ))

    # Mid
    model.add(layers.Lambda(lambda inputs: inputs[:, -t_out:, :]))

    # Post
    for _ in range(depth_post):
        model.add(layers.Dense(width_post))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(23))

    # Compile
    model.compile(loss = 'mse', optimizer = 'adam')
    
elif architecture == 3:
    # ARCHITECTURE 3
    # 		depth_pre 	width_pre 	depth_post 	width_post 	dropout 	mses
    # 10 	2 	 	 	64 	 	 	2 	 	 	64 	 	 	0.0 	 	0.097303

    depth_pre = 2
    width_pre = 64
    depth_post = 2
    width_post = 64
    dropout = 0.0

    # Model building
    model = keras.Sequential([
        layers.Input((t_in, 23))
    ])

    # Pre
    for _ in range(depth_pre):
        model.add(layers.Bidirectional(
            layers.LSTM(
                width_pre, 
                return_sequences = True, 
                dropout = dropout
            )
        ))

    # Mid
    model.add(layers.Lambda(lambda inputs: inputs[:, -t_out:, :]))

    # Post
    for _ in range(depth_post):
        model.add(layers.Bidirectional(
            layers.LSTM(
                width_post, 
                return_sequences = True, 
                dropout = dropout
            )
        ))

    model.add(layers.Dense(23))

    # Compile
    model.compile(loss = 'mse', optimizer = 'adam')
    

#--ARCHITECTURES END
#-------------------
#-------------------



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
        epochs,
        normalize = (mn, st)
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
            mode = 'val',
            normalize = (mn, st)
        ),
        output_signature = (
            tf.TensorSpec(shape = (validation_batch_size, t_in , 23), dtype = tf.float64),
            tf.TensorSpec(shape = (validation_batch_size, t_out, 23), dtype = tf.float64),
        )
    ),
    validation_batch_size = validation_batch_size,
    validation_steps      = vaidation_steps_per_epoch,
    
    callbacks = [keras.callbacks.EarlyStopping(
        patience = 2,
        min_delta = 1e-3
    )]
)



batch_size      = 128
steps_per_epoch = 300
epochs          = 100

hist2 = model.fit(
    # Training data and configuration
    x = data_generator(
        batch_size, 
        steps_per_epoch, 
        epochs,
        normalize = (mn, st)
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
            mode = 'val',
            normalize = (mn, st)
        ),
        output_signature = (
            tf.TensorSpec(shape = (validation_batch_size, t_in , 23), dtype = tf.float64),
            tf.TensorSpec(shape = (validation_batch_size, t_out, 23),     dtype = tf.float64),
        )
    ),
    validation_batch_size = validation_batch_size,
    validation_steps      = vaidation_steps_per_epoch,
    
    callbacks = [keras.callbacks.EarlyStopping(
        patience = 3,
        min_delta = 5e-4
    )]
)

#Testing
mse = model.evaluate(
    x = data_generator(
        1024,
        1000,
        1,
        mode = 'test',
        normalize = (mn, st)
    )
)

with open('res-{}-{}-{}.pickle'.format(t_in, t_out, architecture), 'wb') as fp:
    pickle.dump({
        'hists': [hist1.history, hist2.history],
        'mse': mse
    }, fp)

print(f'\n\n\n{t_in} - {t_out} - {architecture}\n\n\n')