## Tests on how far we can predict using spectrograms from one channel

### Data

I used 70% of the availiable session s1 data from fcx2 for training (only the channel indexed 0)
The division was made by reserving the first indices for training and the last for testing.
I constructed spectrograms using frequency resolution of 2^8, 100 timepoints for each segment
with a sampling frequency of 1000 Hz, which made each spectrum 23-dimensional.
I used a timeseries of 200 timepoints as input.

### Model

The second best performing model architecture of type 1 in the batch tests was used.
It had 128 BiLSTM cells in the first layer, 256 LSTM cells in the middle layer and
128 BiLSTM cells in the last layer. I used dropout for all layers with a probability of 0.3 of dropping nodes.

### Method

From the reserved training data 10 % was used as validation. Batch size was set to 2048.
I varied the number of timepoints which are fed into the network as true "labels", that is the timepoints
whose spectra we want to predict.
I used 10, 20, 50 and 100 timepoints in the output.
I trained the models for 10 epochs each.

### Results

The saved trained models can be found in the folder spect/models/ in the home directory of the VM jupyter lab.

#
