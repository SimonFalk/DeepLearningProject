## Electrocardiogram generation with a bidirectional LstM-CNN generative adversarial network
### Fei Zhu et.al.

#### Model architecture
- GAN
- Generator: 
	- Two layers of bidirectional LSTM each having 100 cells
	- Dropout layer
	- Fully connected layer
- Discriminator: CNN: 2x convolutional-pooling layers

#### Data
- ECG (electrocardiogram) recordings 
- 31.2 million points in total
- MIT-BIH arrythmia dataset (seems to be accessible)
https://www.physionet.org/content/mitdb/1.0.0/

#### Method
- Training using sequences of 3120 points
- Generation of new data, compare to real with PRD (see paper eq. 23), RMSE, and Frechet distance

#### Results
- Much smaller loss values and faster converging when comparing GAN to other recurrent networks
- Creates synthetic ECG that "matches the real data distribution"

#### Tricks that they use
- Dropout layer ("To prevent slow gradient descent due to parameter inflation in the generator, we add a dropout layer and set the probability to 0.5")
- Different loss function (see the paper)
- Frechet distance ("The Fréchet distance (FD) is a measure of similarity between curves that takes into consideration the location and ordering of points along the curves, especially in the case of time series data.")

#### Why do they say like that
- "The output layer is a two-dimensional vector where the first element represents the time step and the second element denotes the lead.""

