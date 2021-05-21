# Deep learning Project
The project for the course Deep Learning in Data Science, DD2424 at KTH. 
By Simon Falk and Gustav Röhss. 

### Dataset source:
http://crcns.org/data-sets/fcx/fcx-2/about-fcx-2

### File structure

#### data_wrangling/datamanager.py
Code for easy dataset access.

#### figures
Currently only a single figure showcasing model architectures.

#### historical
Various things yet to be removed.

#### reading_material
Various texts we've read within the scope of this project.

#### results_most_batch_tests
Results from most batch tests. Subfolder organization recently put in place, all sub folders used to be in the root directory of the repository.

Interpreting batch test folder names:
<ul>
    <li>ms_mc: Multichannel, multisession. </li>
    <li>ms: Multichannel, single session. Misleading name.</li>
    <li>None of the above: Single session, single channel spectrogram.</li>
</ul>

#### spect_res
Results from most the batch tests concerned with spectrogram data. Histograms and mse values from the tests conducted in spect-ms-mc.py.

#### ae-multisession-multichannel.ipynb
Notebook for conducting multisession multichannel autoencoder experiments.

#### batch_test-n.py | n in {1, 2, 3}
Batch tests for architectures 1, 2, 3 for single session signal or single session single channel spectrogram.

#### batch-ms-mc.py
Batch tests for all architectures - only multichannel multisession signal.

#### multisession-multichannel.ipynb
Experimentation notebook for multisession multichannel signal prediction. Contains informative figures.

#### spect-ms-mc.py
Batch tests for all architectures - multichannel multisession (for spectrogram data, with sample estimates of mean and std).

### run-ms-mc.sh
Shell script for running the batch tests, with the prescribed parameter combinations.

#### spect-results.ipynb
Presents the results produced in the batch tests made on spectrogram data.
