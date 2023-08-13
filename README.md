# Predictive-Maintenance-PdM-of-Aero-Engines-Using-Machine-Learning-

This repository showcases a PdM system that utilizes NN to predict Aero Engine RUL. The implementation emphasizes improving accuracy, reducing complexity, and enhancing efficiency through statistical data preprocessing in a data-driven approach. To achieve this, various Neural Network models are employed.


This work draws inspiration from prior contributions by mohyunho, available at https://github.com/mohyunho/N-CMAPSS_DL/tree/main

# N-CMAPSS Data Processing Instructions
This repository contains code for processing the NASA N-CMAPSS dataset to create training and test sample arrays for machine learning models, particularly those using deep learning architectures that accept time-windowed data. Below are the instructions for using the code:

# Download N-CMAPSS Dataset:

Download the Turbofan Engine Degradation Simulation Data Set-2, known as the N-CMAPSS dataset, from NASA's prognostic data repository.avaliable at https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

# Select Dataset:

The dataset DS02 is used for data-driven prognostics, which is what we need. Make sure to focus on this dataset for the processing.

# Data-Exploration:
 Data exploration for DS02 avaliable at https://github.com/mohyunho/N-CMAPSS_DL/tree/main/Dataset_exploration

# Locate Dataset File:

Place the file named "N-CMAPSS_DS02-006.h5" in the /N-CMAPSS folder of this repository.
# Configure Parameters:

Open the relevant Python script and set the following parameters:

w: Window length (time window size).

s: Stride of the window.

test: Choose "0" to extract samples from the engines used for training; otherwise, set it to create samples from test engines.

sampling: Subsample the data before creating the output array to mitigate memory issues. Adjust the sampling rate as needed (e.g., "10" for a 0.1Hz sampling rate).

# Data Type and Subsampling (Optional):

By default, the data type is set to 'np.float32' to reduce data size and avoid excessive memory use. If necessary, you can modify the data type in the 'data_preparation_unit.py' file located in the /utils folder.
Subsampling is available to handle potential 'out-of-memory' issues resulting from the original dataset's 1Hz sampling rate.

# Data Sampling Methods for Effective Model Learning

For our dataset DS02, we divide it into subgroups based on engine unit and Remaining Useful Life (RUL). This approach allows us to extract explicit outcomes and ensures adequate representation of each subgroup in the sample. Stratified sampling is particularly beneficial when working with varying numbers of RUL data points for each engine unit.

```
py sample_creator_strata_unit_auto.py -w 50 -s 1 --test 0 --sampling 10
```
Execute the Python code to generate npz files for each of the engines based on the configured parameters. Sampling files based on stratification method for the 9 Unit Engines 

# Generated Files:

After running the code, you will find nine npz files stored in the /N-CMAPSS/Samples_whole folder. Each compressed file contains two arrays: 'sample' and 'label'.
For test units, 'label' indicates the ground truth Remaining Useful Life (RUL) of the test engines for evaluation.


# NN Model 
Run the file inference_cnn_aggr_strata_unit_based.py with entering the arguments of the CNN/NN


# Applying PCA 

PCA is conducted to DS02 data-set reduce the dimensionality of extensive datasets. But make sure you choose the right file from the utils in the sample_creator_unit_auto.py

```
py sample_creator_strata_unit_based_PCA.py  -w 50 -s 1 --test 0 --sampling 10
```
Note: When running the above line you have to choose which method in sample_creater_strata_unit_based_auto.py 

# References

[1] Chao, Manuel Arias, Chetan Kulkarni, Kai Goebel, and Olga Fink. "Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics." Data. 2021; 6(1):5. https://doi.org/10.3390/data6010005

[2] Chao, Manuel Arias, Chetan Kulkarni, Kai Goebel, and Olga Fink. "Fusing physics-based and deep learning models for prognostics." Reliability Engineering & System Safety 217 (2022): 107961.



