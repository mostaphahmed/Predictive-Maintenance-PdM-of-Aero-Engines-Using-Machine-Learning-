# Predictive-Maintenance-PdM-of-Aero-Engines-Using-Machine-Learning-

This repository showcases a PdM system that utilizes NN to predict Aero Engine RUL. The implementation emphasizes improving accuracy, reducing complexity, and enhancing efficiency through statistical data preprocessing in a data-driven approach. To achieve this, various Neural Network models are employed.


This work draws inspiration from prior contributions by mohyunho, available at https://github.com/mohyunho/N-CMAPSS_DL/blob/main/README.md.

You can prepare training and test sample arrays for machine learning models, particularly for deep learning architectures that accept time-windowed data, using NASA's N-CMAPSS dataset. Follow the steps below:

Download the Turbofan Engine Degradation Simulation Data Set-2, known as the N-CMAPSS dataset [2], from NASA's prognostic data repository, available at https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

Once you have the dataset downloaded, focus on dataset DS02, as it is used for data-driven prognostics, which is what we need.

Locate the file named "N-CMAPSS_DS02-006.h5" and place it in the /N-CMAPSS folder.

Now, you can generate npz files for each of the nine engines by executing the provided Python code. These npz files will serve as your training and test sample arrays for the machine learning model, especially useful for deep learning architectures that require time-windowed data as input.

By following these instructions, you'll have the necessary data ready for your machine learning model, allowing you to perform tasks like model-based diagnostics and data-driven prognostics on the Turbofan Engine Degradation Simulation Data Set-2.
