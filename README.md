# hgrlab
Hand Gesture Recognition experiments.

## Requirements

We suggest using a virtual environment to install the necessary dependencies and
run the code. To install the required package dependencies using `pip`, please
run the following command:

```pip install -r requirements.txt```

Please note that currently, this project does not offer an installation package.
Therefore, all commands must be executed from the project root directory
`hgrlab`.

## Experiment delello_lnlm2024

The source code of the experiment `delello_lnlm2024` allows the reproducibility
of the results presented in the paper titled *"Comparison of sEMG-based Hand
Gesture Classifiers"* authored by Guilherme C. De Lello, Gabriel S. Chaves,
Juliano F. Caldeira, and Markus V.S. Lima.

The experiment aims to analyze the effect of different classifiers on hand
gesture classification. The study considers five supervised learning
classifiers, namely support vector machine, logistic regression, linear
discriminant analysis, *k*-nearest neighbors, and decision tree.

For this experiment, the model `hgr_dtw` was used, which is based on the HGR
system described in the paper *"Real-Time Hand Gesture Recognition Based on
Artificial Feed-Forward Neural Networks and EMG"* by Benalcázar et al. The
*k*-fold cross-validation method was employed to estimate the threshold that
optimizes individual accuracy for all subjects, instead of using a fixed muscle
detection threshold. This process is explained in the paper *"Hand Gesture
Classification using sEMG Data: Combining Gesture Detection and
Cross-Validation"* by Chaves et al., published at the conference CBIC 2023.

To run this experiment, please follow these steps:

1. Download the project source code
2. Open your terminal application
3. Set the current directory to the project root directory `hgrlab`
4. Run the command `python -m hgrlab.experiments.delello_lnlm2024`

This program will perform the following tasks:

1. Download the HGR datasets used in the experiment, including data from 10
subjects
2. Experiment 1: Determine the best segmentation thresholds for each of the 10
subjects and 5 classifiers using *4*-fold cross-validation
3. Experiment 2: Compare the accuracy of the HGR system for each of the 5
classifiers
