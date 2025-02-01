# HGRLab

HGRLab: Open-Source Toolkit for Hand Gesture Recognition

## Requirements

We suggest using a virtual environment to install the necessary dependencies and
run the code. To install the required package dependencies using `pip`, please
run the following command:

`pip install -r requirements.txt`

Please note that currently, this project does not offer an installation package.
Therefore, all commands must be executed from the project root directory
`hgrlab`.

## Experiment delello_lnlm2024

### Publication

[Comparison of sEMG-based Hand Gesture Classifiers](https://doi.org/10.21528/lnlm-vol22-no2-art4)

### Description

The source code of the experiment `delello_lnlm2024` allows the reproducibility
of the results presented in the paper entitled _"Comparison of sEMG-based Hand
Gesture Classifiers"_ authored by Guilherme C. De Lello, Gabriel S. Chaves,
Juliano F. Caldeira, and Markus V.S. Lima.

The experiment aims to analyze the effect of different classifiers on hand
gesture classification. The study considers five supervised learning
classifiers, namely support vector machine, logistic regression, linear
discriminant analysis, _k_-nearest neighbors, and decision tree.

For this experiment, the model `hgr_dtw` was used, whose architecture is based
on the HGR system described in the paper _[Real-Time Hand Gesture Recognition
Based on Artificial Feed-Forward Neural Networks and EMG](https://doi.org/10.23919/EUSIPCO.2018.8553126)_
by Benalcázar et al.

The _k_-fold cross-validation method was employed to estimate the threshold that
optimizes individual accuracy for all subjects, instead of using a fixed muscle
detection threshold. This process is explained in the paper _[Hand Gesture
Classification using sEMG Data: Combining Gesture Detection and
Cross-Validation](https://doi.org/10.21528/CBIC2023-029)_ by Chaves et al.

### Instructions

To run this experiment, please follow these steps:

1. Download the project source code
2. Open your terminal application
3. Make sure the required packages are installed
4. Set the current directory to the project root directory `hgrlab`
5. Run the command `python -m hgrlab.experiments.delello_lnlm2024`

This program performs the following tasks:

- Download the HGR datasets used in the experiment, including sEMG data from 10
  subjects
- Experiment 1: Determine the best segmentation thresholds for each of the 10
  subjects and 5 classifiers using _4_-fold cross-validation (Table 1: _Optimum
  individual segmentation thresholds using 4-fold cross-validation_)
- Experiment 2: Compare the accuracy of the HGR system for each of the 5
  classifiers (Table 2: _Mean accuracy and standard deviation of the HGR systems
  using different classifiers_ and Table 3: _Mean accuracy and standard deviation
  by subject for different classifiers_)

## Citation

If you use _HGRLab_ in a scientific publication, we would appreciate citations
to the following paper:

[Comparison of sEMG-based Hand Gesture Classifiers](https://doi.org/10.21528/lnlm-vol22-no2-art4),
De Lello et al., _Learning and Nonlinear Models_, vol. 22, no. 2, pp. 48–61,
October 2024.

Bibtex entry:

    @article{de_lello_comparison_2024,
        title = {Comparison of {SEMG}-{Based} {Hand} {Gesture} {Classifiers}},
        author = {De Lello, G. C. and Chaves, G. S.
                 and Caldeira, J. F. and Lima, M. V. S.},
        journal = {Learning and Nonlinear Models},
        volume = {22},
        number = {2},
        pages = {48--61},
        year = {2024},
        month = oct,
        doi = {10.21528/lnlm-vol22-no2-art4},
    }
