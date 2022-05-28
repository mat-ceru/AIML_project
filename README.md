# AIML_project
Repository of a curricular machine learning project 
included in the university course 
*Artificial intelligence and machine learning*.

## Introduction
The repository contains the code useful to the 
definition of an artificial intelligence 
algorithm, based on neural networks, able to
understand the emotions expressed in audio files, 
using their spectrogram. 
Different datasets were used for
the networks' training part and the testing one: 
for the training audio of actors acting a 
specific emotion; for
the testing, pieces of songs manually 
classified according to the felt emotions. 
The neural networks'
architectures used are: GoogLeNet, VGG and ResNet. 
The Google Colab platform was used for the
analyses, developing the scripts in Python 
code using the Pytorch libraries.

## Repository structure
In the root are contained two different files:
- *final_report.pdf*: the paper describing the project, the analysis
    and the results
- *project_proposal.pdf*: the accepted proposal of the initial project design.

The useful code is contained in directory *code*
and *models*. In the first there are utility scripts,
meaning prepared to the definition of datasets and the 
testing of models. Instead, in the second one are contained
the notebook for training phase for each selected 
NN model, with the output's reports also.

In the directory *CAL500* are contained 
the information about the testing dataset, including
the files with the classification labels.
In the internal subfolders are contained the
results of different
preprocessing phases: original dataset (*CAL500_32kps_complete*),
filtered dataset (*CAL500_songs_selected*), entire extracted spectrograms
(*CAL500_test_entire_spectrograms*) and sliced spectrograms
(*CAL500_test_sliced_spectrograms*).

Instead, in the directory *RAVDESS_dataset* are contained7
the information about the training dataset.
The internal structure is equivalent to the *CAL500*
directory: four subfolders, each for the results of following 
preprocessing phases.
