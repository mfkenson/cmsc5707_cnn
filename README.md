# CMSC5707 Advanced Topics in Artificial Intelligence

## Rapid prototyping audio classification solution using CNN (AlexNet)

### Abstract

This tutorial covers a rapid prototyping implementation of audio classification in Python using CNN with mel-scaled spectrogram plotted images on UrbanSound8K dataset with up to 90% accuracy. The development time of this prototype took only a few hours and there are less than 100 lines of source code.

In this course, students are equipped with knowledge in audio feature extraction and hands-on experience on audio classification techniques using distortion scores/dynamic programming (Assignment 1) and LSTM neural network (Assignment 3). As we know that CNN is good at dealing with image classification problems, I decided to make an attempt on audio classification using CNN with mel-scaled spectrogram images.

Rapid prototyping encourages leveraging existing available resources. Therefore, instead of crafting my own CNN network structures, well-known CNN architecture (e.g. AlexNet) is evaluated. The results are surprisingly good (~90% validation accuracy) without explicitly tuning parameters at feature extraction stage or hyper-parameters tuning at CNN training stage.

### Dataset preparation
UrbanSound8K dataset, contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes.

The dataset will later be transformed to mel-scaled spectrogram images in PNG format. 80% of data is used for CNN training while the rest is used for validation. The filename of transformed images would be the same as original wav files except that the file extensions .wav would be replaced with .png. We should notice that the second part of the filename (splitted by ’-’) indicates the classID.

The dataset can be obtained from https://urbansounddataset.weebly.com/urbansound8k.html.

Due to the large size of the original dataset (6GB), I prepared my preprocessed mel-scaled spectrogram images and made available on the GitHub repository: https://github.com/mfkenson/cmsc5707_cnn

### Technology stack

#### fast.ai
fast.ai is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.

This library is chosen instead of plain pytorch/keras/tensorflow approaches because it offers numbers of wrappers and utility methods to facilitate the needs of rapid prototyping while keeping the compatibility and extensibility.

#### librosa
librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

This python package is responsible for loading raw wav audio files and extracting the mel-scaled spectrogram feature.

#### scikit-image
scikit-image is a collection of algorithms for image processing.

This python package is responsible for saving the preprocessed 8-bit numpy array in PNG image format.

### Environment Setup

This tutorial is successfully tested on following environments
* Python3.6 on ubuntu 18.04 with Nvidia GTX1650 CUDA10.2
* Google colab platform (Jupyter Notebook with GPU)

All packages can be installed using the well-known pip package installer for Python.
It is also recommended to be installed with anaconda virtual environment.

```
pip install fastai librosa scikit-image juypter
```

Environment with GPU support is optional but highly preferred.
Jupyter notebook is recommended for data visualization including the confusion matrix plot.


### References
1. https://urbansounddataset.weebly.com/download-urbansound8k.html
2. https://www.fast.ai/about/
3. https://librosa.org/doc/latest/index.html
4. https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
5. http://soundbible.com/2215-Labrador-Barking-Dog.html
6. http://soundbible.com/2078-Gun-Battle-Sound.html
7. https://stackoverflow.com/questions/56719138/how-can-i-save-a-librosa-spectrogram-plot-as-a-specific-sized-image
