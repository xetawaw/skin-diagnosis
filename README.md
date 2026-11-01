# skin-diagnosis
skin-diagnosis
# Skin Lesion Image Analysis

A research prototype using EfficientNet-B0 to classify dermatoscopic images from HAM10000.

## Live demo

[Open the Streamlit app](https://skin-diagnosis-7pqbz55xvfmkdrtbowlbsa.streamlit.app/)

## Results

Five-fold lesion-grouped validation:
- Mean Macro-F1: **0.7003**
- Mean balanced accuracy: **0.7649**

The validation folds were also used to select the best epoch, so these results may be optimistic.

## Disclaimer

Research use only. This application is not a medical diagnosis or a medical device.
## Dataset

This project uses the HAM10000 dataset:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The dataset is not included in this repository. To test the hosted app,
download a dermatoscopic JPEG from the dataset and upload it.
