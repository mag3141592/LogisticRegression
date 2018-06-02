# Logistic Regression
Code implements L2-Regularized Logistic Regression.

## Functions
L2RegularizedLogisticRegression.py contains all of the source code.

## Data
RealDemo.py downloads the spam dataset from https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data, with the train-test split determined by https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.traintest. This demo predicts whether an email is spam or not, as well as compares gradient descent to fast gradient descent.

SimulatedDemo.py illustrates logistic regression on a simulated dataset. 

Users can use their own data, but labels should be scaled to -1 or +1.

## Sklearn
ScikitLearnComparison.py compares the results of my L2-regularized logistic regression with scikit-learn's.

## Needed Packages
  > import random
  > import numpy as np
  > import pandas as pd
  > import matplotlib.pyplot as plt 
  > import sklearn import preprocessing
