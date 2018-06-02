# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import L2RegularizedLogisticRegression as mylogreg
from sklearn.linear_model import LogisticRegression

# Download data
spam = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data', sep = ' ', header = None)
train_test_split = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.traintest', sep=' ', header = None)

X = np.array(spam.drop(57, axis = 1))
Y = np.array(spam[57] * 2 - 1) #Scale Y to +/- 1

# Split data into the train and test set
train_test_indicator = train_test_split[0].T
train = np.where(train_test_indicator == 1)[0]
test = np.where(train_test_indicator == 0)[0]

X_Train = X[train, :]
X_Test = X[test, :]

Y_Train = Y[train]
Y_Test = Y[test]

# Standardize the data
scaler = preprocessing.StandardScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)

lambda_val = 0.1

# Calculate initial step size
step_size = mylogreg.step_init(X_Train, lambda_val)
print('Initial Step Size:', step_size)

# Initialize coefficients
d = X_Train.shape[1]
beta_init = np.zeros(d)

# Performs fast gradient descent using default maximum iterations and convergance stopping criteria
betas_fgd, obj_fgd = mylogreg.fastgradalgo(beta_init = beta_init, x = X_Train, y = Y_Train, lambda_val = lambda_val, n_init = 1)

beta_T = betas_fgd[-1]

# Convert my regularization parameter to scikit-learn's
c = mylogreg.ConvertLambda(X_Train, lambda_val)

# Fit Sklearn's Logistic Regression
reg = LogisticRegression(C = c)
reg.fit(X_Train, Y_Train)
beta_star = reg.coef_

# Create a dataFrame
data = {'Beta_*': beta_star[0],
        'Beta_T': beta_T}
df = pd.DataFrame(data)

# Prints the final coefficients
print(df)

# Prints the final objective value
print("Objectives:", "Beta_T = ", obj_fgd[-1], "Beta_* = ", mylogreg.obj(beta_star[0], X_Train, Y_Train, lambda_val))
