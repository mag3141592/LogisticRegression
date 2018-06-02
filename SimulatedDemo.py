
# coding: utf-8

# In[1]:


# Import packages
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import L2RegularizedLogisticRegression as mylogreg

# Simulate data
random.seed(0)
x1 = np.random.normal(size = 10000)          
x2 = np.random.normal(size = 10000)
x3 = np.random.normal(size = 10000)
x4 = np.random.normal(size = 10000)
x5 = np.random.normal(size = 10000)
z = x1 + 2 * x2 * 7 * x3 + 3 * x4 + 8 * x5

inv_logit = 1/(1 + np.exp(-z)) 
y = np.array(np.random.binomial(n = 1, p = inv_logit, size = 10000))
x = [x1, x2, x3, x4, x5]
x = np.array(x).T

X_Train = x[0:8000]
X_Test = x[8000:]

Y_Train = y[0:8000]
Y_Test = y[8000:]

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

# Plots Objective vs. Iterations for fast gradient descent
mylogreg.plotresults(plot1 = obj_fgd, lambda_val = lambda_val, plottitle = 'Objective vs. Iterations', label1 = 'fastgradalgo:')

# Calculate the training set misclassification over iterations
mis_fgd_train = mylogreg.misclassification(betas_fgd, X_Train, Y_Train)

# Calculate the testing set misclassification over iterations
mis_fgd_test = mylogreg.misclassification(betas_fgd, X_Test, Y_Test)

#Create a dataFrame
data = {'Algorithm':['Fast Grad'],
        'Train Misclassification': [mis_fgd_train[-1]],
        'Test Misclassification':[mis_fgd_test[-1]]}
df = pd.DataFrame(data)
print(df)

# Plots Misclassification Error vs. Iterations for fast gradient descent
mylogreg.plotresults(plot1 = mis_fgd_train, lambda_val = lambda_val, plottitle = 'Train Misclassification vs. Iterations', label1 = 'fastgradalgo:')
mylogreg.plotresults(plot1 = mis_fgd_test, lambda_val = lambda_val, plottitle = 'Test Misclassification vs. Iterations', label1 = 'fastgradalgo:')

