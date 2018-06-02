# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import L2RegularizedLogisticRegression as mylogreg

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

# Performs both gradient descent and fast gradient descent, for comparision, using default maximum iterations and convergance stopping criteria
betas_gd, obj_gd = mylogreg.graddescent(beta_init = beta_init, x = X_Train, y = Y_Train, lambda_val = lambda_val, n_init = 1)
betas_fgd, obj_fgd = mylogreg.fastgradalgo(beta_init = beta_init, x = X_Train, y = Y_Train, lambda_val = lambda_val, n_init = 1)

# Plots Objective vs. Iterations for gradient descent and fast gradient descent
mylogreg.plotresults(plot1 = obj_gd, plot2 = obj_fgd, lambda_val = lambda_val, plottitle = 'Objective vs. Iterations', label1 = 'graddescent:', label2 = 'fastgradalgo:', plot_2 = True)

# Calculate the training set misclassification over iterations
mis_gd_train = mylogreg.misclassification(betas_gd, X_Train, Y_Train)
mis_fgd_train = mylogreg.misclassification(betas_fgd, X_Train, Y_Train)

# Calculate the testing set misclassification over iterations
mis_gd_test = mylogreg.misclassification(betas_gd, X_Test, Y_Test)
mis_fgd_test = mylogreg.misclassification(betas_fgd, X_Test, Y_Test)

# Create a dataFrame
data = {'Algorithm':['Grad', 'Fast Grad'],
        'Train Misclassification':[mis_gd_train[-1], mis_fgd_train[-1]],
        'Test Misclassification':[mis_gd_test[-1], mis_fgd_test[-1]]} 
df = pd.DataFrame(data)
print(df)

# Plots Misclassification Error vs. Iterations for gradient descent and fast gradient descent
mylogreg.plotresults(plot1 = mis_gd_train, plot2 = mis_fgd_train, lambda_val = lambda_val, plottitle = 'Train Misclassification vs. Iterations', label1 = 'graddescent:', label2 = 'fastgradalgo:', plot_2 = True)
mylogreg.plotresults(plot1 = mis_gd_test, plot2 = mis_fgd_test, lambda_val = lambda_val, plottitle = 'Test Misclassification vs. Iterations', label1 = 'graddescent:', label2 = 'fastgradalgo:', plot_2 = True)

