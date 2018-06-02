import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing

def computegrad(beta, x, y, lambda_val):
    
    """
    Calculates the gradient of L2-regularized logistic regression
    Inputs:
      - beta: Coefficient matrix (d x 1)
      - x: Matrix of samples (n x d)
      - y: Sample labels (n x 1)
      - lambda_val: Regularization parameter (constant)
    Output:
      - grad: Matrix of gradients for each beta (d x 1)
    """
    
    n, d = x.shape
    p_n  = np.exp(-y*(x.dot(beta)))
    P = np.identity(n) - np.diagflat(1-(p_n/(1+p_n)))
    grad = -1/n * x.T.dot(P.dot(y)) + 2 * lambda_val * beta
    return grad

def obj(beta, x, y, lambda_val):
    
    """
    Calculates the objective value of L2-regularized logistic regression
    Inputs:
      - beta: Coefficient matrix (d x 1)
      - x: Matrix of samples (n x d)
      - y: Sample labels (n x 1)
      - lambda_val: Regularization parameter (constant)
    Output:
      - objective: The objective value (value)
    """
    
    n, d = x.shape
    objective = 1/n * sum(np.log(1 + np.exp(-y*(x.dot(beta))))) + lambda_val * sum(beta**2)
    return objective

def backtracking(beta_init, x, y, lambda_val, n, alpha = 0.5, beta = 0.5, max_iter = 200):
    
    """
    Calculates the next step size
    Inputs:
      - beta_init: Coefficient matrix (d x 1)
      - x: Matrix of samples (n x d)
      - y: Sample labels (n x 1)
      - lambda_val: Regularization parameter (constant)
      - n_init: Initial step size (constant)
      - alpha: Tuning parameter for objective function comparison (constant)
      - beta: Tuning parameter on how much to decrease n with each iteration (constant)
      - max_iters: Maximum iterations to find new step size (constant)
    Output:
      - n: New step size (value)
    """
    
    grad = computegrad(beta_init, x, y, lambda_val)
    norm_grad = np.linalg.norm(grad)
    found_n = False
    
    i = 0
    while (found_n is False and i < max_iter):     
        if (obj(beta_init - n * grad, x, y, lambda_val) < obj(beta_init, x, y, lambda_val) - alpha * n * norm_grad ** 2):
            found_n = True
        elif i == max_iter - 1:
            raise Exception('Max iterations reached in backtracking.')
        else:
            n *= beta
            i += 1
    return n

def graddescent(beta_init, x, y, lambda_val, n_init, eps = 0.001, max_iter = 1000):
    
    """
    Run gradient descent with backtracking
    Inputs:
      - beta_init: Initial coefficient matrix (d x 1)
      - x: Matrix of samples (n x d)
      - y: Sample labels (n x 1)
      - lambda_val: Regularization parameter (constant)
      - n_init: Initial step size (constant)
      - eps: Convergence criterion for the the norm of the gradient (constant)
      - max_iter: Maximum number of iterations (constant)
    Output:
      - beta_vals: An array of coefficient matrices after each iteration
      - obj_vals: An array of coefficient matrices after each iteration
    """
    
    beta = beta_init
    grad = computegrad(beta, x, y, lambda_val)
    beta_vals = [beta]
    obj_vals = [obj(beta, x, y, lambda_val)]
    n = n_init
    iter = 0
    while np.linalg.norm(grad) > eps:
        n = backtracking(beta, x, y, lambda_val, n)
        beta = beta - n * grad
        beta_vals.append(beta)
        obj_vals.append(obj(beta, x, y, lambda_val))
        grad = computegrad(beta, x, y, lambda_val)
        iter += 1
    return np.array(beta_vals), np.array(obj_vals)

def fastgradalgo(beta_init, x, y, lambda_val, n_init, eps = 0.001, max_iter = 1000):
                                         
    """
    Run fast gradient descent with backtracking
    Inputs:
      - beta_init: Initial coefficient matrix (d x 1)
      - x: Matrix of samples (n x d)
      - y: Sample labels (n x 1)
      - lambda_val: Regularization parameter (constant)
      - n_init: Initial step size (constant)
      - eps: Convergence criterion for the the norm of the gradient (constant)
      - max_iter: Maximum number of iterations (constant)
    Output:
      - beta_vals: An array of coefficient matrices after each iteration
      - obj_vals: An array of coefficient matrices after each iteration
    """                                 
                                         
    beta = beta_init
    theta = np.zeros(len(beta_init))
    grad = computegrad(theta, x, y, lambda_val)
    beta_vals = [beta]
    obj_vals = [obj(beta, x, y, lambda_val)]
    n = n_init
    iter = 0
    while np.linalg.norm(grad) > eps and iter < max_iter:
        n = backtracking(beta, x, y, lambda_val, n)
        beta_new = theta - n * grad
        theta = beta_new + (iter/(iter + 3)) * (beta_new - beta)
        beta = beta_new
        beta_vals.append(beta)
        obj_vals.append(obj(beta, x, y, lambda_val))
        grad = computegrad(theta, x, y, lambda_val)
        iter += 1
    return np.array(beta_vals), np.array(obj_vals)

def step_init(x, lambda_val):

    """
    Calculates initial step size for L2-regularized logistic regression
    Inputs:
      - x: Matrix of samples (n x d)
      - lambda_val: Regularization parameter (constant)
    Output:
      - step size: step_size value (value)
    """
    
    n, d = x.shape
    L = max(np.linalg.eigvals(np.dot(x.T, x)))/n + lambda_val
    return 1/(L)

def ConvertLambda(x, lambda_val):
    """
    Converts the regularization parameter, λ,of my L2-regularized logistic regression to scikit-learn's C
    Inputs:
      - x: Matrix of samples (n x d)
      - lambda_val: Regularization parameter (constant)
    Output:
      - C: Scikit-learn's regularization parameter (constant)
    """
    
    n, d = x.shape
    C = 1/(2 * lambda_val * n)
    return C

def misclassification(betas, x, y):
    
    """
    Calculates misclassification error over an array of coefficient matrices
    Inputs:
      - betas: An array of coefficent matrices (d x 1)
      - x: Matrix of samples (n x d)
      - y: Sample labels (n x 1)
    Output:
      - result: List of misclassification erros for each coefficent matrix (list)
    """
        
    result = []
    n, d = betas.shape
    for i in range(0, n):
        predicted = list(x.dot(betas[i]))
        y = list(y)
        m = []
        for j in range(0, len(predicted)):
            if (predicted[j] < 0 and y[j] == 1) or (predicted[j] >= 0 and y[j] == -1):
                m.append(1)
            else:
                m.append(0)
        result.append(sum(m)/len(m))
    return result

def plotresults(plot1, lambda_val, plottitle = '', label1 = '', label2 = '', plot_2 = False, plot2 = None):
    
    """
    Plots the objective or misclassification error vs. iterations
    Inputs:
      - plot1: set of values
      - plot2: set of values if overlaying two results
      - lambda_val: Regularization parameter (constant)
    Output:
      - plots: displays plot
    """
    
    plt.plot(plot1, color = 'blue', label = label1 + ' λ=' + str(lambda_val))
    if plot_2 == True:
        plt.plot(plot2, color = 'red', label = label2 + ' λ=' + str(lambda_val))
    plt.title(plottitle)
    plt.legend(loc = 'upper right')
    plt.show()