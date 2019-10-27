import numpy as np
from proj1_helpers import *
from annexe import *

def least_squares(y, tx):
    """
    y: the output of the data
    tx: the input of the data
    return: optimal weights and loss using the normal equations
    """

    A = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(A, b)
    loss = MSE(y, tx, w)
    return w, loss

def MSE(y, tx, w):
    """
    y: the output data
    tx: the input data
    w: the actual weight on which we will compute the loss
    return: the loss
    """

    y_hat = tx.dot(w)
    e = y - y_hat
    n = len(e)
    mse = e.dot(e) / (2 * n)
    return mse

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    y: the output data
    tx: the input data
    initial_w: the initial weight form which we wish to proceed the least squares GD algorithm
    max_iters: maximum iterations of the algorithm before returning the result
    gamma: the rate of the descent of the algorithm
    return: the optimal weights and its corresponding loss
    """

    w = initial_w
    for n in range(0,max_iters):
        grad = gradient(y, tx, w)
        loss = MSE(y, tx, w)
        w = w - gamma*grad
    return w, loss

def gradient(y, tx, w):
    """
    y: the output data
    tx: the input data
    w: the actual weight wishing to compute the gradient on
    return: the gradient
    """

    e = y - tx@w
    n = len(y)
    grad = -tx.T@e/n
    return grad

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    y: the output data
    tx: the input data
    initial_w: the initial weight form which we wish to proceed the SGD algorithm
    max_iters: maximum iterations of the algorithm before returning the result
    gamma: the rate of the descent of the algorithm
    return: the optimal weights and its corresponding loss
    """

    w = initial_w
    for n_iter in range(max_iters):    
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            grad = gradient(minibatch_y, minibatch_tx, w)
            loss = MSE(minibatch_y, minibatch_tx, w)
            w = w - gamma*grad
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    y: the input data
    tx: the output data
    lambda_: the penalizing parameter for the ridge regression
    return: optimal weights and loss for the ridge regression
    """

    n = len(y)
    lambda_ = lambda_/(2*n)
    a = tx.T.dot(tx) + lambda_ * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = MSE(y, tx, w)
    return w, loss

def loss_function(h, y):
    """likelihood function loss"""

    return -y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    y: the output data
    tx: the input data
    initial_w: the desired initial weight to begin the algorithm of logistic regression
    max_iters: the maximum of iterations during the algorithm
    gamma: the rate of descent of the gradient
    return: optimal weights and its corresponding loss for logistic regression
    """

    w = initial_w
    h = 0
    for i in range(0,max_iters):
        h = sigmoid(np.dot(tx, w))
        grad = tx.T.dot(h-y)
        w = w - gamma*grad
    h = sigmoid(tx.dot(w))
    loss = loss_function(h, y)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    y: the output data
    tx: the input data
    lambda_: the parameter used for penalization for the logistic regression
    initial_w: the desired initial weight to begin the algorithm of reg logistic regression
    max_iters: the maximum of iterations during the algorithm
    gamma: the rate of descent of the gradient
    return: the optimal weights and its corresponding loss
    """

    w = initial_w
    h = 0
    for i in range(max_iters):
        z = np.dot(tx, w)
        h = sigmoid(z)
        gradient = tx.T.dot(h-y) + lambda_*w
        w = w - gamma*gradient
    h = sigmoid(tx.dot(w))
    loss = loss_function(h, y) + (1/2)*lambda_*np.dot(w.T,w)
    return w, loss 