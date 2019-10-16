import numpy as np
from proj1_helpers import *
from annexe import *
# X = tx w = beta_hat 
def least_squares(y, tx):
    A = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(A, b)
    loss = MSE(y, tx, w)
    return w, loss

# the mean squares error.
def MSE(y, tx, w):
    y_hat = tx.dot(w)
    e = y - y_hat
    n = len(e)
    mse = e.dot(e) / (2 * n)
    return mse

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n in range(0,max_iters):
        grad = gradient(y, tx, w)
        loss = MSE(y, tx, w)
        w = w - gamma*grad
    return w, loss

def gradient(y, tx, w):
    e = y - tx@w
    n = len(y)
    grad = -tx.T@e/n
    return grad

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):    
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            grad = gradient(minibatch_y, minibatch_tx, w)
            loss = MSE(minibatch_y, minibatch_tx, w)
            w = w - gamma*grad
    return w, loss

def ridge_regression(y, tx, lamb):
    n = len(y)
    lamb = lamb/(2*n)
    a = tx.T.dot(tx) + lamb * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = MSE(y, tx, w)
    return w, loss

def loss_function(h, y):
    """likelihood function loss"""
    return -y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
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
    """Regularized logistic regression"""
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