import numpy as np
from proj1_helpers import *

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
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
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