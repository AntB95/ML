#-----------------------------------------------------------------------------------#
#                                       ANNEXE
#-----------------------------------------------------------------------------------#
import numpy as np 
import matplotlib.pyplot as plt
import proj1_helpers
from implementations import *


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split(data_set, indice,index):
    """
    data_set: dataset you want to split based on index of the column
    indice: value of the feature you want (0,1,2,3)
    index: index of the column you want to use 
    return: subdataset subset based on the index feature value selected
    """
    y_set = data_set[0]
    x_set = data_set[1]
    id_set = data_set[2]
    return (y_set[x_set[:,index] == indice],x_set[x_set[:,index] == indice],id_set[x_set[:,index] == indice])


def replace_mean(x_set):
    """
    x_set: dataset you want to change to replace -999 
    return: dataset with the mean of the column instead of -999
    """

    x_set[x_set == -999] = np.nan
    list_mean = x_set.mean(axis = 0)
    for i in range(0,len(list_mean)):
        x_set[np.isnan(x_set[:,i])] = list_mean[i]
    return x_set

def get_na_columns(array, threshold, value):
    """
    array: dataset you want to cheak
    threshold: % limit of na by column
    value: value you want to check in the dataset
    return: list of columns with more value than the threshold
    """
    na_indices = []
    for ind, row in enumerate(array.T):
        count_na = 0
        for j in range(len(row)):
            if row[j] == value:
                count_na += 1
        if (count_na/len(row)) > threshold:
            na_indices.append(ind)
    return na_indices

def standardize(x_train, x_test):
    """
    x_train: train set you want to standardize
    x_test: test set you want to standardize
    return train and test set standardize
    """

    mean = np.mean(x_train)
    norm = np.linalg.norm(x_train)
    x_train_std = (x_train - mean)/norm
    x_test_std = (x_test - mean)/norm
    return x_train_std, x_test_std

def sigmoid(x):
    """
    x: matrix on you want to apply the sigmoid function 
    return: matrix with the sigmoid function apply on it
    """
    return 0.5 * (1 + np.tanh(0.5*x))

def zero_to_neg(array):
    """
    array: matrix on which you want to replace 0 by -1
    return matrix with 0 replace by -1
    """

    ret = np.zeros(len(array))
    for i, v in enumerate(array):
        if v == 0:
            ret[i] = -1
        else:
            ret[i] = v
    return ret

def build_poly(x, degree):
    """
    x: matrix on which we wish to build the polynomial degree
    degree: the polynomial degree built
    return: a matrix having the polynomial degree built
    """

    poly = x
    for deg in range(2, degree+1):
        poly = np.concatenate((poly, np.power(x, deg)), axis = 1)
    return poly

def add_function(x):
    """
    x: matrix on which we wish we want to add sin function 
    return: matrix having sin function add
    """

    return np.concatenate((x,np.sin(x)), axis = 1)
