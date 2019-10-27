import numpy as np
from proj1_helpers import *
from implementations import *
from annexe import *

#import train and test set
train_set = load_csv_data('/Users/bedanian/Desktop/Machine Learning/Project 1/train.csv', sub_sample = True)
test_set = load_csv_data('/Users/bedanian/Desktop/Machine Learning/Project 1/test.csv', sub_sample = False)

#Preprocessing
#split the datset based on column 22 value
#train
(y_train_0,x_train_0,id_train_0) = split(train_set, 0, 22)
(y_train_1,x_train_1,id_train_1) = split(train_set, 1, 22)
(y_train_2,x_train_2,id_train_2) = split(train_set, 2, 22)
(y_train_3,x_train_3,id_train_3) = split(train_set, 3, 22)

#test
(y_test_0,x_test_0,id_test_0) = split(test_set, 0, 22)
(y_test_1,x_test_1,id_test_1) = split(test_set, 1, 22)
(y_test_2,x_test_2,id_test_2) = split(test_set, 2, 22)
(y_test_3,x_test_3,id_test_3) = split(test_set, 3, 22)

#delete column 22 from train and test set
x_train_0 = np.delete(x_train_0, [22] , axis = 1)
x_train_1 = np.delete(x_train_1, [22] , axis = 1)
x_train_2 = np.delete(x_train_2, [22] , axis = 1)
x_train_3 = np.delete(x_train_3, [22] , axis = 1)

x_test_0 = np.delete(x_test_0, [22] , axis = 1)
x_test_1 = np.delete(x_test_1, [22] , axis = 1)
x_test_2 = np.delete(x_test_2, [22] , axis = 1)
x_test_3 = np.delete(x_test_3, [22] , axis = 1)

#clean column of more than 90% of nan for all the 4 subset
nan_0 = get_na_columns(x_train_0, 0.90, -999)
x_train_0 = np.delete(x_train_0, nan_0 , axis = 1)
x_test_0 = np.delete(x_test_0, nan_0 , axis = 1)

nan_1 = get_na_columns(x_train_1, 0.90, -999)
x_train_1 = np.delete(x_train_1, nan_1 , axis = 1)
x_test_1 = np.delete(x_test_1, nan_1 , axis = 1)

nan_2 = get_na_columns(x_train_2, 0.90, -999)
x_train_2 = np.delete(x_train_2, nan_2, axis = 1)
x_test_2 = np.delete(x_test_2, nan_2, axis = 1)

nan_3 = get_na_columns(x_train_3, 0.90, -999)
x_train_3 = np.delete(x_train_3, nan_3, axis = 1)
x_test_3 = np.delete(x_test_3, nan_3, axis = 1)

#replace last nan by the mean
x_train_0 = replace_mean(x_train_0)
x_train_1 = replace_mean(x_train_1)
x_train_2 = replace_mean(x_train_2)
x_train_3 = replace_mean(x_train_3)

x_test_0 = replace_mean(x_test_0)
x_test_1 = replace_mean(x_test_1)
x_test_2 = replace_mean(x_test_2)
x_test_3 = replace_mean(x_test_3)

#subset 0 delete column 18 
x_train_0 = np.delete(x_train_0, [18] , axis = 1)
x_test_0 = np.delete(x_test_0, [18] , axis = 1)


#standardize train and test set
(x_train_0,x_test_0) = standardize(x_train_0,x_test_0)
(x_train_1,x_test_1) = standardize(x_train_1,x_test_1)
(x_train_2,x_test_2) = standardize(x_train_2,x_test_2)
(x_train_3,x_test_3) = standardize(x_train_3,x_test_3)

#parameter of the ridge regression
(l0,d0) = [1e-12, 2]
(l1,d1) = [4.691758698326445e-12, 2]
(l2,d2) = [4.210291410564796e-14, 2]
(l3,d3) = [0.00015581034821123352, 3]


#add polynome
x_train_0 = build_poly(x_train_0, d0)
x_train_1 = build_poly(x_train_1, d1)
x_train_2 = build_poly(x_train_2, d2)
x_train_3 = build_poly(x_train_3, d3)

x_test_0 = build_poly(x_test_0, d0)
x_test_1 = build_poly(x_test_1, d1)
x_test_2 = build_poly(x_test_2, d2)
x_test_3 = build_poly(x_test_3, d3)

#add functions sin 
x_train_0 = add_function(x_train_0)
x_test_0 = add_function(x_test_0)
x_train_1 = add_function(x_train_1)
x_test_1 = add_function(x_test_1)
x_train_2 = add_function(x_train_2)
x_test_2 = add_function(x_test_2)
x_train_3 = add_function(x_train_3)
x_test_3 = add_function(x_test_3)

#ridge regression on the 4 subsets
(w_0,_) = ridge_regression(y_train_0, x_train_0,l0)
(w_1,_) = ridge_regression(y_train_1, x_train_1,l1)
(w_2,_) = ridge_regression(y_train_2, x_train_2,l2)
(w_3,_) = ridge_regression(y_train_3, x_train_3,l3)

#apply sigmoid function and change output to -1,1
y_0 = zero_to_neg(np.around(sigmoid(x_test_0 @ w_0)))
y_1 = zero_to_neg(np.around(sigmoid(x_test_1 @ w_1)))
y_2 = zero_to_neg(np.around(sigmoid(x_test_2 @ w_2)))
y_3 = zero_to_neg(np.around(sigmoid(x_test_3 @ w_3)))

#concateante the 4 subsets
y = np.concatenate((y_0,y_1,y_2,y_3))
id_test = np.concatenate((id_test_0,id_test_1,id_test_2,id_test_3))

# create submission
create_csv_submission(id_test, y, 'prediction.csv')