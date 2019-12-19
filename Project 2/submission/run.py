#!conda install -c conda-forge scikit-surprise

#import package 
from surprise import SVD, NMF, Dataset, Reader, SVDpp, BaselineOnly, KNNBaseline, SlopeOne, accuracy
from surprise.model_selection import cross_validate, GridSearchCV,train_test_split, KFold, GridSearchCV
from sklearn.model_selection import KFold as skFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, Ridge
import pandas as pd 
from project_helpers import *
from math import *

#seed
random.seed(404)
np.random.seed(404)

#Load the train set 
train = pd.read_csv('data_train.csv')
train = df_to_surprise(train)
#Convert the train set into Surprise
reader = Reader(rating_scale=(1, 5))
train_surp = Dataset.load_from_df(train, reader)
train_surp = train_surp.build_full_trainset()
train_surp_test = train_surp.build_testset()

mean = global_mean(train)
users = user_mean(train)
movies = movie_mean(train)

#we train our different model on the train set with the parmaeters we found with the GS
bsl_options = {'method': 'sgd','reg': 10**-11}
bsl_options_knnu = {'method': 'als','n_epochs': 50,}
sim_options_knnu = {'name': 'pearson_baseline', 'user_based' : True}
bsl_options_knni = {'method': 'als','n_epochs': 50,}
sim_options_knni = {'name': 'pearson_baseline', 'user_based' : False}

algo_baseline = BaselineOnly(bsl_options = bsl_options).fit(train_surp)
algo_slope_one = SlopeOne().fit(train_surp)
algo_knn_user = KNNBaseline(k = 400, sim_options = sim_options_knnu, bsl_options = bsl_options_knnu).fit(train_surp)
algo_knn_movie = KNNBaseline(k = 200, sim_options = sim_options_knni, bsl_options = bsl_options_knni).fit(train_surp)

#loading the test set 
test_copy = pd.read_csv('sampleSubmission.csv')
#convert test set to the surprise format
test = test_copy.copy()
test = df_to_surprise(test)
test = Dataset.load_from_df(test, reader)
test = test.build_full_trainset()
test = test.build_testset()

#NMF
algo_NMFb = NMF(n_epochs = 400, biased = True).fit(train_surp)
predictions_NMFb = algo_NMFb.test(test)
est_NMFb = [pred.est for pred in predictions_NMFb]

algo_SVDb = SVD(n_factors = 400, lr_all = 0.0015, biased = True, reg_all = 0.1, n_epochs = 400, random_state = 200).fit(train_surp)
predictions_SVDb = algo_SVDb.test(test)
est_SVDb = [pred.est for pred in predictions_SVDb]

#prediction for each model
predictions_baseline = algo_baseline.test(test)
predictions_SVDb = algo_SVDb.test(test)
predictions_slope_one = algo_slope_one.test(test)
predictions_knn_user = algo_knn_user.test(test)
predictions_knn_movie = algo_knn_movie.test(test)
predictions_NMFb = algo_NMFb.test(test)

#Extract estimated ratings
uids = [pred.uid for pred in predictions_baseline]
mids = [pred.iid for pred in predictions_baseline]
ruis = [pred.r_ui for pred in predictions_baseline]
est_baseline = [pred.est for pred in predictions_baseline]
est_slope_one = [pred.est for pred in predictions_slope_one]
est_knn_user = [pred.est for pred in predictions_knn_user]
est_knn_movie = [pred.est for pred in predictions_knn_movie]
est_global = [mean for i in range(len(ruis))]
est_user_mean = [predict_user(u, users, mean) for u in uids]
est_movie_mean = [predict_movie(m, movies, mean) for m in mids]
est_NMFb = [pred.est for pred in predictions_NMFb]

#matrix containing ratings predictions for each model 
est_baseline = np.array(est_baseline)
est_global = np.array(est_global)
est_user_mean = np.array(est_user_mean)
est_movie_mean = np.array(est_movie_mean)
est_knn_movie = np.array(est_knn_movie)
est_knn_user = np.array(est_knn_user)
est_slope_one = np.array(est_slope_one)
est_SVDb = np.array(est_SVDb)

X = np.column_stack((est_global, est_user_mean, est_movie_mean, est_baseline, 
                     est_knn_movie, est_knn_user, est_slope_one,est_SVDb, est_NMFb))
#we constructe a linear combinaison of the 10 models we fit
#weights
weights = np.array([ 0.1459329 , -0.29250287, -0.22321729, -1.04077736,  0.33121226,
        0.40730471, -0.18782983,  1.39937115,  0.45781328]) #rmse = 1.017

preds = X.dot(weights)
#Clip interval to 1-5 and round predictions to nearest integer
preds = np.clip(preds, 1, 5)
preds = np.around(preds)
#Creating the id with the correct format
ids = np.array(['r'+str(u)+'_c'+str(m) for (u,m) in zip(uids, mids)])
#creating the submission
sub = pd.DataFrame({'Id':ids, 'Prediction':preds})
sub.to_csv('Submission.csv', index = False)