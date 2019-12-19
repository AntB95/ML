import numpy as np
import pandas as pd
import random
import math


def get_users_ID(line):
    '''Get the user_ID in correct surprise format'''
    row, col = line.split("_") # split using _ as separator 
    return int(row.replace("r", "")) #keep only digits as userID

def get_movie_ID(line):
    '''get the movie_ID in the correct surprise format'''
    row, col = line.split("_")
    return int(col.replace("c", "")) #keep only digits as movieID

def df_to_surprise(data):
    '''Change the dataframe from the user_ID_movie_ID_rate format to the userID;itemID;rating format'''
    data['userID'] = data['Id'].apply(get_users_ID)
    data['itemID'] = data['Id'].apply(get_movie_ID)
    data = data.drop('Id', axis = 1)
    data = data.rename(columns = {'Prediction':'rating'})[['userID','itemID','rating']]
    return data

def global_mean(df): return df.rating.mean() # Return global dataset mean

    
def user_mean(df): return df.groupby('userID').rating.mean()
 #Return a serie and for each user its mean 


def movie_mean(df): return df.groupby('itemID').rating.mean()
 #Return a serie with for each movie its mean

def predict_user(id, user, mean): return user.get(id, mean)
 #user mean based on the value returned in user_mean


def predict_movie(id, movie, mean): return movie.get(id, mean)
 #movie mean based on the value returned in movie_mean
    