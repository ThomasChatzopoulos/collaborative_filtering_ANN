import numpy as np
import pandas as pd
from pandas import isnull

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras import backend as K

######################################   Read dataset   ######################################
dataset = pd.read_csv('dataset/ml-100k/u.data', header=None, delimiter="\t",
                      names=['userId', 'itemId', 'rating', 'timestamp'])
# print(dataset)
#####################################   data_centering   #####################################
average_user_rating = {}
for userId in dataset.userId.unique():
    average_user_rating[userId] = dataset.loc[dataset['userId'] == userId, 'rating'].mean()
    dataset.loc[dataset['userId'] == userId, 'rating'] -= average_user_rating[userId]
# print(dataset)

#####################################   missing values   #####################################
# NxM_dataset: NxM Array with user ratings and NaN values
NxM_dataset = pd.pivot_table(dataset, index='userId', columns='itemId', values='rating')
NxM_dataset[np.isnan(NxM_dataset)] = 0
print(NxM_dataset)

###################################   data_normalization   ###################################


