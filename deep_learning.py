import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import keras
from keras.layers import Dense
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.regularizers import l1
import matplotlib.pyplot as plt

#--------------------------------------   parameters   ---------------------------------------
kfold_value = 5  # number of k for kfold CV
epochs_value = 50  # number of epochs
internal_neurons_number_1 = 14  # number of neurons and internal nodes for first level
internal_neurons_number_2 = 13  # number of neurons and internal nodes for second level
min_MAE_distance = 0.0005  # early stopping when MAE(n)-MAE(n-1) < min_MAE_distance
learning_rate_value = 0.05  # value for lr parameter
momentum_value = 0.6  # value for momentum parameter
regularizer_value = 0  # value for regularizer

#-------------------------------------   Read dataset   --------------------------------------
dataset = pd.read_csv('dataset/ml-100k/u.data', header=None, delimiter="\t", names=['userId', 'itemId', 'rating', 'timestamp'])
dataset = dataset.apply(pd.to_numeric)

#------------------------------------   data_centering   -------------------------------------
average_user_rating = {}
for userId in dataset.userId.unique():
    average_user_rating[userId] = dataset.loc[dataset['userId'] == userId, 'rating'].mean()  # centering
    dataset.loc[dataset['userId'] == userId, 'rating'] -= average_user_rating[userId]
# print(max(dataset.loc[:, 'rating']))
# print(min(dataset.loc[:, 'rating']))
#------------------------------------   missing values   -------------------------------------
# NxM_dataset: NxM Array with user ratings and 0 for missing values
NxM_dataset = pd.pivot_table(dataset, index='userId', columns='itemId', values='rating', fill_value=0)
NxM_dataset = NxM_dataset.apply(pd.to_numeric)

#----------------------------------   data_normalization  ------------------------------------
for userId in dataset.userId.unique():
    NxM_dataset.loc[userId, :] = NxM_dataset.loc[userId, :] / max(abs(NxM_dataset.loc[userId, :]))

#-------------------------------   5-fold_cross_validation   --------------------------------
X = np.array(pd.get_dummies(dataset.userId.unique()))  # input vector
Y = np.array(NxM_dataset.loc[:, :])  #outputvector

kf = KFold(n_splits=kfold_value, shuffle=True)

#--------------------------------------   A-NN_model   --------------------------------------
rmseList = []  # list for RSME metric
maeList = []  # list for MAE metric

fold = 0
for (train_index, test_index) in kf.split(X):
    fold += 1
    model = keras.Sequential()

    model.add(Dense(len(X), activation="sigmoid",  activity_regularizer=l1(regularizer_value), input_dim=len(X)))  # input level
    model.add(Dense(internal_neurons_number_1, activation="relu", activity_regularizer=l1(regularizer_value), input_dim=len(X)))  # internal level 1
    model.add(Dense(internal_neurons_number_2, activation="relu", activity_regularizer=l1(regularizer_value), input_dim=internal_neurons_number_1))  # internal level 2
    model.add(Dense(len(Y[0]), activation="sigmoid",  input_dim=internal_neurons_number_2))  # output level

    # rmse calculation function
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    # mae calculation function
    def mae(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true))

    keras.optimizers.SGD(lr=learning_rate_value, momentum=momentum_value, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=[rmse, mae])

    early_stopping = EarlyStopping(monitor='mae', mode='min', min_delta=min_MAE_distance, verbose=1)
    history = model.fit(X[train_index], Y[train_index], validation_data=(X[test_index], Y[test_index]), epochs=epochs_value, batch_size=len(X), callbacks=[early_stopping], verbose=0)

    scores = model.evaluate(X[test_index], Y[test_index], verbose=0)
    rmseList.append(scores[1])
    maeList.append(scores[2])

    print("Fold :", fold, " RMSE:", scores[1], " MAE:", scores[2])

print("RMSE: ", np.mean(rmseList), "MAE: ", np.mean(maeList))

#----------------------------------------   plot   ----------------------------------------
history_dict = history.history
history_dict.keys()

val_loss = history_dict['val_loss']
loss = history_dict['loss']
rmse = history_dict['rmse']
mae = history_dict['mae']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.plot(epochs, rmse, label='rmse')
plt.plot(epochs, mae, label='mae')
plt.title('Metrics Graph')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()
