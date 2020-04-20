import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import keras
from keras.layers import Dense
from keras import backend as K
import matplotlib.pyplot as plt

######################################   Read dataset   ######################################
dataset = pd.read_csv('dataset/ml-100k/u.data', header=None, delimiter="\t", names=['userId', 'itemId', 'rating', 'timestamp'])
dataset = dataset.apply(pd.to_numeric)

#####################################   data_centering   #####################################
average_user_rating = {}
for userId in dataset.userId.unique():
    average_user_rating[userId] = dataset.loc[dataset['userId'] == userId, 'rating'].mean()  # centering
    dataset.loc[dataset['userId'] == userId, 'rating'] -= average_user_rating[userId]

# print(dataset)
#####################################   missing values   #####################################
# NxM_dataset: NxM Array with user ratings and 0 for missing values
NxM_dataset = pd.pivot_table(dataset, index='userId', columns='itemId', values='rating', fill_value=0)
NxM_dataset = NxM_dataset.apply(pd.to_numeric)

###################################   data_normalization  ####################################
for userId in dataset.userId.unique():
    NxM_dataset.loc[userId, :] = NxM_dataset.loc[userId, :] / max(abs(NxM_dataset.loc[userId, :]))

################################   5-fold_cross_validation   #################################
X = np.array(pd.get_dummies(dataset.userId.unique()))
Y = np.array(NxM_dataset.loc[:, :])
kf = KFold(n_splits=5, shuffle=True)

#######################################   A-NN_model   #######################################
rmseList = []
maeList = []
accList = []
fold = 0
for (train_index, test_index) in kf.split(X):
    fold += 1
    model = keras.Sequential()

    model.add(Dense(943, activation="sigmoid", input_dim=943))
    model.add(Dense(12, activation="relu", input_dim=943))
    model.add(Dense(1682, activation="linear", input_dim=12))
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def mae(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true))

    keras.optimizers.SGD(lr=0.001, momentum=0.2, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[rmse, mae, 'accuracy'])
    history = model.fit(X[train_index], Y[train_index], epochs=50, batch_size=len(X), verbose=0)
    scores = model.evaluate(X[test_index], Y[test_index], verbose=0)
    rmseList.append(scores[1])
    maeList.append(scores[2])
    accList.append(scores[3])
    print("Fold :", fold, " RMSE:", scores[1], " MAE:", scores[2])

print("RMSE: ", np.mean(rmseList), "MAE: ", np.mean(maeList), "Acc: ", np.mean(accList))
# print("score", history.history)

##########################################   plot   ##########################################
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
loss = history_dict['loss']
rmse = history_dict['rmse']
mae = history_dict['mae']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='accuracy')
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, rmse, label='rmse')
plt.plot(epochs, mae, label='mae')
plt.title('Metrics Graph')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()
