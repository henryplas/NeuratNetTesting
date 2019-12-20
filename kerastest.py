"""
Dipping my feet into tensorflow, so to say. I wrote this set of code
to test my knowledge of NN to solve some basic and advanced datasets.
This program was designed to compete in a kaggle competition between my classmates and I. 

Go to https://www.kaggle.com/c/appstate-cs-01/leaderboard#score
for (appstate-cs-01 - appstate-cs-10) to see the results.
"""

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def keras_mlp_model_fit(x, y, layer_widths, alpha, n_iter, eta):
	model_list = []

	for i in range(1,len(layer_widths)):
		if(i == len(layer_widths) - 1):
			model_list.append(keras.layers.Dense(layer_widths[i], 
				activation='softmax', 
				kernel_regularizer=keras.regularizers.l2(alpha),
                activity_regularizer=keras.regularizers.l2(alpha),
                bias_regularizer=keras.regularizers.l2(alpha)
                ))

		else:
			model_list.append(keras.layers.Dense(layer_widths[i], 
				activation='tanh', 
				kernel_regularizer=keras.regularizers.l2(alpha),
                activity_regularizer=keras.regularizers.l2(alpha),
                bias_regularizer=keras.regularizers.l2(alpha)
                ))

	model = keras.Sequential(model_list)

	adam = keras.optimizers.Adam(lr=eta, beta_1=0.9, beta_2=0.999, epsilon=1e-8) #keras.optimizers.SGD(lr=eta)
	model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              )
	model.fit(x, y, epochs=n_iter, batch_size= len(x))

	return model

def first_kaggle(x,y):
	alpha = 0.01
	eta = 0.01
	n_iter = 50000
	layer_widths = [2, 1, 2]
	model = keras_mlp_model_fit(x, y, layer_widths, alpha, n_iter, eta)
	return model

def second_kaggle(x,y):
	alpha = 0.001
	eta = 0.0001
	n_iter = 50000
	layer_widths = [2,2,2]
	model = keras_mlp_model_fit(x, y, layer_widths, alpha, n_iter, eta)
	return model

def third_kaggle(x,y):
	alpha = 0.001
	eta = 0.0001
	n_iter = 50000
	layer_widths = [2,2,2]
	model = keras_mlp_model_fit(x, y, layer_widths, alpha, n_iter, eta)
	return model
def fourth_kaggle(x,y):
	alpha = 0.001
	eta = 0.0001
	n_iter = 50000
	layer_widths = [2,1,2]
	model = keras_mlp_model_fit(x, y, layer_widths, alpha, n_iter, eta)
	return model
def fifth_kaggle(x,y):
	alpha = 0
	eta = 0.001
	n_iter = 50000
	layer_widths = [2,11,2]
	model = keras_mlp_model_fit(x, y, layer_widths, alpha, n_iter, eta)
	return model

def sixth_kaggle(x,y):
	alpha = 0.00
	eta = 0.001
	n_iter = 50000
	layer_widths = [2,12,12,12,2]
	model = keras_mlp_model_fit(x,y, layer_widths, alpha, n_iter, eta)
	return model
def seventh_kaggle(x,y):
	alpha = 0.001
	eta = 0.000001
	n_iter = 50000
	layer_widths = [2,20,10,10,2]
	model = keras_mlp_model_fit(x,y, layer_widths, alpha, n_iter, eta)
	return model
def eighth_kaggle(x,y):
	alpha = 0.0
	eta = 0.00001
	n_iter = 10000
	layer_widths = [2,20,10,10,2]
	model = keras_mlp_model_fit(x,y, layer_widths, alpha, n_iter, eta)
	return model
def ninth_kaggle(x,y):
	alpha = 0.001
	eta = 0.000001
	n_iter = 50000
	layer_widths = [2,20,10,10,2]
	model = keras_mlp_model_fit(x,y, layer_widths, alpha, n_iter, eta)
	return model
def tenth_kaggle(x,y):
	alpha = 0.0001
	eta = 0.00001
	n_iter = 20000
	layer_widths = [2,20,10,10,2]
	model = keras_mlp_model_fit(x,y, layer_widths, alpha, n_iter, eta)
	return model

#import data
train_file = 'kaggle_07_train.csv'
df = pd.read_csv(train_file)
y_train = df['y'].values
x_train = df.values[:,1:-1]

#feature reduction -- not necessary 
#i tried this is the jupyter file, takes a long time
#and I had no success
# cols = df.columns
# corr = df.corr()
# corr = corr.iloc[:,:].values
# columnsNotNeeded = []

# rang = np.arange(len(corr))
# yy , xx = np.meshgrid(rang,rang)
# nothing = np.where(abs(corr[cols[xx]][cols[yy]]) >= 0.7 and xx != yy and xx not in columnsNotNeeded, columnsNotNeeded.append(cols[yy], 0))


# for i in range(len(corr)):
# 	for j in range(len(corr[cols[i]])):
# 		if abs(corr[cols[i]][cols[j]]) >= 0.7 and i != j:
# 			if(i not in columnsNotNeeded):
# 				#print(cols[i], " and ", cols[j], " have corr = ", corr[cols[i]][cols[j]])
# 				columnsNotNeeded.append(cols[j])

# columnsNotNeeded = set(columnsNotNeeded)
# len(columnsNotNeeded)



#split up training/test
np.random.seed(1)
p = np.random.permutation(len(x_train))
x = x_train[p]
y = y_train[p]



nr_training = int(len(x) * 0.8)

x_training = x[:nr_training, :]

y_training = y[:nr_training]
x_test = x[nr_training:, :]
y_test = y[nr_training:]


#set up keras_mlp_model_fit
model = seventh_kaggle(x_training, y_training)
y_hat = model.predict(x_test)
y_hat = y_hat.argmax(axis=1)

#Find accuracy
acc = np.where(y_test == y_hat, 1, 0).mean()
print(acc)

# Create a submission file
test_file = 'kaggle_07_test.csv'
df = pd.read_csv(test_file)
x_test = df.values[:,1:]

# model is your classifier (any object with a fit and predict method)
y_hat = model.predict(x_test)
y_hat = y_hat.argmax(axis=1)
#move data to csv
submission_file = 'kaggle_07_submission.csv'
df = pd.read_csv(submission_file)
df['Prediction'] = y_hat
df.to_csv('my_submission.csv', index=False)