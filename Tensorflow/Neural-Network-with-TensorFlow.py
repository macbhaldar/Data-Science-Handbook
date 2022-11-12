# Importing the libraries
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Importing the data
## In this data x and y are the point of coordinates and the color feature is the target value 
## that was generated randomly which is in binary representing Red – 1 , Blue – 0.
df = pd.read_csv('data.txt')

# split the data into train and test set
train, test = train_test_split(
	df, test_size=0.2, random_state=42, shuffle=True)

# Constructing the input
x = np.column_stack((train.x.values, train.y.values))
y = train.color.values

# Building a model
## build a simple neural network to classify the color of the point with two input nodes 
## and a hidden layer and an output layer with relu and sigmoid activation functions

# Defining the model
model = keras.Sequential([
	keras.layers.Dense(4, input_shape=(2,), activation='relu'),
	keras.layers.Dense(2, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam',
			loss=keras.losses.SparseCategoricalCrossentropy(),
			metrics=['accuracy'])

# fitting the model
model.fit(x, y, epochs=10, batch_size=8)

# Evaluating the model
x = np.column_stack((test.x.values, test.y.values))
y = test.color.values
model.evaluate(x, y, batch_size=8)

# Building a better model to improve the model with a few extra hidden layers
# Defining the model
model_better = keras.Sequential([
	keras.layers.Dense(16, input_shape=(2,), activation='relu'),
	keras.layers.Dense(32, activation='relu'),
	keras.layers.Dense(32, activation='relu'),
	keras.layers.Dense(2, activation='softmax')
])

# Compiling the model
model_better.compile(optimizer='adam',
					loss=keras.losses.SparseCategoricalCrossentropy(),
					metrics=['accuracy'])

# Constructing the input
x = np.column_stack((train.x.values, train.y.values))
y = train.color.values

# fitting the model
model_better.fit(x, y, epochs=10, batch_size=8)
