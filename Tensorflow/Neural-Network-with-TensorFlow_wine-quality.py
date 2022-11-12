# Neural-Network-with-TensorFlow_Wine-quality

# import the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# import the datasets
df = pd.read_csv('winequality-red.csv')
df.head()

# 75% of the data is selected
train_df = df.sample(frac=0.75, random_state=4)

# it drops the training data from the original dataframe
val_df = df.drop(train_df.index)

# calling to (0,1) range
max_val = train_df.max(axis= 0)
min_val = train_df.min(axis= 0)

range = max_val - min_val
train_df = (train_df - min_val)/(range)

val_df = (val_df- min_val)/range

# separate the targets and labels
X_train = train_df.drop('quality',axis=1)
X_val = val_df.drop('quality',axis=1)
y_train = train_df['quality']
y_val = val_df['quality']

# We'll need to pass the shape
# of features/inputs as an argument
# in our model, so let's define a variable
# to save it.
input_shape = [X_train.shape[1]]
input_shape

# Create a linear Model
model = tf.keras.Sequential([
tf.keras.layers.Dense(units=1,input_shape=input_shape)])
model.summary()

# Creating a Multilayered Neural Network
model = tf.keras.Sequential([
	tf.keras.layers.Dense(units=64, activation='relu',
						input_shape=input_shape),
	tf.keras.layers.Dense(units=64, activation='relu'),
	tf.keras.layers.Dense(units=1)
])
model.summary()

# Adam optimizer works pretty well for all kinds of problems and is a good starting point
model.compile(optimizer='adam', loss='mae') # MAE error is good for numerical predictions

# Training The Model
losses = model.fit(X_train, y_train,
				validation_data=(X_val, y_val),
				batch_size=256,
				epochs=15, 
				)

# Predict the ‘wine quality’ and Analyze Accuracy
model.predict(X_val.iloc[0:3, :])

y_val.iloc[0:3] # compare our predictions with the target value

# analyze the loss and figure out if it is overfitting
loss_df = pd.DataFrame(losses.history)
loss_df.loc[:,['loss','val_loss']].plot()
