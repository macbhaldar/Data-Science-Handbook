import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input, ReLU
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read the data with pandas library
train = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

# Train data
print("Train shape: ", train.shape)
train

# Test data
print("Test shape: ", test.shape)
test

x_train = train.drop(labels = ["label"], axis = 1)
y_train = train["label"]

x_test = test.drop(labels = ["label"], axis = 1)
y_test = test["label"]

x_train
y_train

k = 0
row, col = 3, 3
fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(16,20),)
for i in range(row):
    for j in range(col):
        img = x_train.iloc[k].to_numpy()
        img = img.reshape((28,28))
        ax[i,j].imshow(img,cmap = "gray")
        ax[i,j].axis("off")
        k += 1
plt.show()

x_train = x_train / 255.0 # Normalization
x_train

k = 0
row, col = 3, 3
fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(16,20),)
for i in range(row):
    for j in range(col):
        img = x_train.iloc[k].to_numpy()
        img = img.reshape((28,28))
        ax[i,j].imshow(img,cmap = "gray")
        ax[i,j].axis("off")
        k += 1
plt.show()

x_train = x_train.to_numpy()
x_train = x_train * 2 - 1
print("x_train shape: ", x_train.shape)
x_train

def create_generator():
    
    generator = Sequential()
    generator.add(Dense(units = 256, input_dim = 100))
    generator.add(ReLU())
    
    generator.add(Dense(units = 512))
    generator.add(ReLU())
    
    generator.add(Dense(units = 1024))
    generator.add(ReLU())
    
    generator.add(Dense(units = 784, activation = "tanh"))
    
    generator.compile(loss = "binary_crossentropy",
                     optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5))
    
    return generator

# Structure of Generator model
g = create_generator()
g.summary()

# Create Discriminator 
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units = 1024, input_dim = 784))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(units = 512))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(units = 256))
    discriminator.add(ReLU())
    
    discriminator.add(Dense(units = 1, activation = "sigmoid"))
    
    discriminator.compile(loss = "binary_crossentropy",
                         optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5))
    
    return discriminator

# Structure of discriminator model
d = create_discriminator()
d.summary()

# Create GANs
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape = (100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs = gan_input, outputs = gan_output)
    gan.compile(loss = "binary_crossentropy", optimizer = "adam")
    return gan
# Structure of GANs model
gan = create_gan(d,g)
gan.summary()

import time

epochs = 250
batch_size = 128

dis_loss = []
gen_loss = []

for e in range(epochs):
    for _ in range(batch_size):
        start = time.time()
        noise = np.random.normal(0, 1, [batch_size, 100])
        
        generated_image = g.predict(noise)
        
        image_batch = x_train[np.random.randint(low = 0, high = x_train.shape[0], size = batch_size)]
        
        x = np.concatenate([image_batch, generated_image])
        
        y_dis = np.zeros(batch_size*2)
        y_dis[:batch_size] = 0.9
        
        d.trainable = True
        dloss = d.train_on_batch(x, y_dis)
        
        
        noise = np.random.normal(0, 1, [batch_size, 100])
        
        y_gen = np.ones(batch_size)
        
        d.trainable = False
        
        gloss =  gan.train_on_batch(noise, y_gen)
        
        end = time.time()
        process_time = str(end - start)
        
    dis_loss.append(dloss)
    gen_loss.append(gloss)

    print("Epoch: {}, Time: {}s, Generator Loss: {:.3f}, Discriminator Loss: {:.3f}".format(e, process_time[2:4], gloss, dloss))

g.save_weights("gans_model.h5")

# Visualizetion Result Of GANs
noise = np.random.normal(loc = 0, scale = 1, size = [100,100])
generated_image = g.predict(noise)
generated_image = generated_image.reshape(100, 28, 28)
plt.figure(figsize=(15,17))
for i in range(0,10):
    plt.subplot(1, 10, i+1)
    plt.imshow(generated_image[i], interpolation = "nearest", cmap = "gray")
    plt.axis("off")
    plt.tight_layout()
plt.show()

# Train and Test Discriminator Loss graphic
epochs_number = []
for i in range(0,epochs):
    epochs_number.append(i)

plt.plot(epochs_number, dis_loss)
plt.title("Train - Discriminator Loss")
plt.xlabel("Number of Epochs")
plt.xlabel("Discriminator Loss")
plt.legend()
plt.show()

# Train and Test Generator Loss graphic
plt.plot(epochs_number, gen_loss)
plt.title("Train - Generator Loss")
plt.xlabel("Number of Epochs")
plt.xlabel("Generator Loss")
plt.legend()
plt.show()

