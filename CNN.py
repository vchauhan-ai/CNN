
import keras
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import math
import numpy as np
import csv
import pandas as pd


# code to download data from CIFAR-10
train_dataset= dsets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)

# extracting images of car from the entire dataset
cars=[]

for i in range(0,len(train_dataset.train_labels)):
    if train_dataset[i][1] == 1 :
        cars.append(train_dataset[i])


#display each image - data labelling
to_img = ToPILImage()
data= to_img(cars[1500][0])
data.show()


# data is created by storing each car type in the csv file
f = open('Data_Project_DL.csv', 'a')
f.write('Hatchback')
f.write('\n')
f.close()


data= pd.read_csv("Data_Project_DL.csv", header=None)
data= np.array(data)


for i in range(0, len(data)):
    if data[i]== 'Sedan':
        data[i]= 1
    elif data[i]== 'SUV':
        data[i]= 2
    elif data[i]== 'Hatchback':
        data[i]= 2
    elif data[i]== 'Convertible':
        data[i]= 2


# separating training, validation and test data from created dataset
X_train= cars[0:1000][0][0].numpy().reshape(1, 32, 32, 3)
X_val= cars[1001:1201][0][0].numpy().reshape(1,32,32,3)
X_test= cars[1202:1501][0][0]
Y_train= data[0:1000]
Y_train= np.array(Y_train)
#Y_train = keras.utils.to_categorical(np.ravel(Y_train))
Y_val= data[1001:1201]
#Y_val = keras.utils.to_categorical(np.ravel(Y_val))
Y_val= np.array(Y_val)
Y_test= data[1202:1501]
Y_test = np.array(Y_test)


X_train_1= []
for i in range(0, 1000):
    X_train_1.append(np.array(cars[i][0].numpy().reshape(1, 32, 32, 3)))


X_train_1= np.array(X_train_1).reshape(1000,32,32,3)
len(X_train_1)


X_val_1= []
for i in range(1001, 1201):
    X_val_1.append(np.array(cars[i][0].numpy().reshape(1, 32, 32, 3)))


X_val_1= np.array(X_val_1).reshape(200,32,32,3)
len(X_val_1)


X_train_1 = X_train_1.astype('float32')
X_val_1 = X_val_1.astype('float32')
X_train_1 /= 255
X_val_1 /= 255


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GaussianNoise
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


#Initialize CNN
classifier= Sequential()
#First Convolutional Layer
classifier.add(Convolution2D(512, 3, 3 , input_shape= (32, 32, 3), activation='relu'))
#Max Pool
classifier.add(MaxPooling2D(pool_size= (2,2)))
classifier.add(Dropout(0.25))
# Second Convolutional Layer
classifier.add(Convolution2D(1024, 3, 3, activation= 'relu'))
#Max pool
classifier.add(MaxPooling2D(pool_size= (2,2)))
classifier.add(Dropout(0.15))
#Third Conv
classifier.add(Convolution2D(2048, 3, 3, activation= 'relu'))
#Third Maxpool
classifier.add(MaxPooling2D(pool_size= (2,2)))
classifier.add(Dropout(0.10))
#Flatten
classifier.add(Flatten())

#Hidden_layer

classifier.add(Dense(output_dim=128, activation = 'relu'))
classifier.add(GaussianNoise(0.25))
classifier.add(Dense(output_dim=128, activation = 'relu'))
classifier.add(Dense(output_dim=128, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(GaussianNoise(0.25))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(GaussianNoise(0.25))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(GaussianNoise(0.25))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))
classifier.add(Dense(output_dim=256, activation = 'relu'))

#Outer_layer

classifier.add(Dense(output_dim=1, activation = 'sigmoid'))

#Compile

from keras import optimizers
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

opt = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(optimizer= opt , loss= 'hinge', metrics= ['accuracy'])

#Image Augmentation

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=True,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode= 'nearest')


datagen.fit(X_train_1)
classifier.fit_generator(datagen.flow(X_train_1, Y_train),
                    steps_per_epoch=len(X_train_1) / 128, epochs=1, validation_data=(X_val_1, Y_val),workers=4 )


def build_classifier(optimizer):

    from keras.models import Sequential
    from keras.layers import Convolution2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import GaussianNoise
    from sklearn.model_selection import GridSearchCV
    from keras.wrappers.scikit_learn import KerasClassifier


    classifier= Sequential()
    classifier.add(Convolution2D(512, 3, 3 , input_shape= (32, 32, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size= (2,2)))
    classifier.add(Dropout(0.25))
    classifier.add(Convolution2D(1024, 3, 3, activation= 'relu'))
    classifier.add(MaxPooling2D(pool_size= (2,2)))
    classifier.add(Dropout(0.15))
    classifier.add(Convolution2D(2048, 3, 3, activation= 'relu'))
    classifier.add(MaxPooling2D(pool_size= (2,2)))
    classifier.add(Dropout(0.10))
    classifier.add(Flatten())

    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(GaussianNoise(0.25))
    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(Dense(output_dim=256, activation = 'relu'))
    classifier.add(Dense(output_dim=1, activation = 'sigmoid'))
    classifier.compile(optimizer= opt , loss= 'hinge', metrics= ['accuracy'])
    return classifier

classifier= KerasClassifier(build_fn= build_classifier)

parameters= { 'batch_size' : [10, 16, 32],'epochs': [30, 100, 200],'optimizer': ['Adagrad', 'RMSprop', 'SGD', 'adam']}

grid_search= GridSearchCV(estimator= classifier, param_grid= parameters, scoring='accuracy', cv=100)

grid_search= grid_search.fit(X_train_1, Y_train)

best_parameters= grid_search.best_params_

best_accuracy= grid_search.best_score_
