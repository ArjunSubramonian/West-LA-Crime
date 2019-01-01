import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam

import get_West_LA_crime_data as gcd

# Train a deep, fully-connected neural network that, given the month, the time of day, age, sex, descent, and location, outputs the probability of each type of crime occurring

print('Please wait while the program runs!')

def get_model(input_shape, NUM_CLASSES):

    X_input = Input((input_shape,))
    
    # First hidden layer(s)
    X = Dense(units = 16, activation = 'relu')(X)
    # Hidden layer feature normalization
    X = BatchNormalization()(X)
    # X = Dropout(0.5)(X)

    # Second hidden layer(s)
    X = Dense(units = 32, activation = 'relu')(X)
    X = BatchNormalization()(X)
    # For regularization
    X = Dropout(0.5)(X)

    # Third hidden layer(s)
    X = Dense(units = 64, activation = 'relu')(X)
    X = BatchNormalization()(X)
    # X = Dropout(0.5)(X)

    # Output layer
    X = Dense(units = NUM_CLASSES, activation = 'softmax')(X)

    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model

X_train, Y_train, NUM_CLASSES = gcd.get_features()
X_train = X_train.T
# Normalize X_train (remember: one feature per column!)
X_train = (X_train - np.mean(X_train, axis = 0)) / np.std(X_train, axis = 0)

Y_train = to_categorical(Y_train, num_classes = NUM_CLASSES)
print(X_train.shape, Y_train.shape)

model = get_model(X_train.shape[1], NUM_CLASSES)
model.compile(optimizer = Adam(lr = 0.002), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 20000, batch_size = X_train.shape[0])

model.save('model.h5')

