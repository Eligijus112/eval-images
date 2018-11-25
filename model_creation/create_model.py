"""
Script to create/test the deep learning model 
"""

### Loading libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

### Defining the share of data that will be in the training data

share_in_train = 1

### Loading the data and spliting to train and test sets

mnist = keras.datasets.fashion_mnist

### Saving the class decoder for future use

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_index = range(len(class_names))
class_df = pd.DataFrame(class_names, index=class_index)
class_df = class_df.rename({0 : 'class_label'}, axis = 'columns')
class_df['class_code'] = class_df.index
class_df.to_csv('main_model/class_decoder.csv', index = False)

## We divide by 255 in order to have the pixel values in the range of 
## [0, 1] 

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

## We concatinate the arrays in order to have the full dataset

X_matrix = np.concatenate((x_train, x_test))
Y_matrix = np.concatenate((y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(X_matrix, Y_matrix, 
                                   test_size= 1 - share_in_train, 
                                   random_state=42)

### Defining the model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(60, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

### Fiting the model

model.fit(x_train, y_train, epochs=10)

### Testing the model on the test set

if(share_in_train != 1):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.3f%%" % (model.metrics_names[1], score[1]*100))

### Saving the model for future use 
## We will only save that model which was created using the full dataset
## We will save the model weights and the specification

if(share_in_train == 1):
    model_json = model.to_json()
    with open("main_model/model_specs.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("main_model/model_weights.h5")

