import pandas as pd                                         # loading and cleaning the dataset
import numpy as np                                          # manipulating it

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

import tensorflow as tf
from tensorflow import keras

####################################### Preparing data

path="C:/Users/vanes/OneDrive - Universitat de Barcelona/8e semestre/ML-EELS/"

ds = path + '/Mn_Fe_dataset.pkl'
lb = path + '/Mn_Fe_labels.pkl'

X = pd.read_pickle(ds)
y = pd.read_pickle(lb)

#X = X.to_numpy()
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

####################################### Defining the model

model = keras.Sequential([                                
    keras.layers.Dense(1200, activation='relu', input_shape=(X_train.shape[1],)),          
    keras.layers.Dropout(0.7),
    keras.layers.Dense(1200/(2*(1200+1+1))),
    keras.layers.Dropout(0.7),
    keras.layers.Dense(1200/(2*(1200+2+1))),
    keras.layers.Dropout(0.7),
    keras.layers.Dense(1200/(2*(1200+3+1))),
    keras.layers.Dropout(0.7),
    keras.layers.Dense(5, activation='softmax')
])

# the last layer need to be 2, affiliation yes (1) or not (0), but with softmax we get their probabilities

model.compile(optimizer=tf.keras.optimizers.Adam(), # optimizer='adam'
              loss='sparse_categorical_crossentropy',   
              metrics=['accuracy'])

######################################## Training and evaluating the mdeo

model.fit(X_train, y_train, validation_split=0.2, batch_size=256, epochs=200) 
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc) 

######################################## Making predictions

pred_prob = model.predict(X_test)
predictions = []

for i in range (0,len(pred_prob)):
    predictions.append(np.argmax(pred_prob[i]))

######################################## Accuracy

print('Accuracy', accuracy_score(y_test, predictions))
