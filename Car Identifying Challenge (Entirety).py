#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Team Note indicated as TM, for team especially to read for ease of reading 

##Example: EM Python does not acknowledge whitespace so I have seperated
##variavles in brackets so it is easier to read

#The 'train' variable here has been used because the training dataset is in a 
#locally stored folder called 'train' that contains 43 folders.
#Given that arrays begin with a 0, this results in the folders to train with 
#ending in 42 instead of 43
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#PIL (Python Image Library) allows you to insert image content into an array
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

#Images are being stored in the following arrays
data = []
labels = []
classes = 43
cur_path = os.getcwd()

for i in range( classes ):
        path = os.path.join( cur_path, 'train', str(i) )
        images = os.listdir( path )
        
        #We set a as a variable to create functionsfor the training images we 
        #select
        for a in images:
            try:
                image = Image.open( path + '\\' + a )
                image = Image.resize( ( 30,30 ) )
                image = np.array( image )
                #sim - Image.fromarray(image)
                data.append( image )
                labels.append( i )
            except:
                print("Error loading image")
                
        #Convert lists into numpy arrays for further use
        data = np.array( data )
        labels = np.array( labels )
        
print( data.shape, labels.shape )
X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.2, random_state=42 )

print( X_train.shape, X_test.shape, y_train.shape, y_test.shape )

y_train = to_categorical( y_train, 43 )
y_test = to_categorical( y_test, 43 )

#Building a Convolutional Neural Network

#Structure is:
#2 Convs2D layer ( filter=32 , kernel_size=(5,5), activation="relu" )
#MaxPool2D layer ( pool_size=( 2,2 ) )
#Dropout layer ( rate = 0.25 )
#2 Convs2D layer ( filter=64 , kernel_size=(3,3), activation="relu" )
#MaxPool2D layer ( pool_size=( 2,2 ) )
#Dropout layer ( rate = 0.25 )
#Flatten layer to squeeze the layers into 1 dimension
#Dense Fully connected layer ( 256 nodes, activation=”relu” )
#Dropout layer ( rate=0.5 )
#Dense layer ( 43 nodes, activation=”softmax” )

model = Sequential()
model.add( Conv2D( filters=32, kernel_size(5,5), activation='relu', input_shape=X_train.shape[1:] ) )
model.add( Conv2D( filters=32, kernel_size(5,5), activation='relu') )
model.add( MaxPool2D( pool_size=( 2,2 ) ) )
model.add( Dropout( rate = 0.25 ) )
model.add( Conv2D( filters=64, kernel_size( 3,3 ), activation='relu' ) )
model.add( Conv2D( filters=64, kernel_size( 3,3 ), activation='relu' ) )
model.add( Dropout( rate=0.25 ) )
model.add( Flatten() )
model.add( Dense ( 256, activation='relu' ) )
model.add( Dropout ( rate=0.5 ) )
model.add( Dense (43, activation='softmax' ) )

#Compilation of the model:
model.compile( loss = 'categorical_crossetropy', optimizer='adam', metrics=[ 'accuracy' ] )

#Here we want to finally train the model to ensure it is valid
#We will run 15 total tests
epochs = 15
history = model.fit( X_train, y_train, batch_sizes=64, epoch=epochs, validation_data=( X_tests, y_test ) )

#Plotting a graph with results of the accuracy and loss overtime through testing
plt.figure( 0 )
plt.plot( history.history[ 'accuracy' ], label='training accuracy')
plt.plot( history.history[ 'val_accuracy' ], label='val accuracy')
plt.title( 'Accuracy' )
plt.xlabel( 'epochs' ) 
plt.ylabel( 'accuracy' )
plt.legend()
plt.show()

plt.figure( 1 )
plt.plot( history.history[ 'loss' ], label='training loss')
plt.plot( history.history[ 'val_loss' ], label='val loss')
plt.title( 'Loss' )
plt.xlabel( 'epochs' ) 
plt.ylabel( 'loss' )
plt.legend()
plt.show()

#Testing accuracy on test dataset
from sklearn.metrics import accuracy_score
import pandas as pd
y_test = pd.read_csv( 'Test.csv' )
labels = y_test[ "ClassId" ].values
imgs = y_test[ "Path" ].values

data=[]

for img in imgs:
    image Image.open( img )
    image = image.resize( (30,30) )
    data.append( np.array( image ) )
    
X_test = np.array( data )

pred = model.predict_classes( X_test )

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score( labels, pred ) )

model.save('traffic_classifier.h5')


# In[ ]:




