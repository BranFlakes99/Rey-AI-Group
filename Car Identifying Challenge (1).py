#!/usr/bin/env python
# coding: utf-8

# In[3]:


conda install -c anaconda pillow
#We need 'pillow' because in code we call it PIL which stands for Python Image
#Library so we can use a visual form of data with our AI, Pillow is jpg compatible in this regard


# In[1]:


#We want to install tensorflow as it is open-sourced and used for deep neural networking. Keras would be used
#for the same reason being that it is an open-source library where we can 

conda install tensorflow keras sklearn matplotlib pandas pil


# In[2]:


#Team Note indicated as TM, for team especially to read for ease of reading 

##Example: EM Python does not acknowledge whitespace so I have seperated
##variavles in brackets so it is easier to read

#The 'train' variable here has been used because the training dataset is in a 
#locally stored folder called 'train' that contains 18 folders.
#Given that arrays begin with a 0, this results in the folders to train with 
#ending in 17 instead of 18
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
#'Classes' covers the training data we're using(images and their respective folders)
classes = 17
#This method returns current working directory of a process below
cur_path = os.getcwd()

#'Classes' covers the training data we're using(images and their respective folders)
for i in range( classes ):
        #Here we set a path
        path = os.path.join( cur_path, 'trainSigns', str(i) )
        #Here we assign it so the images themselves are what the path searches the directory for
        images = os.listdir( path )
        
        #We set 'a' as a variable to set a sort of bookmark in the array. We use it to resize the images
        for a in images:
            try:
                image = Image.open( path + '\\' + a )
                image = Image.resize( ( 30,30 ) )
                image = np.array( image )
                data.append( image )
                labels.append( i )
            except:
                print("Error loading image")
                
        #Convert lists into numpy arrays for further use
        #Data is specifically the images themselves whereas Labels is their file name
        data = np.array( data )
        labels = np.array( labels )
        
#Regarding errors below, I've tried installing tensorflow, and other packages for that matter to resolve these issues
#but have had no such luck in getting through the issue


# In[1]:


#The below code is suppose to train and validate the model, data is defined above and I'm unsure why it isn't running
print( data.shape, labels.shape )
#Use train_test_split() to split apart training and testing data, eventually breaking it down into a binary (2x2) matrix
X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.2, random_state=17 )

print( X_train.shape, X_test.shape, y_train.shape, y_test.shape )

#Convert images to a binary matrix
y_train = to_categorical( y_train, 17 )
y_test = to_categorical( y_test, 17 )


# In[6]:


#Building a Convolutional Neural Network

#Structure is:
#2 Convs2D layer ( filter=32 , kernel_size=(5,5), activation="relu" )
#MaxPool2D layer ( pool_size=( 2,2 ) )
#Dropout layer ( rate = 0.25 )
#2 Convs2D layer ( filter=64 , kernel_size=(3,3), activation="relu" )
#MaxPool2D layer ( pool_size=( 2,2 ) )
#Dropout layer ( rate = 0.25 )
#Flatten layers to 1 dimension
#Dense Fully connected layer ( 256 nodes, activation=”relu” )
#Dropout layer ( rate=0.5 )
#Dense layer ( 17 nodes, activation=”softmax” )

#For all of this to be possible we require the use of the Keras python package to manipulate a model by adding dimensions

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
model.add( Dense (17, activation='softmax' ) )

#Compilation of the model:
model.compile( loss = 'categorical_crossetropy', optimizer='adam', metrics=['accuracy'] )


# In[7]:


#Here we want to finally train the model to ensure it is valid

#We will run 15 total tests
epochs = 15
history = model.fit( X_train, y_train, batch_sizes=64, epoch=epochs, validation_data=( X_tests, y_test ) )


# In[8]:


#Plotting a graph with results of the accuracy and loss overtime through testing

plt.figure( 0 )
plt.plot( history.history[ 'accuracy' ], label='training accuracy')
plt.plot( history.history[ 'val_accuracy' ], label='val accuracy')
#plt.title just names the graph 
plt.title( 'Accuracy' )
#Epochs are effectively how many times a machine runs througha  database
plt.xlabel( 'epochs' ) 
#Overtime will calculate and display 'loss'
plt.ylabel( 'loss' )
plt.legend()


# In[9]:


#Testing the model with the dataset, finally

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
accuracy_score( labels, pred )
0.9532066508313539


# In[10]:


#Save model using the appropriate function and format
model.save('traffic_Classifier.h5')


# In[ ]:




