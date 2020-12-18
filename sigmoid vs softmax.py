# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:44:46 2020

@author: kirte
"""

import os
import numpy as np
import pandas as pd
#from keras.layers import Densimport tensorflow as tf
#from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import PowerTransformer,  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score,auc, roc_auc_score, roc_curve,confusion_matrix,classification_report
#plt.style.use('seaborn-whitegrid')
import sklearn.metrics as sk
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.utils import np_utils



# ------------------ Data Landing ------------------------
print("\nLoading data in dataframe by providing only file name as input: FileName='bank-full'")
banknotes_FileName='bank-full' 
here = os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here,os.pardir))
file_name = os.path.join(input_dir, banknotes_FileName + ".csv")
print("\nThe file is semicolon delimited so we have set the separator accordingly")
df = pd.read_csv(file_name, sep = ";")



# ------------------ Data Preprocessing ------------------------
print("\nPre-processing to remove non-relevant features and remove outliers.'")
print("\nWe have also normalized teh numerical attributes and perform one hot enconding on categorical features.'")

#Removing non-relevant variables
df_pp = df.drop(columns=['day','pdays','poutcome'],axis=1)

# Remove Outlier
df_pp = df_pp[df_pp.age<75][df_pp.balance<60000][df_pp.duration<2200][df_pp.campaign<40]

#Pre-processing to normalize numerical data
for c in df_pp.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    df_pp[c] =  pt.fit_transform(np.array(df_pp[c]).reshape(-1, 1))

#Replacing all the binary variables to 0 and 1
df_pp.default.replace(('yes', 'no'), (1, 0), inplace=True)
df_pp.housing.replace(('yes', 'no'), (1, 0), inplace=True)
df_pp.loan.replace(('yes', 'no'), (1, 0), inplace=True)

# Assigning Subscribed as 'yes' = 0, to treat it as Positive Class
df_pp.subscribed.replace(('yes', 'no'), (0, 1), inplace=True)


#creating Dummies for categorical variables
df_cat = pd.get_dummies(df_pp)

#Removing extra dummy variables & checking descriptive stats
df_cat=df_cat.drop(columns=['job_unknown','education_unknown'],axis=1)



# ------------------  Data Classification Starts ----------------------------

print("\n---------------------Data Classification Starts ----------------------------")

# Creating X and Y datasets for classifier training and testing
X = df_cat.drop(['subscribed'], axis=1).values
y = df_cat['subscribed'].values

# Using stratified split to get proper sample from smaller class
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.25, random_state = 3)



# ------- Neural Network with two hidden layer and RELU as activation function-------------------------
# ----- Handling Class imbalance with class weights in neural network and with two hidden layers -------------

print("\nWeighted neural network and with 2 hidden layer and 1 output layer with sigmoid function")

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(32,activation="relu"))
# Adding the second hidden layer
classifier.add(Dense(16,activation="relu"))
# Adding the output layer
classifier.add(Dense(1,activation="sigmoid"))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print ("\nGiving ten times more weightage to minority class, as the the ratio of subscription is 1:10")
weights = {0:10, 1:1}

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, class_weight=weights,
                       batch_size = 10, epochs=25,validation_split=0.3)

# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with sigmoid function {1}.".format(25,test_results[1]))
print("The model test average precision score with sigmoid function is {}.".format(pre_score))



print("\nConfusion Matrix for Neural Net with sigmoid function is : \n %s"%confusion_matrix(y_test,y_pred))

print("\nClassification Report with sigmoid function is: \n %s"%classification_report(y_test,y_pred))


# ------- Neural Network with two hidden layer and RELU as activation function-------------------------
# ----- Handling Class imbalance with class weights in neural network and with two hidden layers -------------

print("\nWeighted neural network and with 2 hidden layer and 2 output layer with softmax function")

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(32,activation="relu"))
# Adding the second hidden layer
classifier.add(Dense(16,activation="relu"))
# Adding the output layer
classifier.add(Dense(2,activation="softmax"))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print ("Giving ten times more weightage to minority class, as the the ratio of subscription is 1:10")
weights = {0:10, 1:1}


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_Y)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_test)
encoded_YTest = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_test= np_utils.to_categorical(encoded_YTest)

# Fitting the ANN to the Training set
history=classifier.fit(x_train, dummy_y_train, class_weight=weights,
                       batch_size = 10, epochs=25,validation_split=0.3)

# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, dummy_y_test)
print("For epoch = {0}, the model test accuracy with 2 output nodes is {1}.".format(25,test_results[1]))
print("The model test average precision score with 2 output nodes is {}.".format(pre_score))




print("\nConfusion Matrix for Neural Net with softmax and 2 output nodes is : \n %s"%confusion_matrix(y_test,y_pred))

print("\nClassification Report with softmax and 2 output nodes is: \n %s"%classification_report(y_test,y_pred))