# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:29:28 2020

@author: kirte
"""

import os
import numpy as np
import pandas as pd
import keras
#from keras.layers import Densimport tensorflow as tf
#from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import recall_score, accuracy_score,confusion_matrix,classification_report
import sklearn.metrics as sk
import matplotlib.pyplot as plt


# Setting option to display desired columns in output
pd.set_option('display.max_columns',6)

# ------------------ Data Landing ------------------------
print("\nLoading data in dataframe by providing only file name as input: FileName='bank-full'")
banknotes_FileName='bank-full' 
here = os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here,os.pardir))
file_name = os.path.join(input_dir, banknotes_FileName + ".csv")
print("\nThe file is semicolon delimited so we have set the separator accordingly")
df = pd.read_csv(file_name, sep = ";")


print("\nShowing sample records from the file")
print(df.head())


# ---------------------------------- Data Preprocessing --------------------------------------
print("\nPre-processing to remove non-relevant features and remove outliers.'")
print("\nWe need to normalize the numerical attributes and perform one hot enconding on categorical features.'")

#Removing non-relevant variables
df_pp = df.drop(columns=['day','pdays','poutcome'],axis=1)

# Remove Outlier
df_pp = df_pp[df_pp.age<75][df_pp.balance<22000][df_pp.duration<2200][df_pp.campaign<40]

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
df_cat.head()

#Removing extra dummy variables & checking descriptive stats
df_cat=df_cat.drop(columns=['job_unknown','education_unknown'],axis=1)



# -----------------------------  Data Classification Starts ---------------------------------

print("\n---------------------Data Classification Starts ----------------------------")

# Creating X and Y datasets for classifier training and testing
X = df_cat.drop(['subscribed'], axis=1).values
y = df_cat['subscribed'].values

# Using stratified split to get proper sample from smaller class
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.25, random_state = 3)


# ------------------ Starting with One hidden layer architecture and with class imbalance -------
print("\nExecuting classifier with existing class imblanace and with One hidden layer")

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(16,activation="softmax"))
# Adding the output layer
classifier.add(Dense(1,activation="sigmoid"))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, batch_size = 10, epochs=25,validation_split=0.3)
# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with one hidden layer is {1}.".format(25,test_results[1]))
print("The model test average precision score with one hidden layer is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)
ann_accuracy_1 = accuracy_score(y_test,y_pred)
ann_TP = ann_cm[0,0]
ann_FP = ann_cm[1,0]
ann_TN = ann_cm[1,1]
ann_FN = ann_cm[0,1]
# True Positive Rate (TPR)
ann_TPR_1= (ann_TP/(ann_TP+ann_FN))
# True Negative Rate (TNR)
ann_TNR_1= (ann_TN/(ann_TN+ann_FP))

print("\nConfusion Matrix for Neural Net with one hidden layer is: \n %s"%ann_cm)
print("\nTPR - True Positive Rate with one hidden layer is: \n %s"%ann_TPR_1)
print("\nTNR - True Negative Rate with one hidden layer is: \n %s"%ann_TNR_1)

# ----------------------------------------------------------------

# ------- Neural Network with two hidden layer -------------------------

print("\nExecuting classifier with existing class imblanace and with TWO hidden layer")

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(32,activation="softmax"))
# Adding the second hidden layer
classifier.add(Dense(16,activation="softmax"))
# Adding the output layer
classifier.add(Dense(1,activation="sigmoid"))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, batch_size = 10, epochs=25,validation_split=0.3)
# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with two hidden layer is {1}.".format(25,test_results[1]))
print("The model test average precision score with two hidden layer is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)
ann_accuracy_2 = accuracy_score(y_test,y_pred)
ann_TP = ann_cm[0,0]
ann_FP = ann_cm[1,0]
ann_TN = ann_cm[1,1]
ann_FN = ann_cm[0,1]
# True Positive Rate (TPR)
ann_TPR_2= (ann_TP/(ann_TP+ann_FN))
# True Negative Rate (TNR)
ann_TNR_2= (ann_TN/(ann_TN+ann_FP))

print("\nConfusion Matrix for Neural Net with two hidden layer is: \n %s"%ann_cm)
print("\nTPR - True Positive Rate with two hidden layer is: \n %s"%ann_TPR_2)
print("\nTNR - True Negative Rate with two hidden layer is: \n %s"%ann_TNR_2)


# ------- Neural Network with two hidden layer and RELU as activation function-------------------------

print("\nExecuting classifier with existing class imblanace and with TWO hidden layer")

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

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, batch_size = 10, epochs=25,validation_split=0.3)
# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with RELU is {1}.".format(25,test_results[1]))
print("The model test average precision score with RELU is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)
ann_accuracy_relu = accuracy_score(y_test,y_pred)
ann_TP = ann_cm[0,0]
ann_FP = ann_cm[1,0]
ann_TN = ann_cm[1,1]
ann_FN = ann_cm[0,1]
# True Positive Rate (TPR)
ann_TPR_relu= (ann_TP/(ann_TP+ann_FN))
# True Negative Rate (TNR)
ann_TNR_relu= (ann_TN/(ann_TN+ann_FP))

print("\nConfusion Matrix for Neural Net with RELU function is: \n %s"%ann_cm)
print("\nTPR - True Positive Rate with RELU function is: \n %s"%ann_TPR_relu)
print("\nTNR - True Negative Rate with RELU function is: \n %s"%ann_TNR_relu)

# ----------------------------------------------------------------
# ----- Handling Class imbalance with class weights in neural network and with two hidden layers -------------

print("Handling Class imbalance with weighted neural network")

print ("Giving ten times more weightage to minority class, as the the ratio of subscription is 1:10")
weights = {0:10, 1:1}

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, class_weight=weights,
                       batch_size = 10, epochs=25,validation_split=0.3)

# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with class weights is {1}.".format(25,test_results[1]))
print("The model test average precision score with class weights is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)

ann_accuracy_cost_sense = accuracy_score(y_test,y_pred)
ann_TP = ann_cm[0,0]
ann_FP = ann_cm[1,0]
ann_TN = ann_cm[1,1]
ann_FN = ann_cm[0,1]
# True Positive Rate (TPR)
ann_TPR_cost_sense= (ann_TP/(ann_TP+ann_FN))
# True Negative Rate (TNR)
ann_TNR_cost_sense= (ann_TN/(ann_TN+ann_FP))

print("\nConfusion Matrix for Neural Net after class imbalance handling is : \n %s"%ann_cm)
print("\nTPR - True Positive Rate after class handling is: \n %s"%ann_TPR_cost_sense)
print("\nTNR - True Negative Rate after class handling is: \n %s"%ann_TNR_cost_sense)
print("Classification Report after class handling is: \n %s"%classification_report(y_test,y_pred))

# ------------------------------------------------------------------------------------

print("Increasing weights further in favor of Positive Class - Subscribed")

# Giving 20 times more weightage to minority class for testing
weights = {0:20, 1:1}

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, class_weight=weights,
                       batch_size = 10, epochs=25,validation_split=0.3)

# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with greater weight is {1}.".format(25,test_results[1]))
print("The model test average precision score with greater weight is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)
ann_accuracy_cost_sense_2 = accuracy_score(y_test,y_pred)
ann_TP = ann_cm[0,0]
ann_FP = ann_cm[1,0]
ann_TN = ann_cm[1,1]
ann_FN = ann_cm[0,1]
# True Positive Rate (TPR)
ann_TPR_cost_sense_2= (ann_TP/(ann_TP+ann_FN))
# True Negative Rate (TNR)
ann_TNR_cost_sense_2= (ann_TN/(ann_TN+ann_FP))

print("\nConfusion Matrix for Neural Net with greater weight is : \n %s"%ann_cm)
print("\nTPR - True Positive Rate with greater weights is: \n %s"%ann_TPR_cost_sense_2)
print("\nTNR - True Negative Rate with greater weights is: \n %s"%ann_TNR_cost_sense_2)


# ------------------------------------------------------------------------------------

print("Increasing epoch hyperparameter for experiment and evaluation")

# Giving 10 times more weightage to minority class for testing
weights = {0:10, 1:1}

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, class_weight=weights,
                       batch_size = 10, epochs=50,validation_split=0.3)

# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with greater weight is {1}.".format(50,test_results[1]))
print("The model test average precision score with greater weight is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)
ann_accuracy_cost_sense_50 = accuracy_score(y_test,y_pred)
ann_TP = ann_cm[0,0]
ann_FP = ann_cm[1,0]
ann_TN = ann_cm[1,1]
ann_FN = ann_cm[0,1]
# True Positive Rate (TPR)
ann_TPR_cost_sense_50= (ann_TP/(ann_TP+ann_FN))
# True Negative Rate (TNR)
ann_TNR_cost_sense_50= (ann_TN/(ann_TN+ann_FP))

print("\nConfusion Matrix for Neural Net with epoch 50 is : \n %s"%ann_cm)
print("\nTPR - True Positive Rate with epoch 50 is: \n %s"%ann_TPR_cost_sense_50)
print("\nTNR - True Negative Rate with epoch 50 is: \n %s"%ann_TNR_cost_sense_50)


# ------------------------------------------------------------------------------------

print("Increasing epoch and batch size  for experiment and evaluation")

# Giving 10 times more weightage to minority class for testing
weights = {0:10, 1:1}

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, class_weight=weights,
                       batch_size = 100, epochs=50,validation_split=0.3)

# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with greater weight is {1}.".format(50,test_results[1]))
print("The model test average precision score with greater weight is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)
ann_accuracy_cost_sense_bat = accuracy_score(y_test,y_pred)
ann_TP = ann_cm[0,0]
ann_FP = ann_cm[1,0]
ann_TN = ann_cm[1,1]
ann_FN = ann_cm[0,1]
# True Positive Rate (TPR)
ann_TPR_cost_sense_bat= (ann_TP/(ann_TP+ann_FN))
# True Negative Rate (TNR)
ann_TNR_cost_sense_bat= (ann_TN/(ann_TN+ann_FP))

print("\nConfusion Matrix for Neural Net with epoch 50 is : \n %s"%ann_cm)
print("\nTPR - True Positive Rate with epoch 50 is: \n %s"%ann_TPR_cost_sense_bat)
print("\nTNR - True Negative Rate with epoch 50 is: \n %s"%ann_TNR_cost_sense_bat)


# ------------------------------------------------------------------------------------

print("Increasing epoch further with same original batch size for comparision")

# Giving 10 times more weightage to minority class for testing
weights = {0:10, 1:1}

# Fitting the ANN to the Training set
history=classifier.fit(x_train, y_train, class_weight=weights,
                       batch_size = 10, epochs=100,validation_split=0.3)

# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)
pre_score = sk.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, y_test)
print("For epoch = {0}, the model test accuracy with greater weight is {1}.".format(100,test_results[1]))
print("The model test average precision score with greater weight is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)
ann_accuracy_cost_sense_100 = accuracy_score(y_test,y_pred)
ann_TP = ann_cm[0,0]
ann_FP = ann_cm[1,0]
ann_TN = ann_cm[1,1]
ann_FN = ann_cm[0,1]
# True Positive Rate (TPR)
ann_TPR_cost_sense_100= (ann_TP/(ann_TP+ann_FN))
# True Negative Rate (TNR)
ann_TNR_cost_sense_100= (ann_TN/(ann_TN+ann_FP))

print("\nConfusion Matrix for Neural Net with epoch 50 is : \n %s"%ann_cm)
print("\nTPR - True Positive Rate with epoch 50 is: \n %s"%ann_TPR_cost_sense_100)
print("\nTNR - True Negative Rate with epoch 50 is: \n %s"%ann_TNR_cost_sense_100)

# ---------------------------------------------------------------------------------------------------

cm = pd.DataFrame(
{
"Model": ['1 Hidden layer with Softmax','2 Hidden layer with Softmax','2 Hidden layer with RELU',\
          'Cost Sensitive with 2 Hidden layer','Higher Minority Class Weight','Increased Epoch to 50 with Cost Sensitive model',\
         'Increased Epoch to 50 and batch size =100','Increased Epoch to 100 with Cost Sensitive model'],   
"Accuracy": [ann_accuracy_1, ann_accuracy_2, ann_accuracy_relu, ann_accuracy_cost_sense, \
             ann_accuracy_cost_sense_2,ann_accuracy_cost_sense_50,ann_accuracy_cost_sense_bat,ann_accuracy_cost_sense_100 ],
"TPR": [ann_TPR_1, ann_TPR_2, ann_TPR_relu, ann_TPR_cost_sense, ann_TPR_cost_sense_2,\
        ann_TPR_cost_sense_50,ann_TPR_cost_sense_bat,ann_TPR_cost_sense_100], 
"TNR": [ann_TNR_1, ann_TNR_2, ann_TNR_relu, ann_TNR_cost_sense, ann_TNR_cost_sense_2,\
        ann_TNR_cost_sense_50,ann_TNR_cost_sense_bat,ann_TNR_cost_sense_100]		 		 
})

print("\nPerformance Summary for different classifier setup are:\n% s " %cm)



# --------------------------------------------------------------------------------------

