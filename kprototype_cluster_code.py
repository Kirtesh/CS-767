# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 22:49:27 2020

@author: kirte
"""


import os
import numpy as np
import pandas as pd
import keras
import sklearn 
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import train_test_split
#from keras.layers import Densimport tensorflow as tf
#from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from keras.utils import np_utils




print("\nLoading data in dataframe by providing only file name as input: FileName='bank-full'")
banknotes_FileName='bank-full' 
here = os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here,os.pardir))
file_name = os.path.join(input_dir, banknotes_FileName + ".csv")
print("\nThe file is semicolon delimited so we have set the separator accordingly")
df = pd.read_csv(file_name, sep = ";")


#Removing non-relevant variables
df_pp = df.drop(columns=['day','pdays','poutcome'],axis=1)
# Remove Outlier
df_pp = df_pp[df_pp.age<75][df_pp.balance<60000][df_pp.duration<2200][df_pp.campaign<40]

print("\nTaking stratified sample from the full data set for faster execution'")
strat_data, _ = train_test_split(df_pp, test_size=0.9, stratify=df_pp[['subscribed']])
kprot_data = strat_data.copy()
kprot_data = kprot_data.reset_index(drop=True)
print("\nShowing data:",kprot_data.count)


print("\nPre-processing to normalize numerical data")
for c in kprot_data.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))


# Specify correct indices for categorical attributes
categorical_columns = [1,2,3,4,6,7,8,9,13] 


#Actual clustering
print("\nStart Clustering using KPrototype with previously identified optimal number of clusters as 7")
kproto = KPrototypes(n_clusters= 7, init='Huang')
clusters = kproto.fit_predict(kprot_data, categorical=categorical_columns)

#Prints the count of each cluster group
print("\nNumber of records in each cluster are:\n",pd.Series(clusters).value_counts())

print("\nAssigning Cluster/Group to each client")
Cluster_Output=pd.concat([kprot_data, pd.DataFrame(clusters,columns=["Group"])], axis=1)

print(Cluster_Output.head())

print("\nWe will use Neural net classification to evaluate Clustering results by treating each assigned cluster/group as a separate class\n")

# ------------------ Data Preprocessing for Classification Evaluation ------------------------


df_cluster = Cluster_Output.copy()

#Replacing all the binary variables to 0 and 1
df_cluster.default.replace(('yes', 'no'), (1, 0), inplace=True)
df_cluster.housing.replace(('yes', 'no'), (1, 0), inplace=True)
df_cluster.loan.replace(('yes', 'no'), (1, 0), inplace=True)


#creating Dummies for categorical variables
df_cluster_cat = pd.get_dummies(df_cluster)
df_cluster_cat.head()

#Removing extra dummy variables & checking descriptive stats
df_cluster_cat=df_cluster_cat.drop(columns=['job_unknown','education_unknown'],axis=1)


#------------------ Cluster Evaluation --  Neural Net Classification  ----------------------------

# Creating X and Y datasets for classifier training and testing
X = df_cluster_cat.drop(['Group'], axis=1).values
Y = df_cluster_cat['Group'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 3)

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


# ------- Neural Network with two hidden layer -------------------------

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(32,activation="relu"))

# Adding the second hidden layer
classifier.add(Dense(16,activation="relu"))

# Adding the output layer
classifier.add(Dense(7,activation="softmax"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history=classifier.fit(x_train, dummy_y_train, batch_size = 10, epochs=40,validation_split=0.3)
# Predicting the Test set results
y_pred = classifier.predict_classes(x_test)

#pre_score = sklearn.average_precision_score(y_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x_test, dummy_y_test)
print("For epoch = {0}, the model test accuracy with two hidden layer is {1}.".format(40,test_results[1]))
#print("The model test average precision score with two hidden layer is {}.".format(pre_score))

ann_cm =  confusion_matrix(y_test,y_pred)

ann_accuracy = accuracy_score(y_test,y_pred)


print("\nShwoing no. of records in different clusters for test data:\n",pd.Series(y_test).value_counts())

print("Confusion Matrix for Neural Net with Multi-Class: \n %s"%ann_cm)

print("Classification Report for all clusters: \n %s"%classification_report(y_test,y_pred))

print("Overall Accuracy for the KPrototype Clustering is: \n %s"%ann_accuracy)


# --------------------------------------------------------------------------------

