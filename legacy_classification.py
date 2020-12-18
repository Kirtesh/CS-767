# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:20:23 2020

@author: kirte
"""



# Importing Python libraries 
# Importing Python libraries 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
#from pylab import savefig
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import PowerTransformer, MinMaxScaler,LabelEncoder,OneHotEncoder, StandardScaler
from sklearn import tree, feature_selection
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# Setting option to display desired columns in output
pd.set_option('display.max_columns',7)



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


# Assigning Subscribed as 'yes' = 0, to treat it as Positive Class
df.subscribed.replace(('yes', 'no'), (0, 1), inplace=True)


categorical_columns=['job','marital','education','default','housing','loan','contact','day','month','poutcome']
continuous_columns=['age','balance','duration','campaign','pdays','previous']

																

#---------------------------  Data Transformation -----------------------------


print("\n Convert Categorical Variables Into Dummy Variables through get_dummies() method.\
 The label encoder will not give correct results as the labels has inherit hierarchy or statistical significance with one value higher or lower than other.\
 But in real sense all these categorical variables are of same significance.")

df_prc = df.copy()
scaler = MinMaxScaler()
df_prc[continuous_columns] = scaler.fit_transform(df_prc[continuous_columns])

input_data = df_prc[categorical_columns]
dummies = [pd.get_dummies (df_prc[c], columns=[c], prefix="flag") for c in input_data.columns]
binary_data = pd.concat (dummies, axis =1)

df_convert = pd.concat([binary_data, df_prc[continuous_columns]], axis=1, sort=False)
print("\nDataframe after assigning dummy variables and normalizing\n %s"%df_convert.head())

X = df_convert.values
Y = df['subscribed'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state =3)

print("\nThe Y_train obtained from full data is")
print(np.unique(Y_train, return_counts=True))

print("\nUsing the SMOTE method again to address class imbalance issue")
sm= SMOTE(random_state=3)
X_train_Conv, Y_train_Conv = sm.fit_sample(X_train, Y_train)

print("\nThe Revised_Y_train now has balanced values from both classes as shown below")
print(np.unique(Y_train_Conv, return_counts=True))


# -----------------------------------------------------------------------------------------



print("\nLogistic Regression with full features after SMOTE oversampling")
log_reg_classifier = LogisticRegression (max_iter=10000)
log_reg_classifier.fit (X_train_Conv,Y_train_Conv)
prediction = log_reg_classifier.predict(X_test)
lr_smote_cm = confusion_matrix(Y_test, prediction)
lr_smote_Accuracy=accuracy_score(Y_test, prediction)
lr_smote_TP = lr_smote_cm[0,0]
lr_smote_FP = lr_smote_cm[1,0]
lr_smote_TN = lr_smote_cm[1,1]
lr_smote_FN = lr_smote_cm[0,1]
# True Positive Rate (TPR)
lr_smote_TPR= (lr_smote_TP/(lr_smote_TP+lr_smote_FN))
# True Negative Rate (TNR)
lr_smote_TNR= (lr_smote_TN/(lr_smote_TN+lr_smote_FP))

print("Confusion Matrix for Logistic Regression fitted with SMOTE method is: \n %s"%confusion_matrix(Y_test, prediction))
print("Classification Report for Logistic Regression fitted with SMOTE method: \n %s"%classification_report(Y_test, prediction))
