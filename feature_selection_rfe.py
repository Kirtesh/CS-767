# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 23:45:47 2020

@author: kirte
"""

# Importing Python libraries 
import os
import pandas as pd
import numpy as np
import keras
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
#from pylab import savefig
from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer,  OneHotEncoder, StandardScaler,MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import recall_score,auc, roc_auc_score, roc_curve,confusion_matrix,classification_report
#plt.style.use('seaborn-whitegrid')
import sklearn.metrics as sk
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.utils import np_utils



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


															

#---------------------------  Data Transformation -----------------------------


categorical_columns=['job','marital','education','default','housing','loan','contact','day','month','poutcome']
continuous_columns=['age','balance','duration','campaign','pdays','previous']

ftr_df= df.copy()  

# Assigning Subscribed as 'yes' = 0, to treat it as Positive Class
ftr_df.subscribed.replace(('yes', 'no'), (0, 1), inplace=True)

print("\nWe need to normalize the continuous variables. We have used MinMax scaler instead of scaler as feature selection methods cannot accept negative values.")
scaler = MinMaxScaler()
ftr_df[continuous_columns] = scaler.fit_transform(ftr_df[continuous_columns])


print("\nWe have used label Encoder to transform the categorical variable into a continuous sequence for feature selection and then used get dummies method for actual classification.")

le = LabelEncoder()
ftr_df[categorical_columns] = ftr_df[categorical_columns].apply(lambda col:le.fit_transform(col))

print("\nShowing sample records after completing transformations on continuous and categorical variables \n%s"%ftr_df)


#-------------------------------  Feature Selection -----------------------------------------


# Splitting the data
print("\nBefore applying feature selection method, we need to split the data and then address class imbalance.\
 The reason is that feature selection should be applied on same dataset as how we would run classification.")

df_ftr=ftr_df.iloc[:,:-2]
#X = ftr_df[ftr_df.columns[:-2]].values
X = df_ftr.values
Y = ftr_df['subscribed'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state =3)


print("\nWe are using SMOTE - Synthetic Minority Oversampling Technique to remove imbalance and fit training dataset")
sm= SMOTE(random_state=12)
X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train)
print("\nThe Revised_Y_train now has balanced values from both classes as shown below")
print(np.unique(Y_train_res, return_counts=True))



print("\n --------------- Recursive feature elimination using logistic regression as the mode")
# Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model),
# recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of 
# features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either 
# through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from the current set of features
# The features which were selected are rank 1.

model_logistic = LogisticRegression(multi_class='multinomial', max_iter=1000)
sel_rfe_logistic = feature_selection.RFE(estimator=model_logistic,n_features_to_select=10, step=1)
X_train_rfe_logistic = sel_rfe_logistic.fit_transform(X_train_res, Y_train_res)
print("\nPrinting support and Ranking of Recursive feature selection")
print(sel_rfe_logistic.get_support())
print(sel_rfe_logistic.ranking_)
rfe_columns=df_ftr.columns[sel_rfe_logistic.get_support()]

print("\n\n Columns selected from logistic RFE are: \n %s"%rfe_columns.values)


# ------------------------------------------------------------------------

print("\nRunning classification with reduced feature set'")



# ------------------ Data Preprocessing ------------------------
print("\nPre-processing to remove non-relevant features and remove outliers.'")
print("\nWe have also normalized the numerical attributes and perform one hot enconding on categorical features.'")

#Removing non-relevant variables
df_pp = df.drop(columns=['job','marital','education','day','month','pdays','poutcome'],axis=1)

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
df_cat.head()



# ------------------------------------------------------------------------
# Creating X and Y datasets for classifier training and testing
X = df_cat.drop(['subscribed'], axis=1).values
y = df_cat['subscribed'].values

# Using stratified split to get proper sample from smaller class
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.25, random_state = 3)


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
print("For epoch = {0}, the model test accuracy with reduced feature set {1}.".format(25,test_results[1]))
print("The model test average precision score wwith reduced feature set is {}.".format(pre_score))



print("\nConfusion Matrix for Neural Net with reduced feature set : \n %s"%confusion_matrix(y_test,y_pred))

print("\nClassification Report with reduced feature set: \n %s"%classification_report(y_test,y_pred))
