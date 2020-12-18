# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:53:39 2020

@author: kirte
"""



# Importing Python libraries 
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt





# Setting option to display desired columns in output
pd.set_option('display.max_columns',7)


# ------------------ Data Landing ------------------------
print("\nLoading data in dataframe by providing file name as input: FileName='bank-full'")
banknotes_FileName='bank-full' 
here = os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here,os.pardir))
file_name = os.path.join(input_dir, banknotes_FileName + ".csv")
print("\nThe file is semicolon delimited so we have set the separator accordingly")
df = pd.read_csv(file_name, sep = ";")




# ---------------------------------- Data Preprocessing --------------------------------------
print("\nPre-processing to remove non-relevant features and remove outliers.'")
print("\nWe need to group numerical attributes in bins to make them categorical.")
print("Perform one hot enconding on all categorical features'")


#Removing non-relevant variables
df_pp = df.drop(columns=['day','pdays','poutcome','previous'],axis=1)

# Remove Outlier
df_pp = df_pp[df_pp.age<75][df_pp.balance<22000][df_pp.duration<2200][df_pp.campaign<40]


# Assigning Subscribed as 'yes' = 0, to treat it as Positive Class
df_pp.subscribed.replace(('yes', 'no'), (0, 1), inplace=True)


categorical_columns=['job','marital','education','default','housing','loan','contact','month']
continuous_columns=['age','balance','duration','campaign']


print("\nShowing sample records from the file")
print(df_pp.head())


print("\nCreating historgram to identify the appropriate bins/groups to be created for continuous columns")
df_cont = df_pp[continuous_columns]
hist = df_cont.hist()


# # Binning Age values in different groups
def ageSegment(x):  
  if x['age'] > 0 and x['age'] <=25:       
      return 'Age_G1'
  elif x['age'] > 25 and x['age'] <= 35: 
      return 'Age_G2'
  elif x['age'] > 35 and x['age'] <= 50: 
      return 'Age_G3'
  elif x['age'] > 50 and x['age'] <= 60: 
      return 'Age_G4'
  elif x['age'] > 60: 
      return 'Age_G5'

df_pp['age'] = df_pp.apply(ageSegment, axis=1)

# Binning Duration data in different groups
def durationSegment(x):  
  if x['duration'] > 0 and x['duration'] <=200:       
      return 'Duration_G1'
  elif x['duration'] > 200 and x['duration'] <= 500: 
      return 'Duration_G2'
  elif x['duration'] > 500: 
      return 'Duration_G3'

df_pp['duration'] = df_pp.apply(durationSegment, axis=1)


# Binning Balanace data in different groups
def balanceSegment(x):  
  if x['balance'] > 0 and x['balance'] <=2000:       
      return 'Balanace_G1'
  elif x['balance'] > 2000 and x['balance'] <= 5000: 
      return 'Balanace_G2'
  elif x['balance'] > 5000: 
      return 'Balanace_G3'
df_pp['balance'] = df_pp.apply(balanceSegment, axis=1)


# Grouping Campaign data
def campaignSegment(x):  
  if x['campaign'] > 0 and x['campaign'] <=1:       
      return 'Campaign_G1'
  elif x['campaign'] > 1 and x['campaign'] <= 3: 
      return 'Campaign_G2'
  elif x['campaign'] > 3 and x['campaign'] <= 6: 
      return 'Campaign_G3'
  elif x['campaign'] > 6 and x['campaign'] <= 10:
      return 'Campaign_G4'
  elif x['campaign'] > 10:
      return 'Campaign_G5'
      
df_pp['campaign'] = df_pp.apply(campaignSegment, axis=1)



print("\nShowing sample records after binning or discretization of the continous data")
print(df_pp.head())



#creating Dummies for categorical variables
df_cat = pd.get_dummies(df_pp)
df_cat.head()

#Removing extra dummy variables & checking descriptive stats
df_cat=df_cat.drop(columns=['job_unknown','education_unknown'],axis=1)



# -----------------------------------------------------------------------------------------

print("\n -------------------- Now Running Classification on the dataset -------------------------------")


# Creating X and Y datasets for classifier training and testing
X = df_cat.drop(['subscribed'], axis=1).values
y = df_cat['subscribed'].values													


X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state =3)


print("\nUsing the SMOTE method to address class imbalance issue")
sm= SMOTE(random_state=3)
X_train_Conv, Y_train_Conv = sm.fit_sample(X_train, Y_train)


NB_classifier = CategoricalNB().fit(X_train_Conv,Y_train_Conv)
prediction = NB_classifier.predict(X_test)

print("\nConfusion Matrix for Naive Bayes fitted with SMOTE method is: \n %s"%confusion_matrix(Y_test, prediction))
print("Classification Report for Naive Bayes fitted with SMOTE method: \n %s"%classification_report(Y_test, prediction))


print("\nNow collecting the probabilities from the Bayes model to be used for segmentation")

prediction_prob_actual = NB_classifier.predict_proba(X_test)
print(prediction_prob_actual)

result = pd.DataFrame(prediction_prob_actual[:,0], columns = ['0'])


print("\nCreating a function to assign different segments based on positive class probability")

def getSegment(P):    
    if P['0'] > 0.8 :
        return "Very Likely"
    elif P["0"] < 0.8 and P["0"] > 0.6:
        return "Likely"
    elif P["0"] < 0.6 and P["0"] > 0.4:
        return "Undecided"
    elif P["0"] < 0.4 and P["0"] > 0.2:
        return "Unlikely "
    elif P["0"] < 0.2 :
        return "Not Interested"
       

result['Segment'] = result.apply(getSegment, axis=1)
print("\nShowing segment assignment as grapgh and sample data \n",result['Segment'])

sns.catplot(x="Segment", kind="count", data=result)
plt.show()

