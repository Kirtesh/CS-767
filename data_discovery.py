
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:29:28 2020

@author: kirte
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

print("\nLoading data in dataframe by providing only file name as input: FileName='bank-full'")
banknotes_FileName='bank-full' 
here = os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here,os.pardir))
file_name = os.path.join(input_dir, banknotes_FileName + ".csv")
print("\nThe file is semicolon delimited so we have set the separator accordingly")
df = pd.read_csv(file_name, sep = ";")


print("\nShowing sample records from the file")
print(df.head())


#data info
df.info()

#  --------------------    Numerical Data Discovery   ----------------------------

# Outlier analysis for numerical data

numd = df[["age", "balance","duration","campaign","previous","subscribed"]]

sns.boxplot(data=numd,x="subscribed",y="age")
plt.show()
sns.boxplot(data=numd,x="subscribed",y="balance")
plt.show()

sns.distplot(numd["balance"])
plt.show()

sns.boxplot(data=numd,x="subscribed",y="duration")
plt.show()
sns.boxplot(data=numd,x="subscribed",y="campaign")
plt.show()
sns.boxplot(data=numd,x="subscribed",y="previous")
plt.show()


# Assigning Subscribed as 'yes' = 0, for correlation
numd.subscribed.replace(('yes', 'no'), (0, 1), inplace=True)

corr = numd.corr()
f, ax = plt.subplots(figsize=(10,12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
_ = sns.heatmap(corr, cmap="YlGn", square=True, ax=ax, annot=True, linewidth=0.1)
plt.title("Pearson correlation of Features", y=1.05, size=15)
plt.show()



# --------------------- Categorical data Discovery -----------------------

catd = df[["job", "marital","education","default","housing","contact","loan","poutcome","subscribed"]]

#Replacing all the binary variables to 0 and 1
catd.default.replace(('yes', 'no'), (1, 0), inplace=True)
catd.housing.replace(('yes', 'no'), (1, 0), inplace=True)
catd.loan.replace(('yes', 'no'), (1, 0), inplace=True)

# Assigning Subscribed as 'yes' = 0, to treat it as Positive Class
catd.subscribed.replace(('yes', 'no'), (0, 1), inplace=True)

#creating Dummies for categorical variables
df_cat = pd.get_dummies(catd)
df_cat.head()	 		 	



#Correlation plot w.r.t subscription
plt.figure(figsize=(14,8))
df_cat.corr()['subscribed'].sort_values(ascending = False).plot(kind='bar')
plt.show()



# Label encoding categorical features to determine best features and correlation
ftr_df= df.copy()  
categorical_columns=["job", "marital","education","default","housing","contact","loan","poutcome"]

le = LabelEncoder()
ftr_df[categorical_columns] = ftr_df[categorical_columns].apply(lambda col:le.fit_transform(col))


# Finding correlation between categorical features
corr2 = ftr_df[categorical_columns].corr()
f, ax = plt.subplots(figsize=(10,12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
_ = sns.heatmap(corr2, cmap="YlGn", square=True, ax=ax, annot=True, linewidth=0.1)
plt.title("Correlation of converted categorical data", y=1.05, size=15)
plt.show()




