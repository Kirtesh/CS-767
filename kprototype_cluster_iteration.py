
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:41:21 2020

@author: kirte
"""

import os
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt



print("\nLoading data in dataframe by providing only file name as input: FileName='bank-full'")
banknotes_FileName='bank-full' 
here = os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here,os.pardir))
file_name = os.path.join(input_dir, banknotes_FileName + ".csv")
print("\nThe file is semicolon delimited so we have set the separator accordingly")
df = pd.read_csv(file_name, sep = ";")


print("\nShowing sample records from the file")
print(df.head())

# Taking 10% sample data through Stratefied Sampling to run multiple iterations
strat_data, _ = train_test_split(df, test_size=0.9, stratify=df[['subscribed']])

kprot_data = strat_data.copy()

print(kprot_data.count)


#Pre-processing to normalize continuous numerical data
for c in df.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))


print("\nShowing sample records after normalization")
print(kprot_data.head())


# Specifying indices for categorical attributes
categorical_columns = [1,2,3,4,6,7,8,9,10,15,16] 


#Elbow plot with cost (will take a LONG time)
costs = []
n_clusters = []
clusters_assigned = []

for i in tqdm(range(2, 16)):
    try:
        kproto = KPrototypes(n_clusters= i, init='Cao', verbose=2)
        clusters = kproto.fit_predict(kprot_data, categorical=categorical_columns)
        costs.append(kproto.cost_)
        n_clusters.append(i)
        clusters_assigned.append(clusters)
    except:
        print(f"Can't cluster with {i} clusters")

plt.plot(n_clusters, costs, 'o', color='black')
plt.legend(title='Elbow Plot for cluster analysis: ', fontsize='medium')
plt.xlabel('Clusters')
plt.ylabel('Cost')

plt.show()

      