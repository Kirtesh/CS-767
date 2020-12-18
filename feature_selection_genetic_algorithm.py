
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:20:23 2020

@author: kirte
"""


# Importing Python libraries 
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
#from pylab import savefig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, MinMaxScaler,LabelEncoder,OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import random




# Setting option to display desired columns in output
pd.set_option('display.max_columns',7)



# ------------------------------ Data Landing ------------------------
print("\nLoading data in dataframe by providing only file name as input: FileName='bank-full'")
banknotes_FileName='bank-full' 
here = os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here,os.pardir))
file_name = os.path.join(input_dir, banknotes_FileName + ".csv")
print("\nThe file is semicolon delimited so we have set the separator accordingly")
df_raw = pd.read_csv(file_name, sep = ";")


print("\nTaking stratified sample from the full data set for faster execution'")
strat_data, _ = train_test_split(df_raw, test_size=0.9, stratify=df_raw[['subscribed']])
df = strat_data.copy()
df = df.reset_index(drop=True)
print("\nShowing data:\n",df.count)


																

#---------------------------  Data Transformation -----------------------------


print("\n Convert Categorical Variables Into Dummy Variables through get_dummies() method \
      and normalized the continuous attributes")



categorical_columns=['job','marital','education','default','housing','loan','contact','day','month','poutcome']
continuous_columns=['age','balance','duration','campaign','pdays','previous']

# Assigning Subscribed as 'yes' = 0, to treat it as Positive Class
df.subscribed.replace(('yes', 'no'), (0, 1), inplace=True)


df_prc = df.copy()
scaler = MinMaxScaler()
df_prc[continuous_columns] = scaler.fit_transform(df_prc[continuous_columns])

input_data = df_prc[categorical_columns]
dummies = [pd.get_dummies (df_prc[c], columns=[c], prefix=c) for c in input_data.columns]
binary_data = pd.concat (dummies, axis =1)

df_convert = pd.concat([binary_data, df_prc[continuous_columns]], axis=1, sort=False)
print("\nDataframe after assigning dummy variables and normalizing\n %s"%df_convert.head())

X = df_convert
Y = df['subscribed']

X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state =3)


print("\nUsing the SMOTE method again to address class imbalance issue")
sm= SMOTE(random_state=3)
X_train_conv, y_train_conv = sm.fit_sample(X_train_raw.values, y_train_raw.values)

X_train = pd.DataFrame(X_train_conv, columns = df_convert.columns)

y_train = y_train_conv
#y_train = pd.DataFrame(y_train_conv, columns = ['subscribed'])



# -----------------------------------------------------------------------------------------

print("\n -------------------- Now Running Classification on the dataset -------------------------------")

print("\nLogistic Regression with full features after SMOTE oversampling")
logmodel = LogisticRegression (max_iter=10000)
logmodel.fit (X_train,y_train)
prediction = logmodel.predict(X_test)


print("\nAccuracy for Logistic Regression fitted with SMOTE method is= "+str(accuracy_score(y_test,prediction)))
print("Classification Report for Logistic Regression fitted with SMOTE method: \n %s"%classification_report(y_test, prediction))


# ---------------------------------------------------------------------



#defining various steps required for the genetic algorithm
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],y_train)
        predictions = logmodel.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross,mutation_rate):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    #print(population_nextgen)
    return population_nextgen

def generations(size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                    X_test, y_train, y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print(scores[:2])
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score


chromo,score=generations(size=200,n_feat=81,n_parents=100,mutation_rate=0.10,
                      n_gen=38,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)


logmodel.fit(X_train.iloc[:,chromo[-1]],y_train)
predictions = logmodel.predict(X_test.iloc[:,chromo[-1]])
print("\nAccuracy score after genetic algorithm is= "+str(accuracy_score(y_test,predictions)))


column_list = sorted(X_train.iloc[:,chromo[-1]])

print("\nShwoing features selected from genetic algorithm \n \n",column_list)


