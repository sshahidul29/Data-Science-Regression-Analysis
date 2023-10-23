#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import sklearn as sk
from scipy.stats import randint as sp_randint
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

## Standerdize the input training dataset ##
def scaling(X_train, X_test):  
    
    ##### Scaling the features ########
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test    


## Gridsearch to find the optimum values of different parameters of MLP regressor 
## ------------------------------------------------------------------------------

## Initialize the MLP model
model = MLPRegressor(max_iter=500)

## Initilize different tuples of the hidden layers
hiddenlayers = [ [22, 20, 10, 5], [22, 15, 8, 4], [22, 15, 5, 2] ]; 

## Initilize different values of the regularization parameters, alpha 
alpha = [x for x in np.linspace(0.0001, 0.05, num = 5) ]

## Initilize different values of the learning rates
learning_rate_init = [x for x in np.linspace(0.0001, 0.05, num=5) ]

## Initialize different grid search parameters 
param_grid_MLP = [{'solver' : ['adam'],
                  'hidden_layer_sizes' : hiddenlayers,
                   'activation' : ['logistic', 'relu'],
                   'alpha' : alpha,
                   'learning_rate_init' : learning_rate_init
                  }]

## Define the folds for cross validation   
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)

## Define the grid search using cross-validation (CV) and other predefined parameters
search = GridSearchCV(estimator=model, param_grid=param_grid_MLP, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, verbose=True)

## Execute search
result = search.fit(X_train, y_train)

# Summarize the optimum parameters
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

## Read input and output dataset, and scaling the dataset
## ------------------------------------------------------
datasetHeader = pd.read_csv("C:/COVID-19/TrainingData/finalTrainData.csv")
colName = datasetHeader.columns
dataset = pd.read_csv("C:/COVID-19/TrainingData/finalTrainDataNoHeader.csv",header=None, skiprows=1)

## Converting from data frame to numpy data structure  
dataset.astype('int32').dtypes
data = dataset.to_numpy()
(r,c) = data.shape

## Total input dataset [22 input features]
inputData = data[:, 0:-5]

## Output dataset [ CFR = Cummulative number of Death / Cumulative number of cases ]
CFR = data[:,-3]

## Output dataset [CSR = Cummulative number of Cases / Total population ]
CSR = data[:,-1]


## Split the dataset into training and testing parts for the CFR
## -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(inputData, CFR, train_size=0.7, random_state=0)

## Scaling of the train and testing dataset ##
X_train, X_test = scaling(X_train, X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
y_test = sc_y.fit_transform(y_test.reshape(-1, 1))

## Neural netwrok regressor for CFR ##
regrCFR = MLPRegressor(verbose=False, activation='relu', solver='adam', alpha = 0.0001, learning_rate_init=0.012575, hidden_layer_sizes=([22, 20, 10, 5]), max_iter=400)
regrCFR.fit(X_train, y_train)
pd.DataFrame(regrCFR.loss_curve_).plot()

## Save the model to disk 
filename = 'finalized_modelCFR.sav'
regrCFR = pickle.load(open("C:/COVID-19/Models/"+filename, 'rb'))
predictionsTrain = regrCFR.predict(X_train)

## Load the best trained model for CFR 
## pickle.dump(regr, open("C:/COVID-19/Models/"+filename, 'wb'))

## Train mean absolute error for CFR
print(mean_absolute_error(y_train, predictionsTrain) )

## Test mean absolute error for CFR
predictionsTest = regrCFR.predict(X_test)
print(mean_absolute_error(y_test, predictionsTest))


## Split the dataset into training and testing parts for the CSR
## -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(trainData, CSR, train_size=0.7, random_state=0)

## Scaling of the train and testing dataset ##
X_train, X_test = scaling(X_train, X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
y_test = sc_y.fit_transform(y_test.reshape(-1, 1))

## Neural netwrok regressor for CSR ##
regrCSR = MLPRegressor(verbose=False, activation='relu', solver='adam', alpha = 0.0001, learning_rate_init=0.011, hidden_layer_sizes=([22, 18, 14, 6]), max_iter=400)
regrCSR.fit(X_train, y_train)
pd.DataFrame(regrCSR.loss_curve_).plot()

## save the model to disk
filename = 'finalized_modelCSR.sav'
regrCFR = pickle.load(open("C:/COVID-19/Models/"+filename, 'rb'))

## Load the best trained model for CSR 
#pickle.dump(regrCSR, open("C:/COVID-19/Models/"+filename, 'wb'))

## Train mean absolute error for CSR
predictionsTrain = regrCSR.predict(X_train)
print(mean_absolute_error(y_train, predictionsTrain) )

## Train mean absolute error for CSR
predictionsTest = regrCSR.predict(X_test)
print(mean_absolute_error(y_test, predictionsTest) )


## Correlation computation 
## -------------------------
prepCFR = (np.corrcoef(precipData,CFR) )
tempCFR = (np.corrcoef(tempData,CFR) )
pm25CFR = (np.corrcoef(pm25Data,CFR) )
solarCFR = (np.corrcoef(solarRadData,CFR) )
haqCFR = ( np.corrcoef( haqIndexData,CFR) )

prepCSR = (np.corrcoef(precipData,CSR) )
tempCSR = (np.corrcoef(tempData,CSR) )
pm25CSR = (np.corrcoef(pm25Data,CSR) )
solarCSR = (np.corrcoef(solarRadData,CSR) )
haqCSR = (np.corrcoef(haqIndexData,CSR) )

corrCoeffCFR = [prepCFR[0,1], tempCFR[0,1], pm25CFR[0,1], solarCFR[0,1], haqCFR[0,1] ]
corrCoeffCSR = [prepCSR[0,1], tempCSR[0,1], pm25CSR[0,1], solarCSR[0,1], haqCSR[0,1] ]

print(corrCoeffCFR)
print(corrCoeffCSR)

## Data visualization ##
## ---------------------
## Training Data categorization based on different inpute features 
for i in range(inputData.shape[1]):
    plt.figure(figsize=(9,5))
    plt.ylabel("Locations",fontsize=13)
    plt.xlabel(colName[i],fontsize=13)
    plt.grid(1)
    plt.scatter(x, inputData[0:57,i], edgecolors=(0,0,0), s=30, c='g')
    #plt.hist(trainData[0:57,i], 5, density=True, facecolor='g', alpha=0.75)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("C:/COVID-19/Figures/" + colName[i]+"_category_hist.png", dpi=250)  
    
## Generating the correlation figures based on the CFR    
for i in range(inputData.shape[1]):
    plt.figure(figsize=(9,5))
    plt.xlabel(colName[i],fontsize=13)
    plt.ylabel("Case fatality ratio",fontsize=13)
    plt.grid(1)
    plt.scatter(inputData[:,i], CFR, edgecolors=(0,0,0), s=30, c='g')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("C:/COVID-19/Figures/" + colName[i]+"_CFR.png", dpi=250)
    
## Generating the correlation figures based on the CSR
for i in range(inputData.shape[1]):
    plt.figure(figsize=(9,5))
    plt.xlabel(colName[i],fontsize=13)
    plt.ylabel("COVID-19 spreading ratio",fontsize=13)
    plt.grid(1)
    plt.scatter(inputData[:,i], CSR, edgecolors=(0,0,0), s=30, c='g')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("C:/COVID-19/Figures/" + colName[i]+"_CSR.png", dpi=250)    


