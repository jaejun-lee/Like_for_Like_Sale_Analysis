import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import importlib as imt 
import etl
import utils
import dataset
imt.reload(utils)
imt.reload(etl)
imt.reload(dataset)

from collections import defaultdict

class Cluster_brand_code(object):
    '''dummy class for Impulsify Prediction Model'''

    def __init__(self):
        self.model = defaultdict(float)

    def fit(self, X, y):
        '''cluster X by brand and average y
            -X: np array of brand_code
            -y: np array of spor
        OUTPUT: None
        '''
        d = defaultdict(list)
        for i, e in enumerate(X):
            d[e].append(y[i])
        for key, value in d.items():
            self.model[key] = np.mean(value)

    def predict(self, X):
        '''
        INPUT:
            -X: nnumpy array of brand_code 
        OUTPUT: numpy array of predicted spor
        '''
        y = []
        for e in X:
            y.append(self.model[e])
        
        return np.array(y)
            

if __name__=='__main__':

    data = dataset.Data_2019()
    data.load()
    data.clean()

    X = data.df["brand_code"].to_numpy()
    y = data.target_spor().to_numpy()

    #shuffle
    X, y = shuffle(X, y, random_state = 77)

    #split for holdout data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=77)

    reg = Cluster_brand_code()
    reg.fit(X_train, y_train)
    
    y_train_hat = reg.predict(X_train)
    y_test_hat = reg.predict(X_test)

    print(f"Train Data MSE: {mean_squared_error(y_train, y_train_hat):.2f}")
    print(f"Train Data R2: {r2_score(y_train, y_train_hat):.2f}")
    print(f"Holdout Data MSE: {mean_squared_error(y_test, y_test_hat):.2f}")
    print(f"Holdout Data R2: {r2_score(y_test, y_test_hat):.2f}")
    
    # Train Data MSE: 0.53
    # Train Data R2: 0.09
    # Holdout Data MSE: 0.48
    # Holdout Data R2: -0.14