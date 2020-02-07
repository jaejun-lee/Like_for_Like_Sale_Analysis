import importlib as imt 
import etl
import utils
import dataset
imt.reload(utils)
imt.reload(etl)
imt.reload(dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Always make it pretty.
plt.style.use('ggplot')
#need tabulate for panda markdown 10

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBRegressor

def search_best_parameters(X, y):

    xgb_grid = {
        'n_estimators': [80, 100, 120],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.2, 0.5],      
        'booster' : ['gbtree', 'gblinear', 'dart'],
        'gamma': [0, 0.2, 0.5],
        'subsample': [0.5, 0.8],
        'reg_alpha': [0.2, 0.3, 0.5],
        'reg_lambda': [0.5, 0.8, 1],
        'colsample_bytree': [1, 0.8, 0.5],
        'colsample_bylevel': [1, 0.8, 0.5],
        'colsample_bynode': [1, 0.8, 0.5],
        'random_state': [77]
    }

    xgb_gridsearch = GridSearchCV(
        XGBRegressor(),
        xgb_grid,
        n_jobs=-1,
        verbose=True,
        scoring='r2'
    )

    xgb_gridsearch.fit(X, y)
    print(f"best parameters: {xgb_gridsearch.best_params_}")


if __name__=='__main__':

    
    data = dataset.Data_2019()
    data.load()
    data.clean()
    
    features = [data.feature_flags(), data.feature_num_of_rooms(), data.feature_loc_type(), data.feature_region()]
    
    X = pd.concat(features[0:4], axis=1).to_numpy()
    y = data.target_spor().to_numpy()

    #shuffle
    X, y = shuffle(X, y, random_state = 77)

    #split for holdout data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=77)


    clf = XGBRegressor(
        objective='reg:squarederror',
        booster = "dart",
        colsample_bylevel = 0.8, 
        colsample_bynode = 0.8,
        colsample_bytree =0.8,
        gamma = 0.5, 
        learning_rate = 0.1, 
        max_depth = 3, 
        n_estimators = 80, 
        reg_alpha = 0.3,
        reg_lambda = 0.8, 
        subsample = 0.5
    )

    clf.fit(X_train, y_train)
    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)

    print(f"Train Data MSE: {mean_squared_error(y_train, y_train_hat):.2f}")
    print(f"Train Data R2: {r2_score(y_train, y_train_hat):.2f}")
    print(f"Holdout Data MSE: {mean_squared_error(y_test, y_test_hat):.2f}")
    print(f"Holdout Data R2: {r2_score(y_test, y_test_hat):.2f}")
    # Train Data MSE: 0.34
    # Train Data R2: 0.42
    # Holdout Data MSE: 0.55
    # Holdout Data R2: -0.28

    #search_best_parameters(X_train, y_train)
    #best parameters: {'booster': 'dart', 'colsample_bylevel': 0.8, 'colsample_bynode': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 80, 'random_state': 77, 'reg_alpha': 0.3, 'reg_lambda': 0.8, 'subsample': 0.5}

    

    