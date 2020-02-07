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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from itertools import combinations
from copy import deepcopy
from sklearn.metrics import r2_score
import quantecon as qe

pd.options.display.max_columns = None


def Eliminate_One(reg, X, y, included_features):
    '''
    Eliminate One
    For a provided regressor, evaluate the ideal subset of n-1 features to try from 
    a provided list of n features.  This function can be called itterative to perform
    recursive feature elimination for a general regression model (not leveraging feature
    importances).
    
    Inputs - 
    * reg = a regression model matching the sklearn mechanics
    * X, y = numpy arrays for the features and targets
    * included_features = numpy array indicating the column indexes to subset from
    '''
    best_r2 = 0 #-1
    best_reg = deepcopy(reg)
    best_features = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
    print("*** Trying {} combinations of features.".format(included_features.shape[0]-1))
    for feats in combinations(included_features, included_features.shape[0]-1):
        reg.fit(X_train[:, feats], y_train)
        r2 = mean_squared_error(y_test, reg.predict(X_test[:, feats]))
        if r2 > best_r2:
            best_r2 = r2
            best_reg = deepcopy(reg)
            best_features = np.array(feats)
    return best_reg, best_r2, best_features


def plot_CV_rmse(X_train, y_train):
    cv_scores = []
    for n in range(2, 50):
        reg = KNeighborsRegressor(n_neighbors=n)
        train_errors = cross_val_score(reg, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
        cv_scores.append(np.negative(np.mean(train_errors)))
    plt.plot(list(range(2, 50)),cv_scores)
    plt.xlabel('n_neighbors')
    plt.ylabel('rmse')
    plt.title('average test rmse n_neighbors n')

def plot_CV_r2(X_train, y_train):
    cv_scores = []
    for n in range(2, 50):
        reg = KNeighborsRegressor(n_neighbors=n)
        train_errors = cross_val_score(reg, X_train, y_train, scoring='r2', cv=5)
        cv_scores.append(np.mean(train_errors))
    plt.plot(list(range(2, 50)),cv_scores)
    plt.xlabel('n_neighbors')
    plt.ylabel('R2')
    plt.title('average test R2 n_neighbors n')

if __name__ == '__main__':

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

    #Standardize X_train
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    #X_train = scaler.inverse_transform(X_train)

    #Cross Validation
    # reg = KNeighborsRegressor(n_neighbors=10)
    # scoring = 'neg_root_mean_squared_error' #"neg_root_mean_squared_error"
    # train_errors = cross_val_score(reg, X_train, y_train, scoring=scoring, cv=5)
    # print(np.mean(train_errors))
    
    #Hold Out Validation
    X_test = scaler.transform(X_test)
    reg = KNeighborsRegressor(n_neighbors=10)
    reg.fit(X_train, y_train)
    y_train_hat = reg.predict(X_train)
    y_test_hat = reg.predict(X_test)

    print(f"Train Data MSE: {mean_squared_error(y_train, y_train_hat):.2f}")
    print(f"Train Data R2: {r2_score(y_train, y_train_hat):.2f}")
    print(f"Holdout Data MSE: {mean_squared_error(y_test, y_test_hat):.2f}")
    print(f"Holdout Data R2: {r2_score(y_test, y_test_hat):.2f}")
    # Train Data MSE: 0.51
    # Train Data R2: 0.12
    # Holdout Data MSE: 0.49
    # Holdout Data R2: -0.16


    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    #rmse = np.zeros(n_features)
    r2_scores = np.zeros(n_features)
    #adj_r2_scores = np.zeros(n_features)
    feats = np.array(list(range(n_features)))

    # Hint: consider debugging with 3 steps instead of all 100.
    # N.B. this is a brute force approach, and will take a long time.  This is a nice demonstration
    # of the practical drawbacks to KNN.  
    for p in range(1, 25):
        qe.util.tic()
        reg, r2, feats = Eliminate_One(KNeighborsRegressor(n_neighbors=10), X_train, y_train, feats)
        qe.util.toc()
        #adj_r2 = 1 - ((1 - r2)*(n_samples - 1)/(n_samples - (n_features - p) - 1))
        r2_scores[p-1] = r2
        #adj_r2_scores[p-1] = adj_r2
        print("With {0} features eliminated we have an MSE of {1:.3f}".format(p, r2))

    plt.plot(np.arange(1,n_features +1), r2_scores)
    plt.title('Recursive Feature Elimination Mean Square Error')
    plt.xlabel('Number of Removed Features')
    plt.ylabel('MSE');

    # plt.plot(np.arange(1,n_features +1), adj_r2_scores)
    # plt.title('Recursive Feature Elimination Adjusted Explained Variance')
    # plt.xlabel('Number of features')
    # plt.ylabel('Adjusted R^2');

    # 10 - 20 features show improved R2. (Not really good though)
    # However, those features are....
    
    a = data.feature_flags().columns.to_numpy()
    b = data.feature_loc_type().columns.to_numpy()
    c = data.feature_region().columns.to_numpy()
    feature_names = np.concatenate([a, ["room"], b, c])
    print(feature_names[feats])
    

    '''    
    Adjusted R2 Metric Used.
    array(['Aston', 'Crowne Plaza Hotels and Resorts', 'Curio by Hilton',
       'Hampton Inn', 'Hampton Inn and Suites', 'Hilton',
       'Holiday Inn Express', 'Holiday Inn Express & Suites', 'Home2',
       'Homewood Suites', 'La Quinta Inns and Suites', 'Tru by Hilton',
       'room', 'Airport', 'Resort', 'Suburban', 'Urban', 'M', 'N', 'S',
       'W'], dtype=object)
    '''    
    
    
    '''
    MSE metric used
    ['Avid' 'Best Western Plus' 'Candlewood Suites'
 'Crowne Plaza Hotels and Resorts' 'Doubletree by Hilton' 'Embassy Suites'
 'Hampton Inn' 'Hampton Inn and Suites' 'Hilton' 'Hilton Suites'
 'Holiday Inn Express' 'Home2' 'Homewood Suites' 'Independent'
 'Quality Suites' 'Renaissance' 'Tru by Hilton' 'room' 'Airport'
 'Interstate' 'Suburban' 'Urban' 'M' 'N' 'S' 'W']
 '''
'''
 With 37 features eliminated we have an MSE of 0.867
['Doubletree by Hilton' 'Hampton Inn and Suites' 'Suburban']
'''
'''
With 24 features eliminated we have an MSE of 0.878
['Crowne Plaza Hotels and Resorts' 'Doubletree by Hilton' 'Hampton Inn'
 'Hampton Inn and Suites' 'Home2' 'Homewood Suites' 'Independent' 'room'
 'Airport' 'Interstate' 'Suburban' 'Urban' 'M' 'N' 'S' 'W']
'''
    

    










