import numpy as np
from kmodes.kprototypes import KPrototypes

import importlib as imt 
import etl
import utils
import dataset
imt.reload(utils)
imt.reload(etl)
imt.reload(dataset)

import pandas as pd
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

from sklearn.metrics import silhouette_score

from collections import defaultdict

def plot_costs(X, min_k, max_k):
    """Plots sse for values of k between min_k and max_k
    Args:
    - X - feature matrix
    - min_k, max_k - smallest and largest k to plot sse for
    return: list of costs
    """
    k_values = range(min_k, max_k+1)
    costs = []
    for k in k_values:
        kp = KPrototypes(n_clusters = k, init = 'Cao', n_init =22, verbose = 0, random_state=4, n_jobs=4 ) 
        clusters=kp.fit_predict(X, categorical = [1,2,3])
        costs.append(kp.cost_)
    plt.plot(k_values, costs)
    plt.xlabel('k')
    plt.ylabel('costs')
    plt.show()
    plt.savefig("../image/kprototype_costs.png")
    return costs

class Cluster_Regression(object):
    '''dummy class for Cluster Prediction Model'''

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

    features = ['num_of_rooms', 'flag_name', 'region', 'location_type']
    X = data.df[features].to_numpy()
    y = data.target_spor().to_numpy()
    #y = data.df["revenue"].to_numpy()

    #shuffle
    #X, y = shuffle(X, y, random_state = 77)

    #split for holdout data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    kproto = KPrototypes(n_clusters = 13, init = 'Cao', n_init =22, verbose = 1, random_state=4, n_jobs=4) 
    clusters = kproto.fit_predict(X, categorical=[1,2,3])


    ### ran Kprototype cost function from 10 to 20. 
    ### ***silhouette_score could not be used. clustering X value return numerical and categorical together
    #plot_costs(X, 10, 20)
    ### after 13, costs elbow down.
    #  [169655.69381188636,
    #  164364.36907638382,
    #  161368.5891212315,
    #  158775.1149836406, <------ k = 13
    #  88726.7001211234,
    #  87130.27749468954,
    #  83816.32073226149,
    #  83332.45085480264,
    #  82269.60881991593,
    #  78920.15376932337,
    #  79341.07766656693]

    #develpe prediction model returning score mean of cluster.
    
    print(kproto.cluster_centroids_)
    df = data.df.copy()
    df['cluster'] = clusters
    
    cluster_spor_std = df.groupby(by="cluster")["spor"].std().to_numpy()
    cluster_spor_mean = df.groupby(by="cluster")["spor"].mean().to_numpy()
    cluster_profit_mean = df.groupby(by="cluster")["profit_margin"].mean().to_numpy()
    #
    #cluster_hat = kproto.predict(np.array(x), categorical=[1,2,3])
    # print(f"predicted_spor: {cluster_spor_mean[cluster_hat[0]]:.2f}")
    # #1.7159
    # print(f"predicted_profit: {cluster_profit_mean[cluster_hat[0]]:.2f}")
    # #0.64
    #print(f"standard variance: {np.sum(cluster_std)/cluster_std.shape}")
    #0.7571044
    
    #### Test cluster Prediction
    X = df["cluster"].to_numpy()
    y = data.target_spor().to_numpy()
    #y = df["revenue"].to_numpy()

    #shuffle
    X, y = shuffle(X, y, random_state = 77)

    #split for holdout data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=77)

    reg = Cluster_Regression()
    reg.fit(X_train, y_train)
    
    y_train_hat = reg.predict(X_train)
    y_test_hat = reg.predict(X_test)

    print(f"Train Data MSE: {mean_squared_error(y_train, y_train_hat):.2f}")
    print(f"Train Data R2: {r2_score(y_train, y_train_hat):.2f}")
    print(f"Holdout Data MSE: {mean_squared_error(y_test, y_test_hat):.2f}")
    print(f"Holdout Data R2: {r2_score(y_test, y_test_hat):.2f}")




["crown", 503, 6545.79, 0.44]


    
    #compare r2 and mse for test data

#     Best run was number 8
# [array([[151.23809524],
#        [311.80952381],
#        [121.38157895],
#        [825.66666667],
#        [ 92.49      ],
#        [423.33333333],
#        [216.4375    ],
#        [105.12345679],
#        [245.10526316],
#        [197.76470588],
#        [172.96296296],
#        [ 75.25806452],
#        [135.53846154]]), array([['Hilton Garden Inn', 'W', 'Interstate'],
#        ['Crowne Plaza Hotels and Resorts', 'S', 'Urban'],
#        ['Hilton Garden Inn', 'S', 'Suburban'],
#        ['Holiday Inn Express', 'S', 'Campus'],
#        ['Tru by Hilton', 'S', 'Suburban'],
#        ['Crowne Plaza Hotels and Resorts', 'N', 'Urban'],
#        ['Doubletree by Hilton', 'W', 'Urban'],
#        ['Hilton Garden Inn', 'S', 'Suburban'],
#        ['Embassy Suites', 'S', 'Urban'],
#        ['Crowne Plaza Hotels and Resorts', 'S', 'Urban'],
#        ['Hilton Garden Inn', 'S', 'Urban'],
#        ['Hampton Inn', 'S', 'Interstate'],
#        ['Hilton Garden Inn', 'S', 'Urban']], dtype='<U31')]
# Train Data MSE: 0.55
# Train Data R2: 0.06
# Holdout Data MSE: 0.41
# Holdout Data R2: 0.04






#df.groupby(by="cluster").hist(["spor"])


# import pickle

# with open('../data/kproto.pkl', 'wb') as handle:
#     pickle.dump(kproto, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('../data/kproto.pkl', 'rb') as handle:
#     b_kproto = pickle.load(handle)

# print (kproto == b_kproto)