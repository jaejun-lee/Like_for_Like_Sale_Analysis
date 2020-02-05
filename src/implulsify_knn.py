import importlib as imt 
import etl
import utils
imt.reload(utils)
imt.reload(etl)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Always make it pretty.
plt.style.use('ggplot')
#need tabulate for panda markdown 10

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = None

if __name__ == '__main__':
    df = pd.read_pickle("../data/data2019_monthly.pkl")

    #remove Platt and Blueb, only one room 
    df[df.num_of_rooms > 10]
    spor = np.round(df.revenue / (30.62 * 0.68 * df.num_of_rooms), 2)
    spor.plot.hist()

    # add spor and brand_code
    df["spor"] = np.round(df.revenue / (30.62 * 0.68 * df.num_of_rooms), 2)
    df["brand_code"] = df.property_name.apply(lambda x: x[0:5].lower())

    # add region
    df["region"] = df.state.apply(lambda x: utils.region(x))

    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    #pd.get_dummies(df, columns=["flag_name"])

    df_flags = pd.get_dummies(df.flag_name)
    df_region = pd.get_dummies(df.region)
    df_type = pd.get_dummies(df.location_type)

    df_X = pd.concat([df[["num_of_rooms", "spor"]], df_flags], axis=1)
    df_y = df_X.pop("spor")

    X = df_X.to_numpy()
    y = df_y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train. y_train)
    y_hat = knn.predict(X_test)
    
    print(mean_squared_error(y_test, y_hat))

