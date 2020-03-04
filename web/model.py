import pandas as pd
import numpy as np
from utils import region
import pickle
import copy

IAO = 0.68 #Industry Average Occupancy
DIM = 30.62 #Days in one month

class Store(object):
    '''
        Dictionary Like Class with business logic for SPOR, gross_profit
    '''
    def __init__(self, property_name, property_code, flag_name, num_of_rooms, 
                revenue, profit_margin, occupancy_rate = IAO, location_type = None, state = None):
        
        self.property_name = property_name
        self.property_code = property_code
        self.flag_name = flag_name
        self.num_of_rooms = num_of_rooms
        self.revenue = revenue
        self.profit_margin = profit_margin
        self.occupancy_rate = occupancy_rate
        self.location_type = location_type
        self.state = state
    
    def get_brand(self):
        return self.property_name[0:5].lower()
        
    def get_spor(self):
        return np.round(self.revenue / (self.num_of_rooms * self.occupancy_rate * DIM), decimals=2)
    
    def get_gross_profit(self):
        return np.round(self.revenue * self.profit_margin, 2)
    
    def get_region(self):
        return region(self.state) 
    
    def to_numpy(self):
        np.array([self.property_name, self.get_brand(), self.num_of_rooms, self.revenue, self.profit_margin, self.profit_margin, self.get_spor()])
        
    
    def upgrade_to(self, spor, profit_margin):
        store_hat = copy.deepcopy(self)
        store_hat.revenue = spor * self.num_of_rooms * self.occupancy_rate * DIM
        store_hat.profit_margin = profit_margin
        return store_hat
        
    def __str__(self):
      return (f"Hotel Name: {self.property_name}\n"
            f"Brand: {self.get_brand()}\n"
            f"Room Count: {self.num_of_rooms}\n"
            f"Average Occupancy: {self.occupancy_rate:.0%}\n"
            f"Current Average Monthly Retail Revenue: ${self.revenue:.2f}\n"
            f"Current Profit Margin: {self.profit_margin:.2%}\n"
            f"Current Monthly Profit: ${self.get_gross_profit():.2f}\n"
            f"Current SPOR: ${self.get_spor():.2f}")

    def __repr__(self):
      return f"{self.property_name}:{self.get_brand()}:{self.num_of_rooms}:{self.revenue}:{self.profit_margin}:{self.occupancy_rate}:{self.get_spor()}"

    # @staticmethod
    # def from_numpy(x):
    #     return Store(x[0], x[1], x[2], x[3])

class Impulsify(object):
    '''dummy class for Cluster Prediction Model'''

    def __init__(self):
        self.model = pd.read_pickle("./data/model.pkl")
        with open('./data/kproto.pkl', 'rb') as handle:
            self.kproto = pickle.load(handle)

    def predict(self, store):
        x = [store.num_of_rooms, store.flag_name, store.location_type, store.get_region()]
        X = np.array([x])
        y = self.kproto.predict(X, categorical=[1,2,3])
        return y[0]

    def predict_spor(self, store):
        cluster = self.predict(store)
        return self.model[self.model.cluster==cluster].spor.mean()
    
    def predict_profit_margin(self, store):
        cluster = self.predict(store)
        return self.model[self.model.cluster==cluster].profit_margin.mean()
    
    def predict_comparable_stores(self, store, num = 3):
        cluster = self.predict(store)
        return self.model[(self.model.cluster==cluster) & (self.model.flag_name == store.flag_name) ].head(num).to_numpy()

    