import pandas as pd
import numpy as np

IAO = 0.68 #Industry Average Occupancy
DIM = 30.62 #Days in one month

class Store(object):
    '''
        Dictionary Like Class with business logic for SPOR, gross_profit
    '''

    def __init__(self, property_name, num_of_rooms, revenue, profit_margin, average_occupancy = IAO):
        
        self.property_name = property_name
        self.num_of_rooms = num_of_rooms
        self.revenue = revenue
        self.profit_margin = profit_margin
        self.average_occupancy = average_occupancy

    def get_brand(self):
        return self.property_name[0:5]
        
    def get_spor(self):
        return np.round(self.revenue / (self.num_of_rooms * self.average_occupancy * DIM), decimals=2)
    
    def get_gross_profit(self):
        return np.round(self.revenue * self.profit_margin, 2)
    
    def to_numpy(self):
        np.array([self.property_name, self.get_brand(), self.num_of_rooms, self.revenue, self.profit_margin, self.profit_margin, self.get_spor()])

    def upgrade_to(self, y):
        revenue = y.get_spor() * self.num_of_rooms * self.average_occupancy * DIM
        return Store(self.property_name, self.num_of_rooms, np.round(revenue, 2), y.profit_margin, self.average_occupancy)
        
    def __str__(self):
      return (f"Hotel Name: {self.property_name}\n"
            f"Brand: {self.get_brand()}\n"
            f"Room Count: {self.num_of_rooms}\n"
            f"Average Occupancy: {self.average_occupancy:.0%}\n"
            f"Current Average Monthly Retail Revenue: ${self.revenue:.2f}\n"
            f"Current Profit Margin: {self.profit_margin:.2%}\n"
            f"Current Monthly Profit: ${self.get_gross_profit():.2f}\n"
            f"Current SPOR: ${self.get_spor():.2f}")

    def __repr__(self):
      return f"{self.property_name}:{self.get_brand()}:{self.num_of_rooms}:{self.revenue}:{self.profit_margin}:{self.average_occupancy}:{self.get_spor()}"

    @staticmethod
    def from_numpy(x):
        return Store(x[0], x[1], x[2], x[3])

    

class Impulsify(object):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        '''cluster X by brand and prepare average revenue and profit margin
        INPUT:
            -X: Pandas Datafram - Year 2019 Sales Data
            -y: Ingored
        OUTPUT: None
        '''
        #rewrite brand for correct brand category
        X = X.copy()
        X.brand = X.property_name.apply(lambda x: x[0:5].lower())
        self.X = X.groupby(by="brand").agg({"num_of_rooms": np.mean, "revenue" : np.mean, "profit_margin" : np.mean}).reset_index()
        self.X = self.X.round(decimals = {"num_of_rooms": 0, "revenue" : 2, "profit_margin" : 4})
        #self.X["spor"] = np.round(self.X.revenue/(self.X.num_of_rooms * 0.68 * 30.62), decimals=2)


    def predict(self, X):
        '''
        INPUT:
            -X: list of store 
        OUTPUT: list of store - comparable stores
        '''
   
    def predict_one(self, x):
        '''
        INPUT:
            -x: store
        OUTPUT: comparable store
        '''
        return Store.from_numpy(self.X[self.X.brand == x.get_brand()].to_numpy()[0])
    
    
if __name__ == '__main__':

    df_X = pd.read_pickle("/home/jaejun/Dropbox/dsi/capstone/implusify/data/data2019.pkl")
    x = Store.from_numpy(["crown", 503, 6545.79, 0.44])
    x.property_name = "Crowne Plaza Chicago O’Hare Hotel & Conference Center"
    x.average_occupancy = 0.85
    model = Impulsify()
    model.fit(df_X)
    y = model.predict_one(x)
    z = x.upgrade_to(y)

