import pandas as pd
import numpy as np
import utils

def hotel_category(x):
    if x > 449:
        return 'mega'
    elif x > 162:
        return 'big'
    elif x > 98:
        return 'medium'
    else:
        return 'small'

class Data_2019(object):

    def __init__(self, file_path="../data/data2019_monthly.pkl"):
        self.df = pd.read_pickle(file_path)
    
    def clean(self):
        #remove Platt and Blueb
        self.df = self.df[self.df.num_of_rooms > 10]
    
    def load(self):
        
        
        #add brand_code
        self.df["brand_code"] = self.df.property_name.apply(lambda x: x[0:5].lower())
        #add region 
        self.df["region"] = self.df.state.apply(lambda x: utils.region(x))
        #canada 1.33 exchange rate applied
        self.df.revenue = self.df.apply(lambda x: x.revenue * 1.33 if (x.region == "C") else x.revenue, axis=1)
        self.df.gross_profit = self.df.apply(lambda x: x.gross_profit * 1.33 if (x.region == "C") else x.gross_profit, axis=1)
        #add hotel size category
        self.df["hotel_size"] = self.df.num_of_rooms.apply(lambda x: hotel_category(x))
        #add SPOR
        DIM = 30.62 #approximate days in a month
        IAO = 0.68 #Industry Average Occupancy Rate
        self.df["spor"] = self.df.revenue / (DIM * IAO * self.df.num_of_rooms)


    ### Hotcode Features. Categorized
    def fauture_brand_code(self):
        brand_codes = pd.get_dummies(self.df.brand_code)
        return brand_codes.drop('hilto', axis=1) #hilton alaway exist.

    def feature_hotel_size(self):
        category = pd.get_dummies(self.df.hotel_size)        
        return category.drop("mega", axis=1)#mega
    
    def feature_flags(self):
        flags = pd.get_dummies(self.df.flag_name)
        return flags.drop("Hilton Garden Inn", axis=1)#Always Exist
    
    def feature_region(self):
        regions = pd.get_dummies(self.df.region)
        return regions.drop('C', axis=1)#Canada

    def feature_loc_type(self):
        location_types = pd.get_dummies(self.df.location_type)
        return location_types.drop('Campus', axis=1)#Campus

    ### Continuos Value Feature
    def feature_num_of_rooms(self):
        return self.df.num_of_rooms.copy()

    ### two targets 
    def target_profit_margin(self):
        return self.df.profit_margin.copy()
    
    def target_spor(self):
        return self.df.spor.copy()


if __name__ == '__main__':
    
    data = Data_2019()
    data.load()
    data.clean()
    











