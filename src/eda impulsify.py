import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Always make it pretty.
plt.style.use('ggplot')
#need tabulate for panda markdown 10

pd.options.display.max_columns = None

#N - North East W - West M - Mid West S - South O - Other
#C - canada
states = {
        'AK': 'O',
        'AL': 'S',
        'AR': 'S',
        'AS': 'O',
        'AZ': 'W',
        'CA': 'W',
        'CO': 'W',
        'CT': 'N',
        'DC': 'N',
        'DE': 'N',
        'FL': 'S',
        'GA': 'S',
        'GU': 'O',
        'HI': 'O',
        'IA': 'M',
        'ID': 'W',
        'IL': 'M',
        'IN': 'M',
        'KS': 'M',
        'KY': 'S',
        'LA': 'S',
        'MA': 'N',
        'MD': 'N',
        'ME': 'N',
        'MI': 'W',
        'MN': 'M',
        'MO': 'M',
        'MP': 'O',
        'MS': 'S',
        'MT': 'W',
        'NA': 'O',
        'NC': 'S',
        'ND': 'M',
        'NE': 'W',
        'NH': 'N',
        'NJ': 'N',
        'NM': 'W',
        'NV': 'W',
        'NY': 'N',
        'OH': 'M',
        'OK': 'S',
        'OR': 'W',
        'PA': 'N',
        'PR': 'O',
        'RI': 'N',
        'SC': 'S',
        'SD': 'M',
        'TN': 'S',
        'TX': 'S',
        'UT': 'W',
        'VA': 'S',
        'VI': 'O',
        'VT': 'N',
        'WA': 'W',
        'WI': 'M',
        'WV': 'S',
        'WY': 'W',
        'ON': 'C',
        'NB': 'C',
        'QC': 'C',
        'AB': 'C',
        'SK': 'C',
        'NS': 'C'
}

def plot_room_distribution(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = list(range(1, 1001, 10)) 
    df.rooms.plot.hist(bins=bins, ax=ax)
    ax.set_title("Room Distribution")
    plt.tight_layout()

def plot_brand_distribution(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    df.groupby(by="flag_name").store_id.count().plot.bar(ax=ax)
    ax.set_title("Brands Distribution")
    plt.tight_layout()


def plot_boxplot_for_top5_brands(df):

    top_5 = df.groupby(by="brand_code")["property_code"].count().sort_values(ascending=False).head(5).index
    fig, ax = plt.subplots(figsize=(12, 6))
    df.loc[df['brand_code'].isin(top_5)].boxplot("spor", by="brand_code", ax=ax)
    ax.set_title("SPOR Distribution Per Top 5 brands")
    plt.tight_layout()


def run_analysis(df):
    pass

if __name__ == '__main__':
    df = pd.read_pickle("../data/data2019_monthly.pkl")

# # Room Distribution.
#     df.rooms.describe()
#     plot_room_distribution(df)

# # Brands Distribution
#     plot_brand_distribution(df)
#     print(df.groupby(by="flag_name").store_id.count().sort_values(ascending=False).to_markdown())

# # correlation between room # and brand
# df_brand_room = df.groupby(by="flag_name").agg({"store_id": "count", "rooms": np.mean}).sort_values(by="store_id")
# print(df_brand_room.corr().to_markdown())
# # Finding: # of rooms and  # of stores per flag do not show strong correlation. -0.14

# distribution of two big flags 
# Hilton Garden Inn - Room Distribution skewed right. 
# could divide to min - mean+std, mean+std - max
#df[df.flag_name=="Hilton Garden Inn"].rooms.plot.hist()
#df[df.flag_name=="Hilton Garden Inn"].rooms.describe()


    #remove Platt and Blueb, only one room 
    df[df.num_of_rooms > 10]
    spor = np.round(df.revenue / (30.62 * 0.68 * df.num_of_rooms), 2)
    spor.plot.hist()

    # add spor and brand_code
    df["spor"] = np.round(df.revenue / (30.62 * 0.68 * df.num_of_rooms), 2)
    df["brand_code"] = df.property_name.apply(lambda x: x[0:5].lower())

    #scatter spor with 5 character category
    print(df.groupby(by="brand_code")["property_code"].count().sort_values(ascending=False).head(5).to_markdown())
    top_5 = df.groupby(by="brand_code")["property_code"].count()    
    df_top5 = df.loc[df['brand_code'].isin(top_5)]
    plot_boxplot_for_top5_brands(df)

    #hilto with 158 properties, SOPR distribution
    print(df[df.brand_code=="hilto"]["spor"].describe().to_markdown())
    df[df.brand_code=="hilto"]["spor"].plot.hist()

    #hampt 78 properties
    print(df[df.brand_code=="hampt"]["spor"].describe().to_markdown())
    df[df.brand_code=="hampt"]["spor"].plot.hist()

    #tru b 76 properties
    print(df[df.brand_code=="tru b"]["spor"].describe().to_markdown())
    df[df.brand_code=="tru b"]["spor"].plot.hist()

    #compare standard deviation and mean between brand/flag and brand_code
    df_brand_flag = df.groupby(by=["brand_name", "flag_name"]).agg({"property_code": "count", "spor": [np.mean, np.std]})
    df_brand_code = df.groupby(by=["brand_code"]).agg({"property_code": "count", "spor": [np.mean, np.std]})
    print(df_brand_flag.sort_values(by=('property_code', 'count'), ascending=False).to_markdown())
    print(df_brand_code.sort_values(by=('property_code', 'count'), ascending=False).to_markdown())
    # Hilton Garden Inn need to be separated
    # inspect Double Tree distribution
    print(df[np.logical_and(df.brand_name=="Hilton", df.flag_name=="Doubletree by Hilton")].to_markdown())
    # 3 outlier with big spor. there is one embassy suite hotel mislabeled?
    # merge holiday express inn and /suites
    # one or two property brand/flag could be merged others

    brand_code_stds = df_brand_code[('spor',   'std')].to_numpy()
    brand_flag_stds = df_brand_flag[('spor',   'std')].to_numpy()
    bc = brand_code_stds[~np.isnan(brand_code_stds)]
    print(f"brand_code standard variance: {np.sum(bc)/bc.shape}")
    bf = brand_flag_stds[~np.isnan(brand_flag_stds)]
    print(f"brand_flag standard variance: {np.sum(bf)/bf.shape}")
    #brand_code and brand_flag show similiar standard variance after all
    #brand_code standard variance: [0.57874774]
    #brand_flag standard variance: [0.59978149]

    #it seems different in.
    df.groupby(by=["brand_name", "flag_name"])["spor"].hist()
    #df.groupby(by=["brand_name", "flag_name"]).hist(column="spor")
    #distribution per brand/flag is not consistent, but many of them right skewed.
    #df.groupby(by=["brand_code"]).hist(column="spor")
    #Conclusion. Eigther Brand_code or Brand_Flag do not show narrow norm distribution of 
    #spor. size of room or other feature might be better.
    # 

    # location type - overlap?
    df_location_type = df.groupby(by=["location_type"]).agg({"property_code": "count", "spor": [np.mean, np.std]})      
    print(df_location_type.to_markdown())
    df.groupby(by=["location_type"])["spor"].hist(figsize=(12,6))

    # states region - South 215. Not much difference? overlap. There is 15 Canada. Will remove?
    df["region"] = df.state.apply(lambda x: states[x])
    df_region = df.groupby(by=["region"]).agg({"property_code": "count", "spor": [np.mean, np.std]})      
    print(df_region.to_markdown())
    df.groupby(by=["region"])["spor"].hist(figsize=(12,6))


    # number of room
    # correlation with spor
    df.corr()
    