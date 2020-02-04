import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Always make it pretty.
plt.style.use('ggplot')
#need tabulate for panda markdown 10

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

if __name__ == '__main__':
    df = pd.read_pickle("../data/data.pkl")

# Room Distribution.
    df.rooms.describe()
    plot_room_distribution(df)

# Brands Distribution
    plot_brand_distribution(df)
    print(df.groupby(by="flag_name").store_id.count().sort_values(ascending=False).to_markdown())

# correlation between room # and brand
df_brand_room = df.groupby(by="flag_name").agg({"store_id": "count", "rooms": np.mean}).sort_values(by="store_id")
print(df_brand_room.corr().to_markdown())
# Finding: # of rooms and  # of stores per flag do not show strong correlation. -0.14

# distribution of two big flags 
# Hilton Garden Inn - Room Distribution skewed right. 
# could divide to min - mean+std, mean+std - max
#df[df.flag_name=="Hilton Garden Inn"].rooms.plot.hist()
#df[df.flag_name=="Hilton Garden Inn"].rooms.describe()

# 
