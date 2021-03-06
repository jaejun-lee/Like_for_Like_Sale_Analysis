# Like for Like Sales prediction model development
*Capstone II Project for Galvanize Data Science Immersive, Week 8*

*by Jaejun Lee*

<p align="center">
  <img src="image/impulsify_store" width = 800>
</p>

# Business Explained
Impulsify provides its solution to Retail as Secondary business such as Hotel Pantry, Gym retail section, Apartment Convenience store, etc. Those are good to have, but usually not that great in terms of management overhead and financial balance sheet. Impulsify has proven that they can improve sales and operational efficiency through sales/inventory data analytic, automated payment system and it's expertise design and setup.

# What's going on.
After all, it's getting lots of interest from prospective customers. They like to develop a model to analyze sales data from the current stores and guide the business opportunity of future stores. It will help the customer make the business decision quickly with confidence to adopt Impulsify solutions for their stores.

# Data
They provided snapshots (dump) from Postgres Database and an excel spreadsheet with aggregated sales and profit margin for each store for the year 2019. I combined them into Pandas DataFrame.
There are about 460 store observations for last year. It's a clean dataset from the established business. 

|     | property_name                                | property_code   | brand   |   num_of_rooms |   revenue |   profit_margin |   gross_profit |   id | address              | city        | state   |   zip | kind           | time_zone                  | location_type   |   flag_id | flag_name         |   brand_id | brand_name   | brand_code   | region   | hotel_size   |     spor |
|----:|:---------------------------------------------|:----------------|:--------|---------------:|----------:|----------------:|---------------:|-----:|:---------------------|:------------|:--------|------:|:---------------|:---------------------------|:----------------|----------:|:------------------|-----------:|:-------------|:-------------|:---------|:-------------|---------:|
| 122 | Hampton Inn Henderson                        | HDSNC           | Hampt   |             75 |   1107.9  |        0.552091 |        611.933 | 6951 | 385  Ruin Creek Road | Henderson   | NC      | 27536 | Select Service | Eastern Time (US & Canada) | Suburban        |        31 | Hampton Inn       |          1 | Hilton       | hampt        | S        | small        | 0.709458 |
| 228 | Hilton Garden Inn Knoxville West Cedar Bluff | TYSKH           | Hilto   |            118 |   2764.38 |        0.672252 |       1856.24  | 5941 | 216 Peregrine Way    | Knoxville   | TN      | 37922 | Select Service | Eastern Time (US & Canada) | Urban           |         1 | Hilton Garden Inn |          1 | Hilton       | hilto        | S        | medium       | 1.12513  |
| 188 | Hilton Garden Inn Columbus/Edinburgh         | CLUIN           | Hilto   |            125 |   2178.22 |        0.58644  |       1280.94  | 7099 | 12210 N Executive Dr | Edinburgh   | IN      | 46124 | Select Service | Central Time (US & Canada) | Interstate      |         1 | Hilton Garden Inn |          1 | Hilton       | hilto        | M        | medium       | 0.836908 |
| 299 | Hilton Garden Inn Washington DC Downtown     | DCACH           | Hilto   |            300 |   5391.56 |        0.688213 |       3733.33  | 6438 | 815 14th Street N.W. | Washington  | DC      | 20005 | Select Service | Eastern Time (US & Canada) | Urban           |         1 | Hilton Garden Inn |          1 | Hilton       | hilto        | N        | big          | 0.863136 |
| 453 | Tru by Hilton San Antonio Downtown           | SATDO           | Tru b   |             94 |   4412.2  |        0.578319 |       2562.1   | 6853 | 901 E. Houston St    | San Antonio | TX      | 78205 | Select Service | Central Time (US & Canada) | Urban           |       147 | Tru by Hilton     |          1 | Hilton       | tru b        | S        | small        | 2.25431  |

# Target, business performance metrics
> SPOR: Sales per Occupied Room = Monthly Revenue / (30.62 * Average Occupancy Rate)

> Profit Margin: 1 - Monthly Cost(Cost of Good Sold)/Monthly Revenue

<div style="display:flex">
     <div style="flex:1;padding-right:5px;">
          <img src="image/plot_spor_distribution.png">
     </div>
     <div style="flex:1;padding-left:5px;">
          <img src="image/plot_profit_distribution.png">
     </div>
</div>

|    |SPOR | Profit Margin
|---    |---    |---
|Mean    |1.23  |0.61
|STD  |0.7383    |0.0647

# The current model solution in house
> Brand Code Clustering
<p align="center">
  <img src="image/brand_code_model.png" width = 800>
</p>

<b>INPUT to Model</b>

Hotel Name: Crowne Plaza Chicago O’Hare Hotel & Conference Center
Brand: crown
Room Count: 503
Average Occupancy: 85%
Current Average Monthly Retail Revenue: $6545.79
Current Profit Margin: 44.00%
Current Monthly Profit: $2880.15
Current SPOR: $0.50

<b>OUPUT from Model</b>
Hotel Name: crown
Brand: crown
Room Count: 293.0
Average Occupancy: 68%
Current Average Monthly Retail Revenue: $9181.66
Current Profit Margin: 61.29%
Current Monthly Profit: $5627.44
Current SPOR: $1.51

<b>Adapt the business</b> 
Hotel Name: Crowne Plaza Chicago O’Hare Hotel & Conference Center
Brand: crown
Room Count: 503
Average Occupancy: 85%
Current Average Monthly Retail Revenue: $19768.29
Current Profit Margin: 61.29%
Current Monthly Profit: $12115.98
Current SPOR: $1.51

## Analysis on brand_code

| brand_code   | # of properties |
|:-------------|----------------:|
| hilto        |             158 |
| hampt        |              78 |
| tru b        |              76 |
| homew        |              38 |
| crown        |              25 |


<p align="center">
  <img src="image/spor_top5_brand.png" width = 800>
</p>
<p align="center">
  <img src="image/spor_hilton.png" width = 800>
</p>
<p> hilton SPOR Distribution </p>
<p align="center">
  <img src="image/spor_hampton.png" width = 800>
</p>
<p> hampt SPOR Distribution </p>

<p align="center">
  <img src="image/spor_trub.png" width = 800>
</p>
<p> tru b SPOR Distribution </p>

<p> brand_code is easy to interpret but, SPOR distribution within does not form a pattern for several clusters. The largest cluster, hilto, is right-skewed with a long tail. There needs to sub-cluster to improve SPOR distribution.</P>

## Prediction Performance of brand_code

| Score Metrics   |  SPOR prediction Score       |
|:-------------|----------------:|
| Train Data MSE  |           0.53 |
| Train Data R2      |       0.09 |
| Holdout Data MSE    |     0.48 |
| Holdout Data R2     |     -0.14 |

* For this project, I will try to find better segmentation of stores by extending features and different models. Also, I will check whether it will improve the prediction of SPOR.
* I will use flag_name from properties table as brand feature instead of brand_code.

# Additional Features: num_of_rooms, location_type, region

## location_type

| location_type   |   ('property_code', 'count') |   ('spor', 'mean') |   ('spor', 'std') |
|:----------------|-----------------------------:|-------------------:|------------------:|
| Airport         |                           47 |            1.20918 |          0.551063 |
| Campus          |                            8 |            1.27841 |          1.12948  |
| Interstate      |                          104 |            1.05272 |          0.555151 |
| Resort          |                           13 |            1.39664 |          1.08252  |
| Suburban        |                          145 |            1.12662 |          0.563088 |
| Urban           |                          149 |            1.45476 |          0.924047 |

## region

| region   |   ('property_code', 'count') |   ('spor', 'mean') |   ('spor', 'std') |
|:---------|-----------------------------:|-------------------:|------------------:|
| C        |                           15 |          1.08526   |          0.431939 |
| M        |                           69 |          1.18953   |          0.681721 |
| N        |                           69 |          1.42968   |          1.09299  |
| O        |                            1 |          0.0856803 |        nan        |
| S        |                          215 |          1.15481   |          0.666546 |
| W        |                           97 |          1.33447   |          0.614003 |


* N - North East, W - West, M - Mid West, S - South, O - Other, C - canada

## num_of_rooms
correlation with SPOR: 0.10
* num of rooms is already applied in SPOR negatively, but there still remains small positive correlation with SPOR. I made decision to include it as a feature.


# KNN approch

* KNN find close distance stores from target store and calculate the average of SPOR.

## Choose 15 neighbors by CV
* Through cross valdiation, I found that 15 of close neighbors are appropriate to caculate the average.
<div style="display:flex">
     <div style="flex:1;padding-right:5px;">
          <img src="image/plot_mse_for_knn.png">
     </div>
     <div style="flex:1;padding-left:5px;">
          <img src="image/r2_cv_knn.png">
     </div>
</div>

## by grid search for regressive elimination of feature (Step Backward), 
* 15 - 20 features are appropriate to improve MSE or R2 scores. However, the score does not improve from brand_code cluster method. The close neighbors seem not to share the same interests in target, SPOR. Also, the real cluster could differ dramatically because most of the brands lean toward 2 or 3 brands. location_type and region are promising, but the region's distribution is also heavily toward the south region.

* By example, with 24 features eliminated from 40 features, we have an MSE of 0.878 ['Crowne Plaza Hotels and Resorts' 'Doubletree by Hilton' 'Hampton Inn'\n 'Hampton Inn and Suites' 'Home2' 'Homewood Suites' 'Independent' 'room' 'Airport' 'Interstate' 'Suburban' 'Urban' 'M' 'N' 'S' 'W']

* I could observe location type, region, number of rooms are still survive after eliminating 24 features. Several brand name disappear in the features.


| Score Metrics   |  Score       |
|:-------------|----------------:|
| Train Data MSE  |           0.51 |
| Train Data R2      |       0.12 |
| Holdout Data MSE    |     0.47 |
| Holdout Data R2     |     -0.11 |

* SPOR prediction does not improve.


# XGBoost approch with grid search of model(tree, linear) and parameters to fit better in dataset.

## Parameter selected after Grid Search
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

## Score
| Score Metrics   |  Score       |
|:-------------|----------------:|
| Train Data MSE  |           0.34 |
| Train Data R2      |       0.42 |
| Holdout Data MSE    |     0.55 |
| Holdout Data R2     |     -0.28 |

* Dart is improved decision tree model. Still it does not improve SPOR prediction in visible. 


# Clustering by KPrototype(Kmean)
*  K-Prototype allow to find clusters from mixed features with numerical(number of rooms) and categorical(brand, location type, region). 

## K = 13 is best by plotting Cost function.

<p align="center">
  <img src="image/kprototype_costs.png" width = 800>
</p>


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

* Finding 13 of clusters is best based on elbow plot method. 
* However, silhouette_score could not be used. Kprototype Clustering return numerical and categorical together. This plot is based on cost(datapoint distances from centroid in a cluster).


## Centeroid

|rooms | flags | region | location type|
|:-------------|----------------|----|----:|
| 151.24 | Hilton Garden Inn | W | Interstate |
| 311.81 | Crowne Plaza Hotels and Resorts | S | Urban |
| 121.38 | Hilton Garden Inn | S | Suburban |
| 825.67 | Holiday Inn Express | S | Campus |
| 92.49 | Tru by Hilton | S | Suburban |
| 423.33 | Crowne Plaza Hotels and Resorts | N | Urban |
| 216.44 | Doubletree by Hilton | W | Urban |
| 105.12 | Hilton Garden Inn | S | Suburban |
| 245.11 | Embassy Suites | S | Urban |
| 197.76 | Crowne Plaza Hotels and Resorts | S | Urban |
| 172.96 | Hilton Garden Inn | S | Urban |
| 75.26 | Hampton Inn | S | Interstate |
| 135.54 | Hilton Garden Inn | S | Urban |

* There is total 13 clusters found and the above is list of the closest datapoint for each cluster.
* We can see that "Hilton Garden Inn" is further segmented by size, location type and region.

## Convert Clustering as prediction model. mean of spors in each cluster. Score....

| Score Metrics   |  Score       |
|:-------------|----------------:|
| Train Data MSE  |           0.55 |
| Train Data R2      |       0.06 |
| Holdout Data MSE    |     0.41 |
| Holdout Data R2     |     0.04 |

* Still prediction of SPOR is not impoved alot.

## Use case
* You can connect to two page simple web app to show use case of clustering model.
* It will be available at http://54.160.190.65:8105/ for a week.
* submit store page is pre-filled with Crown Plaza hotel information for testing. 

<p align="center">
  <img src="image/submitstore.png" width = 800>
</p>
<p align="center">
  <img src="image/compare_stores.png" width = 800>
</p>


# Conculsions and Recommedation
* There is not much improvement in SPOR prediction even employing complicated models.
* However, K-Prototype is promising to find better clustering for current stores.
* I could not try various parameters and initialization methods for K Prototype. We could find even better clustering by examining various custering results with business insigts.
* The other clustering model such as "clustering hierarchical" could yield better view of how stores are devided. It is feasible with this small dataset.
* For clustering model, it seems better to filter out outcriers more such as SPOR < 50 or SPOR >2.50. 
* If we can obtain the metric of how much the current store improved after Impulsify solution, revenue could be included in the feature set.  
* Single page app will be better flow for this requirement. There is no input validation, guide, detailed business logic in this prototype.
