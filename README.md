# Like for Like Sales prediction model development

# Background and Motivation
Impulsify provides its product and service for Hotel Retail/Pantry business solutions and has been growing successfully with its expertise in technology, data, and design service in this focused market. Since it's getting lots of interest from prospective customers, they like to develop an application to analyze business opportunities for candidates based on utilizing their experience and business data accumulated through years in Hotel retail field. It will help the customer make the decision quickly with confidence to adopt Impulsify business solutions. 

# Goal
As a data scientist, I like to develop a sales prediction model for future stores based on data from Impulsify and verify it's line up with their experience and intuition. 

## Initial Development Strategy 
# Data
Impulsify delivers a database snapshot from their Postgress relational database.
I will build datasets through SQL extraction for features and target(sales) focused on store properties.
Since it is initially comparable store sale analysis with the recent period(time-series or old data will not be relevant), data size could be manageable to process in a single computer. As project requirement grows, it is possible to move data processing into cloud computing like AWS. 

# Exploratory Data Analysis
Explore store properties features distribution and correlation with sales performance.

# Feature Extraction and Engineering
It seems necessary to do Principal Component Analysis or even clustering to explore features more. In contrast, feature inputs for the prediction model are limited to the characteristic of store properties such as brands, location, customer size, etc. Since I will have to examine that these features will eventually shape meaningful components to separate existing data or find the best way to represent these features.

# Prediction Model Development
Experiment with various parameters and features to find the best prediction model.

# Deliverable 
Prediction model and prototype web app to simulate its use case. 

# Challenge or Concern
It seems that basic grouping and distribution analysis with parameter adjustment could be sufficient or accepted in comparable-store sales analysis or comparable property sale analysis. I'm not sure how much extreme modeling can improve the prediction. Presumably, linear regression could be better interpretation after all.
Existing comparable stores(observations) are about 1200 in the table. I'm not sure it's enough to form a good prediction model. I will have to research more about this. 

# explain impulsify business
picture: abandoned retail section vs improved one through Impulsify solution.
good to have but not great in performance. 
ratail as secondary
improve sales, profit margin through better inventory management through data analytic, expertise consultation for store setup, and automated payment processing.

# explain the problem

## current business logic(model) to predict business opportunity of the prospective customers.
## introduce target performance metrics of SPOR and Profit Margin. Why I will focus on SPOR instead Profit Margin. 
## introduce the current feature, brand_code
## explain the opportunity to improve
## SPOR distribution whithin segment divded by brand_code.
for some brand, SPOR disrtibution rightly skewed and have long tail with outliers.
for some brand, SPOR distribution do not show any form of distribution.
standard variance within cluster are large.
from business insight from Robert, some brand could be divided by num of rooms.
by extending feature matrics by num_of_room, location_type, rigions, we could improve segmentation and find better SPOR distribution within.
## review dataset
## one hot coding of category features
## KNN approch
by grid search for stepwise method, 10 neighbors, 15 - 20 features are appropriate to improve R2 scores. However, the score is poor. By nature, KNN do not guarantee the close distance neighbors will share similiar targets. Also, real cluster could be differ dramatically between because it's business lean toward 2 or 3 brands. location_type and region is promissing, but region's distribution are also heavily toward south reigion...
## XGBoost approch with grid search of model(tree, linear) and parameters to fit better in dataset. could extract feature importance but hard to iterpret. R2, RMSE all improve.
## clustering KPrototype to find better cluster and verify SPOR distibution could improve. - not guaranteed for fitting to target.
# conclusion.


