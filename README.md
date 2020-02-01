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
