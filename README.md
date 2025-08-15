 Overview
This project analyzes e-commerce user behavior to:  
- Predict the next page a user visits (classification).  
- Estimate potential revenue (regression).  
- Segment users based on browsing behavior (clustering).  

It also includes an interactive Streamlit app for real-time predictions and visualizations.  



Dataset
- CSV files: `train_data.csv` and `test_data.csv`  
- Key columns: `year, month, day, order, country, session_id, page1_main_category, page2_clothing_model, colour, location, model_photography, price, price_2, page`  



Preprocessing & Features
- One-Hot Encoding for categorical features.  
- StandardScaler for numeric features.  
- Feature engineering: click count, bounce indicator, last page flag, click paths.  
- Removed leakage column `page2_clothing_model`.  



Models:
Classification:
- Logistic Regression, Decision Tree, Random Forest, XGBoost  

Regression:
- Linear, Ridge, Lasso, Gradient Boosting  

Clustering:
- K-Means, DBSCAN, Hierarchical  

Pipelines
- Automate preprocessing, scaling, training, hyperparameter tuning, and evaluation.  



Streamlit App
- Upload CSV or input data manually.  
- Real-time predictions and revenue estimation.  
- Visualize user segments and session metrics.  

