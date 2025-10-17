import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

# Create models folder if not exists

os.makedirs("models", exist_ok=True)

#  Load Data

data_path = "data/train_data_aggregated.xlsx"
df = pd.read_excel(data_path)


#  Features & Targets

feature_cols = [
    'num_colors_clicked', 'total_clicks', 'revisit_count', 'last_page_visited',
    'min_price', 'bounce', 'max_price', 'exit_rate',
    'last_category_clicked', 'most_visited_category'
]

y_clf = df['purchase_complete']  # Classification target
y_reg = df['revenue']            # Regression target

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    df[feature_cols], y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    df[feature_cols], y_reg, test_size=0.2, random_state=42
)


#  Preprocessing

categorical_features = ['last_category_clicked', 'most_visited_category']
numeric_features = list(set(feature_cols) - set(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)


#  Classification Models

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Neural Network": MLPClassifier(max_iter=200, random_state=42)
}

# Lightweight grids for fast tuning
param_grids = {
    "Logistic Regression": {'classifier__C':[0.5, 1]},
    "Decision Tree": {'classifier__max_depth':[3,5], 'classifier__min_samples_leaf':[10,20]},
    "Random Forest": {'classifier__n_estimators':[30,50], 'classifier__max_depth':[3,5]},
    "XGBoost": {'classifier__max_depth':[2,3], 'classifier__n_estimators':[30,50]},
    "Neural Network": {'classifier__hidden_layer_sizes':[(16,8),(8,4)], 'classifier__alpha':[0.01,0.001]}
}

print("\n===== Classification Results =====")
for name, model in clf_models.items():
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    grid = GridSearchCV(pipe, param_grids[name], cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train_clf, y_train_clf)
    y_pred = grid.predict(X_test_clf)
    
    print(f"\n--- {name} ---")
    print(f"Best Params: {grid.best_params_}")
    print(f"Test Accuracy: {accuracy_score(y_test_clf, y_pred):.4f}")
    print(f"Test F1 Score: {f1_score(y_test_clf, y_pred):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test_clf, grid.predict_proba(X_test_clf)[:,1]):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test_clf, y_pred))
    
    #  Save the full classification pipeline
    joblib.dump(grid.best_estimator_, f"models/{name.replace(' ','_')}_clf_pipeline.pkl")


#  Regression Models

reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
}

print("\n===== Regression Results =====")
for name, model in reg_models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train_reg, y_train_reg)
    y_pred = pipe.predict(X_test_reg)
    
    mae = mean_absolute_error(y_test_reg, y_pred)
    mse = mean_squared_error(y_test_reg, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred)
    
    print(f"\n--- {name} ---")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    #  Save the full regression pipeline
    joblib.dump(pipe, f"models/{name.replace(' ','_')}_reg_pipeline.pkl")

#  Clustering Models

print("\n===== Clustering Results =====")
X_cluster = preprocessor.fit_transform(df[feature_cols])

cluster_models = {
    "KMeans": KMeans(n_clusters=4, random_state=42),
    "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
    "Hierarchical": AgglomerativeClustering(n_clusters=4)
}

for name, model in cluster_models.items():
    labels = model.fit_predict(X_cluster)
    if len(set(labels)) > 1:
        sil_score = silhouette_score(X_cluster, labels)
        db_score = davies_bouldin_score(X_cluster, labels)
    else:
        sil_score, db_score = np.nan, np.nan
    
    print(f"\n--- {name} ---")
    print(f"Number of clusters: {len(set(labels))}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    
    #  Save clustering model
    joblib.dump(model, f"models/{name.replace(' ','_')}_cluster.pkl")

print("\n All full pipelines and clustering models saved successfully!")
