# stream.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#  Load Full Pipelines & Data

clf_model = joblib.load("models/Random_Forest_clf_pipeline.pkl")   # Use the full pipeline
reg_model = joblib.load("models/Gradient_Boosting_reg_pipeline.pkl")  # Full pipeline
cluster_model = joblib.load("models/KMeans_cluster.pkl")           # Clustering
df = pd.read_excel("data/train_data_aggregated.xlsx")              # For visualizations

st.set_page_config(page_title="E-commerce Session Analysis", layout="wide")
st.title("E-commerce Session Insights & Predictions")


#  User Input Section

st.sidebar.header("Input Session Features")

def user_input_features():
    total_clicks = st.sidebar.slider("Total Clicks", 1, 50, 5)
    last_page_visited = st.sidebar.slider("Last Page Visited", 1, 10, 2)
    min_price = st.sidebar.slider("Minimum Price", 0, 1000, 35)
    max_price = st.sidebar.slider("Maximum Price", 0, 5000, 60)
    num_colors_clicked = st.sidebar.slider("Number of Colors Clicked", 0, 20, 3)
    revisit_count = st.sidebar.slider("Revisit Count", 0, 20, 2)
    bounce = st.sidebar.selectbox("Bounce (1=yes,0=no)", [0,1])
    exit_rate = st.sidebar.slider("Exit Rate (0-1)", 0.0, 1.0, 0.2)
    last_category_clicked = st.sidebar.selectbox("Last Category Clicked", ["A","B","C","D"])
    most_visited_category = st.sidebar.selectbox("Most Visited Category", ["A","B","C","D"])

    data = {
        "num_colors_clicked": num_colors_clicked,
        "total_clicks": total_clicks,
        "revisit_count": revisit_count,
        "last_page_visited": last_page_visited,
        "min_price": min_price,
        "bounce": bounce,
        "max_price": max_price,
        "exit_rate": exit_rate,
        "last_category_clicked": last_category_clicked,
        "most_visited_category": most_visited_category
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()


#  Predictions

st.subheader(" Predictions")

# Classification
purchase_pred = clf_model.predict(input_df)[0]
purchase_prob = clf_model.predict_proba(input_df)[0][1]
st.write(f"**Purchase Complete:** {'Yes' if purchase_pred==1 else 'No'}")
st.write(f"**Probability of Purchase:** {purchase_prob:.2f}")

# Regression
revenue_pred = reg_model.predict(input_df)[0]
st.write(f"**Estimated Revenue:** ${revenue_pred:,.2f}")

# Clustering
cluster_label = cluster_model.predict(clf_model.named_steps['preprocessor'].transform(input_df))[0]
st.write(f"**Customer Segment (Cluster):** {cluster_label}")


#  Visualizations

st.subheader(" Session Feature Visualizations")

# Total clicks histogram
fig, ax = plt.subplots()
sns.histplot(df["total_clicks"], bins=20, kde=True, color='skyblue', ax=ax)
ax.axvline(input_df["total_clicks"][0], color='red', linestyle='--', label='Your Input')
ax.set_title("Distribution of Total Clicks")
ax.set_xlabel("Total Clicks")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# Purchase pie chart
st.subheader("Purchase Distribution")
purchase_counts = df['purchase_complete'].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(purchase_counts, labels=['Not Purchased','Purchased'], autopct='%1.1f%%', colors=['lightcoral','lightskyblue'])
st.pyplot(fig2)

# Most visited category bar chart
st.subheader("Most Visited Category Counts")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x='most_visited_category', ax=ax3, palette='pastel')
ax3.set_xlabel("Category")
ax3.set_ylabel("Count")
st.pyplot(fig3)
