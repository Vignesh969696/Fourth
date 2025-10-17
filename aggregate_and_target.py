import pandas as pd
import numpy as np

# Load the datasets

train_path = "data/train_data.xlsx"
test_path = "data/test_data.xlsx"

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

print("\nTRAIN DATA PREVIEW:")
print(train_df.head())

# Session-level aggregation

train_df = train_df.sort_values(["session_id", "order"])

session_agg = train_df.groupby("session_id").agg(
    total_clicks=("order", "count"),
    last_page_visited=("page", "max"),
    avg_price=("price", "mean"),
    max_price=("price", "max"),
    min_price=("price", "min"),
    most_visited_category=("page1_main_category", lambda x: x.mode()[0]),
    first_category_clicked=("page1_main_category", lambda x: x.iloc[0]),
    last_category_clicked=("page1_main_category", lambda x: x.iloc[-1]),
    num_colors_clicked=("colour", pd.Series.nunique),
    top_location_clicked=("location", lambda x: x.mode()[0]),
    bounce=("order", lambda x: 1 if len(x) == 1 else 0),
    exit_rate=("page", lambda x: 1 if x.iloc[-1] == x.max() else 0),
    revisit_count=("page1_main_category", lambda x: x.duplicated().sum())
).reset_index()

# Creating a synthetic target

def generate_purchase_prob(row):
    """
    Base probability + mild influence from session behavior.
    Adds some randomness to avoid near-perfect predictability.
    """
    base_prob = 0.1 + np.random.uniform(0, 0.1)  # small randomness
    if row.total_clicks >= 5:
        base_prob += 0.1  # mild effect
    if row.last_page_visited == 5:
        base_prob += 0.1  # mild effect
    if row.revisit_count > 2:
        base_prob += 0.05
    return min(base_prob, 0.8)  # cap at 0.8 to avoid near-certain outcomes

np.random.seed(42)
session_agg["purchase_complete"] = session_agg.apply(
    lambda row: np.random.binomial(1, generate_purchase_prob(row)), axis=1
)


# Create revenue independent of purchase_complete

# Revenue is now based on avg_price and a random multiplier, not deterministic
np.random.seed(42)
session_agg["revenue"] = session_agg.apply(
    lambda row: row.avg_price * np.random.uniform(1, 5) * row.total_clicks, axis=1
)

# Scaling revenue for non-purchasers (to simulate abandoned carts)
session_agg.loc[session_agg["purchase_complete"] == 0, "revenue"] *= np.random.uniform(0, 0.5)

# Basic EDA on aggregated features

print("\n  Aggregated session-level data preview:")
print(session_agg.head())

print("\n Shape of aggregated data:", session_agg.shape)
print("\n Missing values:")
print(session_agg.isnull().sum())

print("\n Descriptive statistics:")
print(session_agg.describe())

print("\n Purchase complete value counts:")
print(session_agg["purchase_complete"].value_counts())

print("\n Revenue statistics:")
print(session_agg["revenue"].describe())


# Save aggregated dataset

session_agg.to_excel("data/train_data_aggregated.xlsx", index=False)
print("\n Aggregated dataset saved as 'train_data_aggregated.xlsx'")

# Additional analysis

df = session_agg.copy()

# Correlation with target
corr_with_target = df.corr(numeric_only=True)["purchase_complete"].abs().sort_values(ascending=False)
print("\n Top Correlated Features with Purchase Complete:")
print(corr_with_target)

# Value counts for categorical/binary features
categorical_features = [
    'most_visited_category', 'first_category_clicked', 
    'last_category_clicked', 'top_location_clicked', 'bounce',
    'purchase_complete', 'exit_rate'
]

for col in categorical_features:
    print(f"\n Value counts for {col}:")
    print(df[col].value_counts())
