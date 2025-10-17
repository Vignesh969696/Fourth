import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Load aggregated dataset

agg_path = "data/train_data_aggregated.xlsx"
df = pd.read_excel(agg_path)

print("\n Aggregated Data Preview:")
print(df.head())

#  Histograms for numeric features

numeric_features = [
    'total_clicks', 'last_page_visited', 'avg_price', 'max_price', 
    'min_price', 'num_colors_clicked', 'revisit_count', 'revenue'
]

df[numeric_features].hist(bins=20, figsize=(14,7), edgecolor='black')
plt.suptitle("Distribution of Numeric Features")
plt.show()

#  Bar plots for categorical features

categorical_features = [
    'most_visited_category', 'first_category_clicked', 
    'last_category_clicked', 'top_location_clicked', 'bounce', 'exit_rate'
]

for col in categorical_features:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Count of {col}")
    plt.xticks(rotation=45)
    plt.show()


#  Correlation heatmap (including target)

plt.figure(figsize=(12,8))
corr = df[numeric_features + ['purchase_complete']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features + Target")
plt.show()


#  Target distribution

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='purchase_complete')
plt.title("Purchase Complete Distribution (Synthetic Probabilistic Target)")
plt.show()


#  Revenue distribution for completed purchases

plt.figure(figsize=(8,4))
sns.histplot(df[df['purchase_complete']==1]['revenue'], bins=20, kde=True, color='orange')
plt.title("Revenue Distribution for Completed Purchases")
plt.show()


#  Mean purchase rate by categorical features

for col in ['most_visited_category', 'first_category_clicked', 'last_category_clicked', 'top_location_clicked']:
    plt.figure(figsize=(8,4))
    df.groupby(col)['purchase_complete'].mean().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Mean Purchase Rate by {col}")
    plt.ylabel("Purchase Rate")
    plt.show()


#  Average revenue for completed purchases by categorical features

for col in ['most_visited_category', 'first_category_clicked', 'last_category_clicked', 'top_location_clicked']:
    plt.figure(figsize=(8,4))
    df[df['purchase_complete']==1].groupby(col)['revenue'].mean().plot(kind='bar', color='orange', edgecolor='black')
    plt.title(f"Average Revenue for Completed Purchases by {col}")
    plt.ylabel("Average Revenue")
    plt.show()

