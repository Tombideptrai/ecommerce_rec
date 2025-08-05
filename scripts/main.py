import pandas as pd 
import kagglehub
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import joblib
import shutil
import os

path = kagglehub.dataset_download("kartikeybartwal/ecommerce-product-recommendation-collaborative")

# LOADING AND CLEANING DATA
os.makedirs('data', exist_ok=True)

for filename in os.listdir(path):
    src_file = os.path.join(path, filename)
    dst_file = os.path.join('data', filename)
    if os.path.isfile(src_file):
        shutil.copy(src_file, dst_file)

print("Files have been copied to /data")

df = pd.read_csv('data/user_personalized_features.csv')

if 'Unnamed: 0' in df.columns:
	df.drop(['Unnamed: 0'], inplace=True, axis=1)
     
num_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Convert numeric columns to numeric types
for col in num_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

cat_columns = df.select_dtypes(include=['object', 'bool']).columns

for col in cat_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# ADDING MORE FEATURES FOR DATASET
np.random.seed(42)

df['Purchased'] = np.random.choice([0, 1], size=len(df), p=[0.60, 0.40])
df['Purchase Quantity'] = np.random.randint(1, 5, size=len(df)) * df['Purchased']
df['Customer'] = np.random.choice(['Returning', 'New'], size=len(df), p=[0.42, 0.58])
df['Customer'] = LabelEncoder().fit_transform(df['Customer'])

df['Product Purchased'] = np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], size=len(df), p=[0.34, 0.21, 0.17, 0.28])
df['Product Purchased'] = LabelEncoder().fit_transform(df['Product Purchased'])

cluster_features = ['Age', 'Income', 'Interests', 'Last_Login_Days_Ago', 'Product_Category_Preference']

scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

inertia = []
K = range(1, 100)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=16, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

cluster_recommendations = {}
for cluster in df['Cluster'].unique():
    cluster_users = df[df['Cluster'] == cluster]
    top_products = cluster_users['Product Purchased'].value_counts().head(3)
    cluster_recommendations[cluster] = set(top_products.index)

hits = 0
total = 0
for idx, row in df.iterrows():
    cluster = row['Cluster']
    actual_product = row['Product Purchased']
    if actual_product in cluster_recommendations[cluster]:
        hits += 1
    total += 1

hit_rate = hits / total 
print(f"Top-3 Recommendation Hit Rate: {hit_rate:.2%}")

# Save the model
kmeans = KMeans(n_clusters=16, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

joblib.dump(kmeans, 'model/kmeans_model.joblib')
print("KMeans model saved as model/kmeans_model.joblib")
