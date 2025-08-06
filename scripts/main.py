import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import joblib

# --- Load and preprocess data ---
df = pd.read_csv('data/user_personalized_features.csv')

# Drop unnecessary columns
for col in ['Unnamed: 0', 'User_ID']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scale numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# --- KMeans clustering for new users ---
cluster_features = ['Age', 'Income', 'Interests', 'Last_Login_Days_Ago', 'Product_Category_Preference']
X_cluster = df[cluster_features]
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

# For each cluster, find top 3 most purchased products
cluster_top_products = {}
for cluster in df['Cluster'].unique():
    cluster_users = df[df['Cluster'] == cluster]
    top_products = cluster_users['Product_Category_Preference'].value_counts().head(3)
    cluster_top_products[cluster] = [
        label_encoders['Product_Category_Preference'].inverse_transform([cat])[0]
        if hasattr(label_encoders['Product_Category_Preference'], 'inverse_transform')
        else cat
        for cat in top_products.index
    ]

# --- Collaborative Filtering for existing users ---
# Use Purchase_Frequency as rating
reader = Reader(rating_scale=(df['Purchase_Frequency'].min(), df['Purchase_Frequency'].max()))
data = Dataset.load_from_df(
    df[['User_ID', 'Product_Category_Preference', 'Purchase_Frequency']],
    reader
)
trainset, testset = train_test_split(data, test_size=0.3, random_state=42)

sim_options = {'name': 'cosine', 'user_based': True}
cf_model = KNNBasic(sim_options=sim_options)
cf_model.fit(trainset)

# --- Hybrid Recommendation Functions ---
def run_user_based_cf(user_id, trainset, cf_model, label_encoder):
    predictions = []
    all_items = trainset.all_items()
    user_inner_id = trainset.to_inner_uid(user_id)
    user_items = set([j for (j, _) in trainset.ur[user_inner_id]]) if user_inner_id in trainset.ur else set()
    item_inner_ids = [iid for iid in all_items if iid not in user_items]
    for iid in item_inner_ids:
        raw_iid = trainset.to_raw_iid(iid)
        pred = cf_model.predict(user_id, raw_iid)
        predictions.append((iid, pred.est))
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    return [label_encoder['Product_Category_Preference'].inverse_transform([trainset.to_raw_iid(iid)])[0] for iid, _ in top_n]

def run_kmeans_suggestion(user_profile_vector, kmeans_model, cluster_top_products):
    cluster_id = kmeans_model.predict([user_profile_vector])[0]
    return cluster_top_products.get(cluster_id, [])

def hybrid_recommend(user_id, user_profile_vector, trainset, cf_model, kmeans_model, cluster_top_products, label_encoder):
    if trainset.knows_user(user_id):
        return run_user_based_cf(user_id, trainset, cf_model, label_encoder)
    else:
        return run_kmeans_suggestion(user_profile_vector, kmeans_model, cluster_top_products)

# --- Example usage ---
# For an existing user
example_user_id = df['User_ID'].iloc[0]
user_profile_vector = df.loc[df['User_ID'] == example_user_id, cluster_features].values[0]
recommendations = hybrid_recommend(
    example_user_id, user_profile_vector, trainset, cf_model, kmeans, cluster_top_products, label_encoders
)
print(f"Recommendations for user {example_user_id}: {recommendations}")

# For a new user (simulate with random features)
new_user_profile = np.random.rand(len(cluster_features))
recommendations_new = hybrid_recommend(
    'new_user_id', new_user_profile, trainset, cf_model, kmeans, cluster_top_products, label_encoders
)
print(f"Recommendations for new user: {recommendations_new}")

# --- Save models ---
joblib.dump(kmeans, 'model/kmeans_model.joblib')
joblib.dump(cf_model, 'model/cf_model.joblib')
print("Models saved.")