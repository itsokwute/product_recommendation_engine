import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# -------------------------------------------------------------------------
# STEP 1: Simulate a Sparse User-Item Interaction Matrix
# -------------------------------------------------------------------------
np.random.seed(42)
n_users = 100
n_items = 20

# Create dummy transactional interaction log (User ID, Product ID, Rating 1-5)
raw_interactions = pd.DataFrame({
    'user_id': np.random.randint(1001, 1001 + n_users, size=500),
    'product_id': np.random.randint(5001, 5001 + n_items, size=500),
    'rating': np.random.randint(1, 6, size=500)
})

# Remove duplicate interactions to keep pivot table clean
raw_interactions = raw_interactions.drop_duplicates(subset=['user_id', 'product_id'])

# Pivot the data into a classic User-Item Matrix (Rows = Users, Columns = Products)
user_item_matrix = raw_interactions.pivot(index='user_id', columns='product_id', values='rating')

# Fill missing values with 0 to represent items the user hasn't seen/bought yet
user_item_matrix_filled = user_item_matrix.fillna(0)

print(f"User-Item Matrix shape: {user_item_matrix_filled.shape}")
print(f"Total interactions recorded: {len(raw_interactions)} out of 2000 possible combination grid.\n")

# -------------------------------------------------------------------------
# STEP 2: Apply Matrix Factorization (SVD)
# -------------------------------------------------------------------------
# We extract 5 latent features (hidden themes/genres connecting users and products)
n_latent_features = 5
svd = TruncatedSVD(n_components=n_latent_features, random_state=42)

# Deconstruct the user interactions into latent spaces
user_embeddings = svd.fit_transform(user_item_matrix_filled)
item_embeddings = svd.components_

# Reconstruct the matrix by multiplying the dense sub-matrices back together.
# This matrix dot-product outputs predicted rating scores for EVERY blank space!
predicted_ratings = np.dot(user_embeddings, item_embeddings)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

print("Matrix factorization complete. Reconstruction successful.\n")

# -------------------------------------------------------------------------
# STEP 3: Operational Business Function to Generate Recommendations
# -------------------------------------------------------------------------
def get_top_product_recommendations(user_id, num_recommendations=3):
    if user_id not in user_item_matrix.index:
        return "User ID not found in database."
    
    # 1. Fetch products the user has historically purchased/rated already
    user_historical_profile = user_item_matrix.loc[user_id]
    already_interacted = user_historical_profile[user_historical_profile.notna()].index.tolist()
    
    # 2. Fetch the model's reconstructed prediction scores for this user
    user_predicted_profiles = predicted_ratings_df.loc[user_id]
    
    # 3. Filter out items already bought, then sort the remaining by highest predicted score
    recommendations = user_predicted_profiles.drop(index=already_interacted)
    top_recommendations = recommendations.nlargest(num_recommendations)
    
    return top_recommendations

# -------------------------------------------------------------------------
# STEP 4: Test the Engine for a Target Account
# -------------------------------------------------------------------------
target_user = user_item_matrix.index[0]  # Grab the first available User ID
top_picks = get_top_product_recommendations(user_id=target_user, num_recommendations=3)

print(f"=== Recommendation Engine Output ===")
print(f"Target User Account ID: {target_user}")
print(f"\nTop 3 Unseen Products to Recommend (and their predicted rating match scores):")
print(top_picks)