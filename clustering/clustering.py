import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
df = pd.read_csv(r'csv_files\final_merged_player_salary_data_final.csv')
df.fillna(0, inplace=True)

import matplotlib.pyplot as plt

# Select numeric columns for t-SNE
numeric_df = df.select_dtypes(include=['number'])

# # Run t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(numeric_df)

# # Add t-SNE results to DataFrame
# df['tsne-2d-one'] = tsne_results[:,0]
# df['tsne-2d-two'] = tsne_results[:,1]

# # Plot t-SNE results
# plt.figure(figsize=(8,6))
# plt.scatter(df['tsne-2d-one'], df['tsne-2d-two'], alpha=0.6)
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.title('t-SNE Clustering')
# plt.show()

numeric_df = numeric_df[numeric_df['pitcher'] == 0].reset_index(drop=True)
print(len(numeric_df))
# Run Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
svd_results = svd.fit_transform(numeric_df)
print(f"Explained variance ratio (2 components): {svd.explained_variance_ratio_}")
print(f"Total variance explained: {svd.explained_variance_ratio_.sum()}")

print(len(svd_results))
# Add SVD results to DataFrame
numeric_df['svd-2d-one'] = svd_results[:, 0]
numeric_df['svd-2d-two'] = svd_results[:, 1]

# Plot SVD results
plt.figure(figsize=(8, 6))
plt.scatter(numeric_df['svd-2d-one'], numeric_df['svd-2d-two'], alpha=0.6)
plt.xlabel('SVD 1')
plt.ylabel('SVD 2')
plt.title('Truncated SVD Clustering')
plt.show()
