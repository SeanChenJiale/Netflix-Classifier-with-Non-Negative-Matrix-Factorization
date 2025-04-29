# %% ### visualization of which topics are most relevant to each document
import pandas as pd
import numpy as np
# document_index_to_show = 0
document_title_to_show = "Crash Landing on You"
# # Load the W matrix from the saved file
# W = np.loadtxt(f"{dir_to_save}/W_matrix.csv", delimiter=",")

W_csv_path = "./output/nndsvd_mu_l2_6tps/W_matrix.csv"
npr = pd.read_csv("./data/netflix_titles.csv")

document_index_to_show = npr[npr['title'] == document_title_to_show].index[0]
W = np.loadtxt(W_csv_path
               ,delimiter=',')

# get movie's rating, type, and catagories
target_rating = npr.loc[document_index_to_show, 'rating']
print(f"Rating for '{document_title_to_show}': {target_rating}")

target_type = npr.loc[document_index_to_show, 'type']
print(f"Type of show: '{document_title_to_show}': {target_type}")

target_categories = npr.loc[document_index_to_show, 'listed_in'].split(", ")
print(f"Catagories of '{document_title_to_show}': {target_categories}")
###
norms = np.sum(W, axis=1, keepdims=True)
# Avoid division by zero
norms = np.maximum(norms, 1e-10)
# Normalize rows of W
W = W / norms

# print(sum(W[document_index_to_show,:]))
print(W[document_index_to_show,].round(3))

print(npr['title'][document_index_to_show],":", npr['description'][document_index_to_show],"\n")
# %%


from sklearn.metrics.pairwise import cosine_similarity
target_vector = W[document_index_to_show]
# Compute similarity scores between the target movie and all others
similarities = cosine_similarity([target_vector], W)[0]

top_n = 3
similar_indices = similarities.argsort()[::-1]  # descending order
filtered_indices = [
    i for i in similar_indices
    if i != document_index_to_show and 
    npr.loc[i, 'rating'] == target_rating and 
    npr.loc[i,'type'] == target_type and
    "Korean TV Shows" in npr.loc[i, 'listed_in']
    # any(category in npr.loc[i, 'listed_in'] for category in target_categories)
][:top_n]

# Show results
print(f"Movies with the same rating ('{target_rating}') as '{document_title_to_show}':")
for i in filtered_indices:
    print(f"{npr['title'][i]}: {npr['description'][i][:150]}... (score 4 d.p.: {similarities[i]:.3f}), target vector :{W[i,].round(3)}\n")

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# List of famous series and movie titles
famous_titles = [
    "Crash Landing on You",
    "Breaking Bad",
    "Bridgerton",
    "The Pixar Story",
    "Neon Genesis Evangelion"    
]

# Create a DataFrame for the document-topic matrix
W_df = pd.DataFrame(W, columns=[f"Topic {i}" for i in range(W.shape[1])])


# Add document titles as row labels
W_df.index = npr['title']

# Filter W_df to include only rows corresponding to the famous titles
filtered_W_df = W_df.loc[W_df.index.isin(famous_titles)]

# Display the filtered DataFrame

# # Plot the heatmap
plt.figure(figsize=(12, 8), facecolor='#f0f0f0')  # Set the figure background to light grey
sns.heatmap(filtered_W_df, cmap="YlGnBu", annot=False, cbar=True, xticklabels=True, yticklabels=True)
# Set a lighter grey background
plt.gca().set_facecolor('#000000')  # Light grey background

plt.title("Document-Topic Matrix Heatmap")
plt.xlabel("Topics")
plt.ylabel("Documents")
plt.tight_layout()

# Show the plot
plt.show()
#%%
# Define a threshold for high scores
threshold = 0.9  # Adjust this value based on your data

# Filter for series with high scores in both Topic 2 and Topic 4
topics_to_check = [2]  # Topic 2 and Topic 4 (zero-indexed)
filtered_series = W_df[
    (W_df.iloc[:, topics_to_check[0]] > threshold) 
    # & (W_df.iloc[:, topics_to_check[1]] > threshold)
    ]

# Display the filtered series
print("Series with high scores in both Topic 2 and Topic 4:")
print(filtered_series)
# Print all indices of the filtered series
print("Indices of series with high scores in both Topic 2 and Topic 4:")
print(filtered_series.index.tolist())
# Optionally, print the titles and scores
for index, row in filtered_series.iterrows():
    print(f"Title: {index}")
    print(f"Score {row[f'Topic {topics_to_check[0]}']:.3f}")
    # print(f"Score in Topic 4: {row[f'Topic {topics_to_check[1]}']:.3f}\n")
# %%
