# %% ### visualization of which topics are most relevant to each document
import pandas as pd
import numpy as np
# document_index_to_show = 0
document_title_to_show = "The 100"
# # Load the W matrix from the saved file
# W = np.loadtxt(f"{dir_to_save}/W_matrix.csv", delimiter=",")

W_csv_path = "./output/als_l2_6tps/W_matrix.csv"
npr = pd.read_csv("./data/netflix_titles.csv")

document_index_to_show = npr[npr['title'] == document_title_to_show].index[0]
W = np.loadtxt(W_csv_path
               ,delimiter=',')

###
norms = np.sum(W, axis=1, keepdims=True)
# Avoid division by zero
norms = np.maximum(norms, 1e-10)
# Normalize rows of W
W = W / norms

# print(sum(W[document_index_to_show,:]))
print(W[document_index_to_show,].round(3))

print(npr['title'][document_index_to_show],":", npr['description'][document_index_to_show])
# %%


from sklearn.metrics.pairwise import cosine_similarity
target_vector = W[document_index_to_show]
# Compute similarity scores between the target movie and all others
similarities = cosine_similarity([target_vector], W)[0]

# Get top 5 most similar movies (excluding the movie itself)
top_n = 5
similar_indices = similarities.argsort()[::-1]  # descending order
similar_indices = [i for i in similar_indices if i != document_index_to_show][:top_n]

# Show results
for i in similar_indices:
    print(f"{npr['title'][i]}: {npr['description'][i][:150]}... (score: {similarities[i]:.3f})")

# %%
