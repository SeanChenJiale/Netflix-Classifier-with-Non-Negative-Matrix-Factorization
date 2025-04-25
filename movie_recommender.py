# %% ### visualization of which topics are most relevant to each document
import pandas as pd
import numpy as np
# document_index_to_show = 0
document_title_to_show = "Angry Birds"
# # Load the W matrix from the saved file
# W = np.loadtxt(f"{dir_to_save}/W_matrix.csv", delimiter=",")

W_csv_path = "./output/mu_l2_6tps/W_matrix.csv"
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
    any(category in npr.loc[i, 'listed_in'] for category in target_categories)
][:top_n]

# Show results
print(f"Movies with the same rating ('{target_rating}') as '{document_title_to_show}':")
for i in filtered_indices:
    print(f"{npr['title'][i]}: {npr['description'][i][:150]}... (score 4 d.p.: {similarities[i]:.3f}), target vector :{W[i,].round(3)}\n")

# %%
