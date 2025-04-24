# %% ### visualization of which topics are most relevant to each document
import pandas as pd
import numpy as np
document_index_to_show = 0
# # Load the W matrix from the saved file
# W = np.loadtxt(f"{dir_to_save}/W_matrix.csv", delimiter=",")

W_csv_path = "./output/mu_l2_4tps/W_matrix.csv"
npr = pd.read_csv("./data/netflix_titles.csv")
W = np.loadtxt(W_csv_path
               ,delimiter=',')

###
norms = np.sum(W, axis=1, keepdims=True)
# Avoid division by zero
norms = np.maximum(norms, 1e-10)
# Normalize rows of W
W = W / norms
print(sum(W[document_index_to_show,:])
)
print(W[document_index_to_show,].round(3))

print(npr['title'][document_index_to_show],":", npr['description'][document_index_to_show])
# %%
