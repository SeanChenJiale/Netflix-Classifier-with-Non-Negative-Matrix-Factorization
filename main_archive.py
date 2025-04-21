#%%
from utils.preprocessing_utils import load_one_hot_encoded_csv , vectorize_text , preprocess_netflix_csv

#check if './data/filtered_one_hot_encoded.csv' exists
import os

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

# https://github.com/akcarsten/Non_Negative_Matrix_Factorization <- adapted from 
def nmf(V, n_components, max_iter=1000, tol=1e-4, seed = 1):
    def update_H(W, H, V):
        numerator = W.T @ V
        denominator = (W.T @ W @ H) + 1e-10
        H *= numerator / denominator
        return H

    def update_W(W, H, V):
        numerator = V @ H.T
        denominator = (W @ H @ H.T) + 1e-10
        W *= numerator / denominator
        return W
    m, n = V.shape
    
    np.random.seed(seed) # Set the random seed for reproducibility
    # Initialize W and H with non-negative random values
    W = np.abs(np.random.rand(m, n_components)) 
    H = np.abs(np.random.rand(n_components, n))

    prev_error = float('inf')
    
    for i in range(max_iter):
        H = update_H(W, H, V)
        W = update_W(W, H, V)

        # Compute error
        WH = W @ H
        # Compute Frobenius norm of the error
        error = np.linalg.norm(V - WH, ord='fro')
        print(f"Iteration {i+1}, Error: {error}")

        # Check relative change
        # Prevents getting stuck in local minima
        if abs(prev_error - error) / prev_error < tol:
            break
        prev_error = error

    return W, H


if __name__ == "__main__":
    if os.path.exists('./data/filtered_one_hot_encoded.csv'):
        npr = pd.read_csv("./data/netflix_titles.csv") ## input your csv here, and the path tot he csv eg "C://.... .csv"
        npr['description']
        print(npr.columns) # this prints the columns within the csv, select the description/summary or text column.
        npr.head() # prints the top 5 entries of the whole data frame

        tfidf = TfidfVectorizer(max_df=0.95,min_df = 2, stop_words= 'english', lowercase=True, strip_accents=   'ascii') # Removing stop words such as full stops, articles eg (a and the, )
        # tfidf = CountVectorizer() # Removing stop words such as full stops, articles eg (a and the, )
        dtm= tfidf.fit_transform(npr['description'])

        W,H =nmf(dtm, n_components=5, max_iter=1000, tol=1e-4,seed =1)
        
        #get 10 highest values of each column in W and map to words
        for i in range(W.shape[1]):
            indices = np.argsort(W[:, i])[::-1][:15]
            words = tfidf.get_feature_names_out()[indices]
            
            print(f"Topic {i+1}:")  
            # Print the words associated with each topic and its corresponding weight
            for j, word in enumerate(words):
                weight = W[indices[j], i]
                print(f"{word}: {weight:.4f}")




    else:
        print("Please run the preprocessing script first to generate the filtered_one_hot_encoded.csv file.")
        exit(1)
# %%
