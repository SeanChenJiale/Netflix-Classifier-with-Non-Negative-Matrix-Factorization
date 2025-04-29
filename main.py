#%%
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition._nmf import _initialize_nmf  # For NNDSVD initialization
from scipy.optimize import nnls  # For ALS
import csv
#%%
def random_init(V,n_components,seed=1):
    """
    Random initialization of W and H for NMF [0,1]"""
    np.random.seed(seed) # Set the random seed for reproducibility
    m, n = V.shape
    W = np.abs(np.random.rand(m, n_components)) 
    H = np.abs(np.random.rand(n_components, n))
    return W,H

def custom_nmf(V,n_components,optimization_type = 'als', optim_loss = 'l2', init = "random_init", max_iter=1000, tol=1e-4, seed=1
               ,normalize=True,norm_type='l1'): #normalize, norm_type are unused args now.
    """
    Custom NMF implementation using multiplicative update rules.
    Parameters:
    V: Input matrix (document-term matrix)
    n_components: Number of topics (components)
    normalize: Whether to normalize W and H to match scikit-learn's behavior 
    init: Initialization method ('random_init' or 'nndsvd')
    max_iter: Maximum number of iterations
    tol: Tolerance for convergence
    seed: Random seed for reproducibility

        """
    def update_H_mu(W, H, V):
        numerator = W.T @ V
        denominator = (W.T @ W @ H) + 1e-10  # Add small constant for stability
        H = H * (numerator / denominator)
        return H  

    def update_W_mu(W, H, V):
        numerator = V @ H.T
        denominator = (W @ H @ H.T) + 1e-10  # Add small constant for stability
        W = W * (numerator / denominator)
        return W  

    def update_W_als(W, H, V):
        for j in range(V.shape[0]):  # Update each row of W
            # W[j, :] = nnls(H.T, V[j, :])[0]
            W[j, :] = nnls(H.T, V[j, :].toarray().flatten())[0]  # Convert row to dense
        return W

    def update_H_als(W, H, V):
        for k in range(V.shape[1]):  # Update each column of H
            # H[:, k] = nnls(W, V[:, k])[0]
            H[:, k] = nnls(W, V[:, k].toarray().flatten())[0]  # Convert column to dense
        return H

    # def _normalize(W, H, norm_type='l1'):
    #     if norm_type == 'l1':
    #         norms = np.sum(W, axis=1)  # Compute row-wise L1 norms
    #     elif norm_type == 'l2':
    #         norms = np.linalg.norm(W, axis=1)  # Compute row-wise L2 norms
    #     else:
    #         print('use l1 or l2')
    #     # print("norms.shape:", norms.shape)  # Debug: Print the shape of norms
    #     W_normalized = W / norms[:, np.newaxis]  # Normalize rows of W
    #     return W_normalized, H  # Do not adjust H
        
    ## check for what kind of initialization 

    if init == "random_init":
        W, H = random_init(V, n_components, seed=seed)
    elif init == "nndsvd":
        # Initialize with NNDSVD
        W, H = _initialize_nmf(V, n_components, init='nndsvd', random_state=seed)
    else:
        raise ValueError("Invalid initialization method. Use 'random_init' or 'nndsvd'.")
    
    prev_error = float('inf')
    error_list = []
    consecutive_increases = 0
    for i in range(max_iter):
        if optimization_type == 'mu':  # Multiplicative Updates
            H = update_H_mu(W, H, V)
            W = update_W_mu(W, H, V)
        elif optimization_type == 'als':  # Alternating Least Squares
            W = update_W_als(W, H, V)
            H = update_H_als(W, H, V)
        else:
            raise ValueError("Invalid loss_function. Use 'mu' or 'als'.")
        
        # if normalize == True:
        #     # Normalize W and H to match scikit-learn's behavior
        #     W, H = _normalize(W, H,norm_type=norm_type)
        
        # Reconstruction error calculation
        WH = W @ H
        # Compute Frobenius norm of the error
        if optim_loss == 'l1':
            # current_error = np.sum(np.abs(V - WH))
            current_error = np.linalg.norm(V - WH, ord=1)
        elif optim_loss == 'l2':
            # current_error = np.sum((V - WH) ** 2)
            current_error = np.linalg.norm(V - WH, 'fro')
        print(f"Iteration {i+1}, Error: {current_error}")
  
        # Check convergence
        error_list.append([i,current_error])

        # Check if error increased
        if current_error > prev_error:
            consecutive_increases += 1
            if consecutive_increases >= 10:  # Stop if error increases for 10 iterations
                print("Stopping: Error increased for 10 consecutive iterations.")
                break
        else:
            consecutive_increases = 0  # Reset the counter if error decreases or stays the same
            
            # Check if the error change is below the tolerance
            if prev_error - current_error < tol:
                print("Stopping: Error change below tolerance.")
                break
        
        # Update previous error
        prev_error = current_error
    return W, H ,error_list

if __name__ == "__main__":
    # Data loading and preprocessing
    if os.path.exists('./data/netflix_titles.csv'):
        npr = pd.read_csv("./data/netflix_titles.csv")
        # print("Available columns:", npr.columns)
        
        # Use the same preprocessing as scikit-learn version
        tfidf = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        dtm = tfidf.fit_transform(npr['description'])
        import time

        number_of_topics_list = [3]
        loss_function_list = ['mu']
        run_folder = 'rand_init'
        error_type = 'l1'
        init='nndsvd'
        # Apply custom NMF
           
        for number_of_topics in number_of_topics_list:
            for loss_function in loss_function_list:
                start_time = time.time() 
                W, H = custom_nmf(dtm,
                                normalize=True,
                                init="nndsvd",
                                loss_function=loss_function,
                                norm_type=error_type,
                                n_components=number_of_topics,
                                max_iter=1000,
                                tol=1e-4, 
                                seed=1)
                total_runtime = time.time() - start_time
                print("--- %s seconds ---" % (total_runtime))

                # Save runtime to a text file
                runtime_file = f'./output/{loss_function}_{error_type}_{number_of_topics}tps/runtime.txt'
                os.makedirs(os.path.dirname(runtime_file), exist_ok=True)
                with open(runtime_file, 'w') as f:
                    f.write(f"Total runtime: {total_runtime:.4f} seconds\n")


                dir_to_save = f'./output/{loss_function}_{error_type}_{number_of_topics}tps' #[0.052 0.35  0.598]
                
                #[0.636 0.054 0.31 ]
                # Display topics with weights (matches scikit-learn's format)
                os.makedirs(dir_to_save, exist_ok=True)
                df = pd.DataFrame(error_list, columns=['iter','err'])
                df.to_csv(f'{dir_to_save}/error.csv',index=False)
                for topic_idx, topic in enumerate(H):
                    with open(f'{dir_to_save}/topics{topic_idx}.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        print(f"\nTopic #{topic_idx}:")
                        top_indices = topic.argsort()[-15:][::-1]
                        top_words = tfidf.get_feature_names_out()[top_indices]
                        top_weights = topic[top_indices]
                        
                        for word, weight in zip(top_words, top_weights):
                            print(f"{word}: {weight:.4f}")
                            writer.writerow([word, weight])
                        


                # save the W and H matrices to csv files
                np.savetxt(f"{dir_to_save}/W_matrix.csv", W, delimiter=",")
                np.savetxt(f"{dir_to_save}/H_matrix.csv", H, delimiter=",")
    else:
        print("netflix_titles.csv not found")
        exit(1)
