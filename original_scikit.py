#%%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

npr = pd.read_csv("./data/netflix_titles.csv") ## input your csv here, and the path tot he csv eg "C://.... .csv"
npr['description']
# print(npr.columns) # this prints the columns within the csv, select the description/summary or text column.
npr.head() # prints the top 5 entries of the whole data frame

tfidf = TfidfVectorizer(max_df=0.95,min_df = 2, stop_words= 'english', lowercase=True, strip_accents=   'ascii') # Removing stop words such as full stops, articles eg (a and the, )
# tfidf = CountVectorizer() # Removing stop words such as full stops, articles eg (a and the, )
dtm= tfidf.fit_transform(npr['description']) ## if you want to use a different csv, change 'description' to
## a different column name of the csv 

# print(dtm)

nmf_model = NMF(n_components= 3) ## model classifies it into differnt components based on the number you choose,
#in this case components is 4
W = nmf_model.fit_transform(dtm)
H = nmf_model.components_

print(nmf_model.reconstruction_err_)

# Viewing top 15 words in each topic
for index, topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    # Get the indices of the top 15 weights in descending order
    top_indices = topic.argsort()[-15:][::-1]
    # Get the corresponding words and weights
    top_words = [tfidf.get_feature_names_out()[i] for i in top_indices]
    top_weights = [topic[i] for i in top_indices]
    # Print the words and their weights
    print("Words:", top_words)
    print("Weights:", top_weights)
    print('\n')


# print(tfidf.get_feature_names_out())
# len(tfidf.get_feature_names_out()) # prints the number of words in the vocabulary
# print(len(topic)) # sorts the words in the topic from highest to lowest relevance



# %%
print("Topic weights for the first document:")
print(W[0])
# %%
