import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Load the data
npr = pd.read_csv("./data/netflix_titles.csv")
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', lowercase=True, strip_accents='ascii')
dtm = tfidf.fit_transform(npr['description'])

# Fit the NMF model
nmf_model = NMF(n_components=5)
nmf_model.fit(dtm)

# Prepare data for CSV
topics_data = {}
for index, topic in enumerate(nmf_model.components_):
    top_indices = topic.argsort()[-15:][::-1]
    top_words = [tfidf.get_feature_names_out()[i] for i in top_indices]
    top_weights = [topic[i] for i in top_indices]
    topics_data[f'topic{index + 1}'] = top_words
    topics_data[f'weights{index + 1}'] = top_weights

# Create a DataFrame and save to CSV
topics_df = pd.DataFrame(topics_data)
topics_df.to_csv('./data/top_words_by_topic.csv', index=False)

