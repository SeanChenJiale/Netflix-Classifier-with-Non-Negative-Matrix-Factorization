from utils.preprocessing_utils import vectorize_text , preprocess_netflix_csv

# Preprocess the Netflix CSV file and vectorize the text

descriptions = preprocess_netflix_csv()
vectorize_text(descriptions)