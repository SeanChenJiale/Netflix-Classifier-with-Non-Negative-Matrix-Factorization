import pandas as pd
from archive_src.wordcloud_archive import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file
topics_df = pd.read_csv('./data/top_words_by_topic.csv')

# Path to the font file (update this path to the font you want to use)
font_path = 'C:/Windows/Fonts/bahnschrift.ttf'  # Example: Arial font on Windows

# Generate word clouds for each topic
for i in range(1, (len(topics_df.columns) // 2) + 1):
    # Create a dictionary of words and their weights
    words = topics_df[f'topic{i}']
    weights = topics_df[f'weights{i}']
    word_freq = {word: weight for word, weight in zip(words, weights)}

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Topic {i}')

    # Save the plot as an image
    plt.savefig(f'./imgdata/topic{i - 1}.png', format='png', dpi=300)
    plt.close()  # Close the figure to avoid overlapping plots