import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file
loss_fn_list = ['als','mu']
topic_count_list = [3,4,5]
l1_or_l2_list = ['l2']
for l1_or_l2 in l1_or_l2_list:
    for topic_count in topic_count_list:
        for loss_fn in loss_fn_list:
            for topic_int in range(topic_count):
                base_dir = f"./output/{loss_fn}_{l1_or_l2}_{topic_count}tps/"
                topics_df = pd.read_csv(f'{base_dir}topics{topic_int}.csv',header=None, sep = ' ')

                # Path to the font file (update this path to the font you want to use)
                font_path = 'C:/Windows/Fonts/bahnschrift.ttf'  # Example: Arial font on Windows
                print(topics_df)
        
                # Access the first and second columns
                words = topics_df.iloc[:, 0]  # First column
                weights = topics_df.iloc[:, 1]  # Second column
                word_freq = {word: weight for word, weight in zip(words, weights)}

                # Generate the word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

                # Plot the word cloud
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for Topic {topic_int}')

                # Save the plot as an image
                plt.savefig(f'{base_dir}topic{topic_int}.png', format='png', dpi=300)
                plt.close()  # Close the figure to avoid overlapping plots