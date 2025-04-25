import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file
loss_fn_list = ['als','mu']
topic_count_list = [3,4,5,6,7]
l1_or_l2_list = ['l2']
for l1_or_l2 in l1_or_l2_list:
    for topic_count in topic_count_list:
        for loss_fn in loss_fn_list:
            for topic_int in range(topic_count):
                base_dir = f"./output/{loss_fn}_{l1_or_l2}_{topic_count}tps/"
                error_df = pd.read_csv(f'{base_dir}error.csv')
                # Plot the error
                plt.figure(figsize=(10, 6))
                plt.plot(error_df.iloc[:, 0], error_df.iloc[:, 1], marker='o', label='Reconstruction Error')
                plt.title(f'Error Plot for {loss_fn.upper()} - {l1_or_l2.upper()} - {topic_count} Topics (Topic {topic_int})')
                plt.xlabel('Iteration')
                plt.ylabel('Error')
                plt.grid(True)
                plt.legend()

                # Save the plot
                plt.savefig(f'{base_dir}error_plot.png', format='png', dpi=300)
                plt.close()  # Close the figure to avoid overlapping plots                