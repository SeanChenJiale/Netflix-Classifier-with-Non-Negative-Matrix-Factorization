import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file
loss_fn_list = ['als','mu']
topic_count_list = [6]
l1_or_l2_list = ['l2']
init_list = ['nndsvd','random_init']
for init in init_list:
    for l1_or_l2 in l1_or_l2_list:
        for topic_count in topic_count_list:
            for loss_fn in loss_fn_list:
                base_dir = f"./output/{init}_{loss_fn}_{l1_or_l2}_{topic_count}tps/"
                error_df = pd.read_csv(f'{base_dir}error.csv')
                # Plot the error
                plt.figure(figsize=(10, 6))
                plt.plot(error_df.iloc[:, 0], error_df.iloc[:, 1], marker='o', label='Reconstruction Error')
                plt.title(f'Error Plot for {init} - {loss_fn.upper()} - {l1_or_l2.upper()} - {topic_count} Topics ')
                plt.xlabel('Iteration')
                plt.ylabel('Error')
                plt.grid(True)
                plt.legend()

                # Save the plot
                plt.savefig(f'{base_dir}error_plot.png', format='png', dpi=300)
                plt.close()  # Close the figure to avoid overlapping plots                