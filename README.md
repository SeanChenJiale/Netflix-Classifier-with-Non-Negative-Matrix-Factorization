# Netflix Classifier with Non-Negative Matrix Factorization (NMF)

## File Structure

Netflix-Classifier-with-Non-Negative-Matrix-Factorization/
│
├── data/
│   └── netflix_titles.csv       # Input dataset
│
├── output/
│     ├── als_l2_3tps/             # Output for ALS optimization, L2 loss, 3 topics
│     │   ├── W_matrix.csv         # Document-topic matrix
│     │   ├── H_matrix.csv         # Topic-word matrix
│     │   ├── topicsX.csv          # Top words and weights for each topic
│     │   ├── error.csv            # Reconstruction error
│     │   └── error_plot.png       # Error plot image
│     ├── als_l2_4tps/             # Output for ALS optimization, L2 loss, 4 topics
│     │   ├── W_matrix.csv         # Document-topic matrix
│     │   ├── H_matrix.csv         # Topic-word matrix
│     │   ├── topicsX.csv          # Top words and weights for each topic
│     │   ├── error.csv            # Reconstruction error
│     │   └── error_plot.png       # Error plot image
│     ├── als_l2_5tps/             # Output for ALS optimization, L2 loss, 5 topics
│     │   ├── W_matrix.csv         # Document-topic matrix
│     │   ├── H_matrix.csv         # Topic-word matrix
│     │   ├── topicsX.csv          # Top words and weights for each topic
│     │   ├── error.csv            # Reconstruction error
│     │   └── error_plot.png       # Error plot image
│     ├── ...                      # Additional folders for other configurations
│
├── main.py                      # Main script for preprocessing and NMF
├── movie_recommender.py          # Movie recommendation script
├── wordcloud_gen.py              # Word cloud generation script
├── error_plotter.py              # Error plotting script
├── original_scikit.py            # Basic scikit-learn NMF example
└── requirements.txt              # Required dependencies

## Scripts Overview

1. **`main.py`**:
   - Preprocesses the Netflix dataset and applies a custom NMF implementation.
   - Outputs topic models, NMF matrices (`W` and `H`), and reconstruction errors.

2. **`movie_recommender.py`**:
   - Recommends movies based on their similarity to a target movie.
   - Filters recommendations by rating, type, and categories.

3. **`wordcloud_gen.py`**:
   - Generates word clouds for topics extracted from the Netflix dataset.
   - Saves word clouds as `.png` images for visualization.

4. **`error_plotter.py`**:
   - Plots reconstruction errors for each topic and saves the plots as `.png` images.

5. **`original_scikit.py`**:
   - Demonstrates a basic NMF implementation using scikit-learn.
   - Extracts topics and their top words from the Netflix dataset.


## Installation

To set up the project, simply install the required dependencies using the provided `requirements.txt` file.

### Steps:

1. **Install Dependencies**:
   Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```
# Main.py
This project implements a custom Non-Negative Matrix Factorization (NMF) algorithm to analyze and classify Netflix titles based on their descriptions. The `main.py` script preprocesses the data, applies NMF, and outputs topic models and associated matrices.

## Features

- **Custom NMF Implementation**:
  - Supports two optimization methods: Multiplicative Updates (`mu`) and Alternating Least Squares (`als`).
  - Supports two loss functions: L1 and L2 norms.
  - Initialization options: Random initialization or NNDSVD.

- **Data Preprocessing**:
  - Uses TF-IDF vectorization to transform Netflix title descriptions into a document-term matrix.

- **Topic Modeling**:
  - Extracts topics and their associated words from the Netflix dataset.
  - Saves the topics, weights, and NMF matrices (`W` and `H`) to CSV files.

- **Error Tracking**:
  - Tracks reconstruction error during NMF iterations.
  - Stops early if the error increases for 10 consecutive iterations or falls below a tolerance threshold.

## Script Specific Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `scipy`

Install the required libraries using:
```bash
pip install numpy pandas scikit-learn scipy
```

# wordcloud_gen.py

This script, `wordcloud_gen.py`, generates word clouds for topics extracted from the Netflix dataset using Non-Negative Matrix Factorization (NMF). Each word cloud visually represents the importance of words in a topic based on their weights.

## Features

- **Dynamic Word Cloud Generation**:
  - Generates word clouds for topics based on their word frequencies.
  - Supports multiple configurations of loss functions, topic counts, and optimization methods.

- **Customizable Font**:
  - Allows specifying a custom font for the word cloud.

- **Output as Images**:
  - Saves each word cloud as a `.png` image in the corresponding output directory.

## Script Specific Requirements

- Python 3.x
- Required libraries:
  - `pandas`
  - `wordcloud`
  - `matplotlib`

Install the required libraries using:
```bash
pip install pandas wordcloud matplotlib]
```
# movie_recommender.py Movie Recommender with Non-Negative Matrix Factorization (NMF)

The `movie_recommender.py` script recommends movies based on their similarity to a target movie using Non-Negative Matrix Factorization (NMF). It filters recommendations by rating, type, and categories to ensure relevance.

## Features

- **Movie Similarity Calculation**:
  - Uses cosine similarity to find movies similar to the target movie based on the NMF `W` matrix.

- **Filtering Criteria**:
  - Filters recommendations by:
    - Same rating as the target movie.
    - Same type (e.g., "Movie" or "TV Show").
    - At least one matching category from the `listed_in` column.

- **Customizable Target Movie**:
  - Allows specifying a target movie by its title.

## Script Specific Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`

Install the required libraries using:
```bash
pip install numpy pandas scikit-learn
```
