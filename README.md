# Netflix Classifier with Non-Negative Matrix Factorization (NMF)

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

## Requirements

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

or

## Installation

To set up the project, simply install the required dependencies using the provided `requirements.txt` file.

### Steps:

1. **Install Dependencies**:
   Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

This script, `wordcloud_gen.py`, generates word clouds for topics extracted from the Netflix dataset using Non-Negative Matrix Factorization (NMF). Each word cloud visually represents the importance of words in a topic based on their weights.

## Features

- **Dynamic Word Cloud Generation**:
  - Generates word clouds for topics based on their word frequencies.
  - Supports multiple configurations of loss functions, topic counts, and optimization methods.

- **Customizable Font**:
  - Allows specifying a custom font for the word cloud.

- **Output as Images**:
  - Saves each word cloud as a `.png` image in the corresponding output directory.

## Requirements

- Python 3.x
- Required libraries:
  - `pandas`
  - `wordcloud`
  - `matplotlib`

Install the required libraries using:
```bash
pip install pandas wordcloud matplotlib]
```
# Movie Recommender with Non-Negative Matrix Factorization (NMF)

The `movie_reccomender.py` script recommends movies based on their similarity to a target movie using Non-Negative Matrix Factorization (NMF). It filters recommendations by rating, type, and categories to ensure relevance.

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

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`

Install the required libraries using:
```bash
pip install numpy pandas scikit-learn
```
