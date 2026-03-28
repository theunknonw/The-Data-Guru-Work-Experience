# Word Search Solver

A machine learning pipeline for automatically solving word search puzzles from images. Built using PCA for dimensionality reduction and K-Nearest Neighbours for letter classification.

---

## Overview

This project tackles the problem of automating word search puzzle solving end-to-end — from raw puzzle images through to located word positions. The system:

1. **Extracts features** from letter images using pixel intensities
2. **Reduces dimensions** via PCA (Principal Component Analysis)
3. **Classifies letters** using a K-Nearest Neighbours (KNN) classifier
4. **Finds words** in the classified letter grid using an 8-directional score-based search

---

## Performance

| Data Quality | Letters Correct | Words Correct |
|---|---|---|
| High | 98.7% | 100.0% |
| Low | 52.7% | 50.0% |

---

## Project Structure

```
├── system.py           # Main pipeline: feature extraction, PCA, KNN, word finder
├── model_high.json     # Trained model parameters for high-quality data
├── model_low.json      # Trained model parameters for low-quality data
├── report.txt          # Assignment report
└── utils/
    └── utils.py        # Utility functions (image loading, puzzle handling)
```

---

## How It Works

### 1. Feature Extraction
Each puzzle image is centred and cropped to remove background noise, then flattened into a raw pixel intensity vector. This preserves the shape and structure of each letter.

### 2. Dimensionality Reduction (PCA)
Raw pixel vectors are high-dimensional, so PCA is used to compress them into a 20-dimensional feature vector:
- Training data is mean-centred
- Covariance matrix is computed
- Top eigenvectors are selected (ordered by eigenvalue magnitude)
- Both train and test data are projected onto the same PCA subspace

### 3. Letter Classification (KNN)
Each reduced feature vector is classified using K-Nearest Neighbours (K=15):
- Euclidean distance is computed against all training samples
- The K closest neighbours vote on the predicted label
- Majority vote determines the final class

### 4. Word Finding
Words are searched in all 8 directions across the classified letter grid. A score-based system handles potential misclassifications:
- Base score: fraction of letters matched
- Bonus points for matching first and last letters
- Extra credit for matching middle letters in longer words
- Positions extending outside the grid are skipped
- Default position `(0, 0, 0, 0)` is returned if no match clears a minimum threshold

---

## Setup & Usage

### Requirements

```bash
pip install numpy scipy
```

### Running the System

The system integrates with the assignment framework via `utils.py`. The key entry points are:

```python
from system import process_training_data, reduce_dimensions, classify_squares, find_words

# Train the model
model = process_training_data(fvectors_train, labels_train)

# Reduce test features
fvectors_test_reduced = reduce_dimensions(fvectors_test, model)

# Classify letters
predicted_labels = classify_squares(fvectors_test_reduced, model)

# Find words in the grid
positions = find_words(label_grid, word_list, model)
```

---

## Design Decisions

**Why PCA?** PCA is an unsupervised transformation, so it avoids introducing class-label bias at the feature extraction stage. It also makes the classifier more robust by removing noise dimensions.

**Why KNN?** KNN naturally supports multi-class classification, requires no parametric assumptions, and with PCA-reduced features performs reliably without complex tuning.

**Why K=15?** A larger K smooths the decision boundary and reduces sensitivity to individual noisy training samples, which matters for letter recognition where some characters look visually similar.

**Score-based word finding** accounts for the reality that classifiers are imperfect — rather than requiring an exact match, the system picks the most plausible position even under partial misclassification.

---

## Author

COM2004/3004 Assignment — University of Sheffield
