"""Letter classification system.

This program includes a possible solution the COM2004/3004 assignment
,including feature dimensionality reduction using PCA, classification 
using the KNN, and word detection in a grid of classified letters.
This system was trained on labeled features vectors, and also applies 
a PCA model to reduce dimensions, as well as using heuristics and a 
score based algorith, to find the closest match when finding words in
the grid.

version: v1.0
"""

from typing import List

import numpy as np
import scipy.linalg
from utils import utils
from utils.utils import Puzzle

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors using a trained PCA model

    The function uses PCA to transform the high dimensional input features into a 
    lower dimension. The "mean_vector" and "principle_components" are provided by 
    the model dictionary from the training stage. 
    
    The input data is centered first 
    using the "mean_vector" and then projects onto PCA space to reduce data 
    dimensionality

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    # loads pca infomation that was learnt during training 
    mean_vector = np.array(model["mean_vector"])
    # contains the pca vectors that was learnt from training 
    principle_components = np.array(model["principle_components"])
    # centers the data using the training data mean 
    centered_data = data - mean_vector
    # projects onto PCA space to reduce data dimensionality
    reduced_data = centered_data @ principle_components
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    This is my PCA based training stage classiffier. It find the dimensionanility 
    reduced representation of the training feature vectors using PCA and stores 
    the model parameters in a dictionary. The returned dictionary stores these 
    models :
        - mean_vector of the training data 
        - principle_comoponents for dimensionality reduction
        - training_data_reduced intion the reduced 
        - labels_train which passes the original training labels

    The PCA steps include :
        - Calculating the mean vector of training data 
        - Centering the data 
        - Computes the eigenvectors and eigenvalues of the 
        covariance matrix
        - The eigenvectors the get normalised 
        - The top 40 principle components gets selected 
        - Eigenvectors gets ordered in descending order of eigenvalues

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
            (n_samples, n_features), one sample per row.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary comtaining the trained model paramters.
    """
    model = {}

    # calculates convariance matrix
    mean_vector = np.mean(fvectors_train, axis=0)
    centered_vectors = fvectors_train - mean_vector

    # calculates the covariance matrix
    covariance_matrix = np.cov(centered_vectors, rowvar=0)
    num_features = covariance_matrix.shape[0]
    
    # calculates eigenvalues/eigen vectors of the covariance matrix
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        covariance_matrix, 
        subset_by_index=(num_features - 40, num_features - 1)
    )
    # flip order to descending order
    eigenvectors = np.fliplr(eigenvectors)
    
    # extracts the eigenvectors corresponding to the selected indices
    principle_components = eigenvectors[:, :num_features - 40]
    # reduced training data 
    # calculates the projected data PCA score by linear transformation
    training_data_reduced = centered_vectors @ principle_components

    # stores model
    model["mean_vector"] = mean_vector.tolist()
    model["principle_components"] = principle_components.tolist()
    model["training_data_reduced"] = training_data_reduced.tolist()
    model["labels_train"] = labels_train.tolist()

    return model



# measures how similar a test letter is to each training letter
def distances(X_train: np.array, test_point: np.ndarray) -> np.ndarray:
    ''' Calculate the distance between training points and test_point
    Inputs:
        X_train: numpy array of shape (N, L)
            Training data with N samples and L features
        test_point: numpy array of shape (L,)
            A single test point for which we want to calculate the distance from

    Outputs:
        distances: numpy array of shape (N,)
            An array of distances from the N training points to the test_point
    '''
    # Actually we could work with the squared distances to avoid the square root
    distances = np.sqrt(np.sum(np.square(X_train - test_point), axis=-1))
    return distances

def KNN(X_train, y_train, test_point, K = 15):
    ''' K Nearest Neighbors algorithm implementation
    Inputs:
        X_train : numpy array of shape (N, L)
            Training data with N samples and L features
        y_train : numpy array of shape (N,)
            Training labels corresponding to the training data
        test_point : numpy array of shape (L,)
            A single test point for which we want to predict the label
        K : int
            The number of nearest neighbors to consider for voting

    Returns:
        label : integer
            The predicted label for the test_point
    '''

    # compute the distance between the test_point and all the training data
    dists = distances(X_train, test_point)

    # find the indices in the training set of K lowest distance values 
    inds = np.argsort(dists)[:K]

    # check the labels from the training set for the indices found above
    labels = y_train[inds]

    # label will be the most common label from the K nearest points
    vals, counts = np.unique(labels, return_counts = True)
    label = vals[np.argmax(counts)]
    return label



def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Dummy implementation of classify squares.

    This function performs the classification stage. Each input function vectors is classified 
    by comparing it to the reduced dimensional training data stored in the model. The KNN alrgorithm 
    is used to assign a albel based on the majority of voting the K training samples.

    The model dictionary contains the PCA reduced training feature vectors and their corresponding labels,
    from the training stage.

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters :
            - training_data_reduced which stores the PCA reduced training feature vectors 
            - labels_train which are the labels taht correspond to the training data 

    Returns:
        List[str]: The predicted class labels for each feature vectors.
    """
    """Classify letter squares using the provided KNN function."""
    
    # extract training from model
    X_train = np.array(model["training_data_reduced"])
    y_train = np.array(model["labels_train"])
    K = 15

    predictions = []

    # Classify each test vector
    for test_point in fvectors_test:
        pred = KNN(X_train, y_train, test_point, K)
        predictions.append(pred)

    return predictions



def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Dummy implementation of find_words.

    This function searches for each target words in the grid of classified letter labels that 
    produced by the classification stage. The words are then searched for in all 8 direction. 
    A score based strategy is used to score the word on how likely it is the best match, if the 
    word becomes out of bounds that position is immedietly skipped. The word position is 
    represented as tuples in the form of (start_row, start_col, end_row, end_col).

    The position with the highest matching score is returned for each each word. If no 
    close match is found a defualt position of (0,0,0,0) is returned.
    
    The model dictionary is provided 


    Args:
        labels (np.ndarray): 2-D array storing the shape containing the 
        characters in each grid cell.
        words (list[str]): A list of words to search in the wordsearch grid.
        model (dict): It is a dictionary containing the model parameters learned 
        during training.

    Returns:
        list[tuple]: A list of tuples  in this format (start_row, start_col, end_row,
        end_col) for each word in the input.
    """

    rows, cols = labels.shape
    found_positions = []

    directions = [
        (0,1),
        (0,-1),
        (1,0),
        (-1,0),
        (1,1),
        (-1,1),
        (1,-1),
        (-1,-1)
    ]

    for word in words:
        word = word.upper()
        word_length = len(word)

        best_match = None
        best_score = -1

        for start_row in range(rows):
            for start_col in range(cols):
                for row_step, col_step in directions:
                
                    end_row = start_row + row_step * (word_length - 1)
                    end_col = start_col + col_step * (word_length - 1)

                    # skips word if it goes out of bounds
                    if not (0 <=end_row < rows and 0 <= end_col < cols):
                        continue

                    matches = 0
                    first_letter_match = False
                    last_letter_match = False
                    
                    for i in range(word_length):
                        row = start_row + row_step * i
                        column = start_col + col_step * i

                        if labels[row,column].upper() == word[i]:
                            matches += 1
                            if i == 0:
                                first_letter_match = True
                            if i == word_length - 1:
                                last_letter_match = True
  
                    # scores potential matches
                    score = matches / word_length

                    extra_points = 0.0
                    if first_letter_match and last_letter_match:
                        extra_points += 0.2
                    elif first_letter_match or last_letter_match:
                        extra_points += 0.15

                    # checks middle letters 
                    if word_length > 4:
                        middle_letter_matches = 0
                        for i in range(1, word_length - 1):
                            row = start_row + row_step * i
                            column = start_col + col_step * i
                            if labels[row,column].upper() == word[i]:
                                middle_letter_matches += 1

                        if middle_letter_matches > (word_length - 2) * 0.4:
                            extra_points += 0.15

                    final_score = score + extra_points

                    # tracks the best score
                    if final_score > best_score:
                        best_score = final_score
                        best_match = (start_row, start_col, end_row, end_col)
            
        if best_match is not None and best_score >= 0.6:
            found_positions.append(best_match)
        else:
            found_positions.append((0, 0, 0, 0))

    return found_positions
