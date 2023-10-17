##################
#                #
# Import modules #
#                #
##################

import sentiment_analysis as sa
import utils
import numpy as np

#############
#           #
# Load data #
#           #
#############
train_data = utils.load_data('../data/reviews_train.tsv')
val_data = utils.load_data('../data/reviews_val.tsv')
test_data = utils.load_data('../data/reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = sa.bag_of_words(train_texts)

train_bow_features = sa.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = sa.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = sa.extract_bow_feature_vectors(test_texts, dictionary)

############################################################################################
#                                                                                          #
# Load a toy dataset, apply three different perceptron-based algorithms                    #
# (perceptron, average perceptron, and Pegasos) to train models on the dataset,            #
# and then plot the results of these algorithms, including their decision boundaries.      #
#                                                                                          #
############################################################################################

# Import necessary modules and functions, including 'utils', 'sa' (a module for perceptron algorithms),
# and 'toy_data' (which contains toy dataset and labels).

toy_features, toy_labels = toy_data = utils.load_toy_data('../data/toy_data.tsv')

# Define two constants 'T' and 'L'.
T = 10 # Number of iterations or epochs for training
L = 0.2 # A regularization parameter (for the Pegasos algorithm)

# Apply the perceptron, average perceptron, and Pegasos algorithms to the toy dataset.
thetas_perceptron = sa.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = sa.average_perceptron(toy_features, toy_labels, T)
thetas_pegasos = sa.pegasos(toy_features, toy_labels, T, L)

# Define a function 'plot_toy_results' that takes an algorithm name ('algo_name') 
# and a set of theta values ('thetas') as input.
def plot_toy_results(algo_name, thetas):
    # Print the theta values for the algorithm.
     print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
    # Print the theta_0 (bias) value for the algorithm.
     print('theta_0 for', algo_name, 'is', str(thetas[1]))
    # Plot the toy dataset with decision boundary determined by the algorithm.
     utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

# Call the 'plot_toy_results' function to plot results for each algorithm.
plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Pegasos', thetas_pegasos)


############################################################################################
#                                                                                          #
# Evaluate three perceptron-based algorithms (perceptron, average perceptron, and Pegasos) #
# on a text classification task using different hyperparameters (T and L).                 #
# Calculate and display training and validation accuracies for each algorithm.             #
#                                                                                          #
############################################################################################

T = 10 # Number of iterations or epochs for training
L = 0.01 # A regularization parameter (for the Pegasos algorithm)

# Calculate training and validation accuracies for the perceptron algorithm
pct_train_accuracy, pct_val_accuracy = \
   sa.classifier_accuracy(sa.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)

# Print the training and validation accuracies for the perceptron algorithm
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

# Calculate training and validation accuracies for the average perceptron algorithm
avg_pct_train_accuracy, avg_pct_val_accuracy = \
   sa.classifier_accuracy(sa.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)

# Print the training and validation accuracies for the average perceptron algorithm
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

# Calculate training and validation accuracies for the Pegasos algorithm
avg_peg_train_accuracy, avg_peg_val_accuracy = \
   sa.classifier_accuracy(sa.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
# Print the training and validation accuracies for the Pegasos algorithm.
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))


#################################################################################
#                                                                               #
# Tune hyperparameters and evaluate three perceptron-based algorithms           # 
# (perceptron, average perceptron, and Pegasos) on a text classification task.  #
#                                                                               #
#################################################################################


# Create a tuple 'data' containing training and validation data.
data = (train_bow_features, train_labels, val_bow_features, val_labels)

# Define lists of hyperparameters 'Ts' and 'Ls' to try during tuning.
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]

# Tune perceptron hyperparameter 'T' and report results.
pct_tune_results = utils.tune_perceptron(Ts, *data)
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))

# Tune average perceptron hyperparameter 'T' and report results.
avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))

# Fix 'L' and tune Pegasos hyperparameter 'T', then report results.
fix_L = 0.01
peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))

# Fix 'T' and tune Pegasos hyperparameter 'L', then report results.
fix_T = Ts[np.argmax(peg_tune_results_T[1])]
peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))

# Plot tuning results for each algorithm and hyperparameter.
utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

################################################################################
#                                                                              #
# Use the best method (perceptron, average perceptron or Pegasos) along with   #
# the optimal hyperparameters according to validation accuracies to test       #
# against the test dataset. The test data has been provided as                 #
# test_bow_features and test_labels.                                           #
#                                                                              #
################################################################################

import numpy as np
from main import test_bow_features, test_labels
from sentiment_analysis import extract_words, bag_of_words, extract_bow_feature_vectors, pegasos, classify, accuracy
from utils import load_data

# Adjusting the data unpacking method to process the returned format
def extract_labels_and_texts(data):
    """Extract labels and texts from the loaded data."""
    labels = [item['sentiment'] for item in data]
    texts = [item['text'] for item in data]
    return texts, labels

# Load and unpack the training, validation, and test data
train_data_raw = load_data('../data/reviews_train.tsv')
val_data_raw = load_data('../data/reviews_val.tsv')
test_data_raw = load_data('../data/reviews_test.tsv')

train_data, train_labels = extract_labels_and_texts(train_data_raw)
val_data, val_labels = extract_labels_and_texts(val_data_raw)
test_data, test_labels = extract_labels_and_texts(test_data_raw)

# Create a dictionary using bag-of-words on the training data
dictionary = bag_of_words(train_data)

# Extract feature vectors for the training, validation, and test data
train_feature_matrix = extract_bow_feature_vectors(train_data, dictionary)
val_feature_matrix = extract_bow_feature_vectors(val_data, dictionary)
test_feature_matrix = extract_bow_feature_vectors(test_data, dictionary)

# Hyperparameters
best_T = 25
best_lambda = 0.01

# Train the Pegasos algorithm on the training data
theta, theta_0 = pegasos(train_feature_matrix, train_labels, best_T, best_lambda)

# Predict labels for the test set
test_preds = classify(test_feature_matrix, theta, theta_0)

# Calculate accuracy on the test set
test_accuracy = accuracy(test_preds, test_labels)

print(f"Accuracy on the test set: {test_accuracy:.4f}")


#############################################
#                                           #
#           Remove stop words               #
#                                           #
#############################################

import numpy as np
from sentiment_analysis import extract_words, bag_of_words, extract_bow_feature_vectors, pegasos, classify
from utils import load_data

# Define the accuracy function
def accuracy(preds, targets):
    return (preds == targets).mean()

# Load the stopwords
with open("../data/stopwords.txt", "r") as f:
    stopwords = set(f.read().splitlines())

# Load and unpack the training, validation, and test data
train_data_raw = load_data('../data/reviews_train.tsv')
val_data_raw = load_data('../data/reviews_val.tsv')
test_data_raw = load_data('../data/reviews_test.tsv')

train_data, train_labels = [item['text'] for item in train_data_raw], [item['sentiment'] for item in train_data_raw]
val_data, val_labels = [item['text'] for item in val_data_raw], [item['sentiment'] for item in val_data_raw]
test_data, test_labels = [item['text'] for item in test_data_raw], [item['sentiment'] for item in test_data_raw]

# Create the new dictionary without stopwords
def bag_of_words_no_stopwords(texts):
    """Compute a bag-of-words from a list of texts excluding stopwords."""
    dictionary = {}
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in stopwords and word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

dictionary_no_stopwords = bag_of_words_no_stopwords(train_data)

# Extract feature vectors using the new dictionary
train_feature_matrix_no_stopwords = extract_bow_feature_vectors(train_data, dictionary_no_stopwords)
val_feature_matrix_no_stopwords = extract_bow_feature_vectors(val_data, dictionary_no_stopwords)
test_feature_matrix_no_stopwords = extract_bow_feature_vectors(test_data, dictionary_no_stopwords)

# Train the Pegasos algorithm on the new training feature matrix
theta_no_stopwords, theta_0_no_stopwords = pegasos(train_feature_matrix_no_stopwords, train_labels, 25, 0.01)

# Predict on the test set using the trained model
test_preds_no_stopwords = classify(test_feature_matrix_no_stopwords, theta_no_stopwords, theta_0_no_stopwords)

# Compute the accuracy on the test set
test_accuracy_no_stopwords = accuracy(test_preds_no_stopwords, test_labels)

print(f"Test accuracy with stopwords removed: {test_accuracy_no_stopwords:.4f}")

#############################################
#                                           #
# Change binary features to count features  #
#                                           #
#############################################

import numpy as np
from sentiment_analysis import extract_words, bag_of_words, pegasos, classify
from utils import load_data

# Define the accuracy function
def accuracy(preds, targets):
    return (preds == targets).mean()

# Load the stopwords
with open("../data/stopwords.txt", "r") as f:
    stopwords = set(f.read().splitlines())

# Load and unpack the training, validation, and test data
train_data_raw = load_data('../data/reviews_train.tsv')
val_data_raw = load_data('../data/reviews_val.tsv')
test_data_raw = load_data('../data/reviews_test.tsv')

train_data, train_labels = [item['text'] for item in train_data_raw], [item['sentiment'] for item in train_data_raw]
val_data, val_labels = [item['text'] for item in val_data_raw], [item['sentiment'] for item in val_data_raw]
test_data, test_labels = [item['text'] for item in test_data_raw], [item['sentiment'] for item in test_data_raw]

# Create the new dictionary without stopwords
def bag_of_words_no_stopwords(texts):
    """Compute a bag-of-words from a list of texts excluding stopwords."""
    dictionary = {}
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in stopwords and word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

dictionary_no_stopwords = bag_of_words_no_stopwords(train_data)

# Define the feature extraction function to use counts
def extract_bow_feature_vectors_counts(reviews, dictionary):
    """
    Compute a bag-of-words representation with counts for a list of texts.
    """
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1
    return feature_matrix

# Extract feature vectors using counts and the dictionary without stopwords
train_feature_matrix_counts = extract_bow_feature_vectors_counts(train_data, dictionary_no_stopwords)
val_feature_matrix_counts = extract_bow_feature_vectors_counts(val_data, dictionary_no_stopwords)
test_feature_matrix_counts = extract_bow_feature_vectors_counts(test_data, dictionary_no_stopwords)

# Train the Pegasos algorithm on the new training feature matrix with counts
theta_counts, theta_0_counts = pegasos(train_feature_matrix_counts, train_labels, 25, 0.01)

# Predict on the test set using the trained model
test_preds_counts = classify(test_feature_matrix_counts, theta_counts, theta_0_counts)

# Compute the accuracy on the test set
test_accuracy_counts = accuracy(test_preds_counts, test_labels)

print(f"Test accuracy with stopwords removed and counts features: {test_accuracy_counts:.4f}")

######################################
#                                    #
# Find the most explanatory unigrams #
#                                    #
######################################

import utils

# Train your model (assuming you've done this already and have theta_counts)
theta_counts, _ = pegasos(train_feature_matrix_counts, train_labels, 25, 0.01)

# Find the most explanatory words for positive and negative classification
num_words = 10
positive_word_indices = np.argsort(theta_counts)[-num_words:]
negative_word_indices = np.argsort(theta_counts)[:num_words]

# Using the dictionary to find the actual words
positive_words = [word for word, idx in dictionary_no_stopwords.items() if idx in positive_word_indices]
negative_words = [word for word, idx in dictionary_no_stopwords.items() if idx in negative_word_indices]

#print("Most explanatory words for positive classification:")
for i, word in enumerate(positive_words, 1):
    print(f"Top {i}: {word}")

#print("\nMost explanatory words for negative classification:")
for i, word in enumerate(negative_words, 1):
   print(f"Top {i}: {word}")

##############################################################################
#                                                                            #
# Assign to best_theta, the weights (and not the bias!) learned by the most #
# accurate algorithm with the optimal choice of hyperparameters.             #
#                                                                            #
##############################################################################

T_optimal = 25
L_optimal = 0.01

best_theta, _ = pegasos(train_feature_matrix, train_labels, T_optimal, L_optimal)
wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10])
