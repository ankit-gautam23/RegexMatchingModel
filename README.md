Its just to revise the ML concepts I used earlier for small datasets, Now will be creating for varying large datasets.

#Model 1

we first define the regular expression to match against (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'). This regular expression matches email addresses.

Next, we define a small dataset of text strings and their corresponding labels (data). In this case, we use a small set of email addresses as the input data, and label each text string as either a match (1) or non-match (0) for the regular expression.

We then preprocess the input text by extracting features using a bag-of-words approach with scikit-learn's CountVectorizer. This converts each text string into a vector of numerical features that can be input into


#Model 2

We define a dataset of input text strings and their corresponding labels, where each label corresponds to a regular expression pattern. We preprocess the input text using a bag-of-words approach with scikit-learn's CountVectorizer, and train a Naive Bayes classifier on the input text and output labels.

We then test the classifier on some example inputs and print the predictions, which include both the input text string and the regular expression pattern that it matches.

This approach allows you to train a single classifier on a large or constantly changing set of regular expressions, without needing to update the code every time a new regular expression is added or changed
