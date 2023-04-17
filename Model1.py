import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Define the regular expression to match against
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Define a dataset of text strings and their labels
# In this example, we'll use a small set of email addresses
# You could replace this with your own dataset of text strings and their corresponding labels
data = [
    ('johndoe@example.com', 1),
    ('jane.doe@example.com', 1),
    ('johndoe', 0),
    ('example.com', 0),
    ('johndoe@subdomain.example.com', 1),
    ('johndoe@example.co.uk', 1),
    ('johndoe@example', 0),
    ('johndoe@.example.com', 0),
]

# Split the data into input (text) and output (label) arrays
texts, labels = zip(*data)

# Preprocess the input text by extracting features using a bag-of-words approach
vectorizer = CountVectorizer(lowercase=False, token_pattern=r'\b\w+\b')
X = vectorizer.fit_transform(texts)

# Train a Naive Bayes classifier on the input text and output labels
clf = MultinomialNB()
clf.fit(X, labels)

# Test the classifier on some example inputs
test_data = [
    'johndoe@example.com',
    'jane.doe@example.com',
    'johndoe',
    'example.com',
    'johndoe@subdomain.example.com',
    'johndoe@example.co.uk',
    'johndoe@example',
    'johndoe@.example.com',
]

# Preprocess the test data using the same vectorizer used to preprocess the training data
X_test = vectorizer.transform(test_data)

# Make predictions on the test data using the trained classifier
y_pred = clf.predict(X_test)

# Print the predictions
for i in range(len(test_data)):
    if y_pred[i] == 1:
        print(f'{test_data[i]} matches the regex')
    else:
        print(f'{test_data[i]} does not match the regex')
