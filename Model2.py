from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Define a dataset of text strings and their labels
# Each label corresponds to a regular expression pattern
data = [
    ('johndoe@example.com', 'email'),
    ('jane.doe@example.com', 'email'),
    ('123-456-7890', 'phone'),
    ('555-555-5555', 'phone'),
    ('http://www.example.com', 'url'),
    ('https://www.example.com', 'url'),
    ('Lorem ipsum dolor sit amet', 'text'),
    ('Ut enim ad minim veniam', 'text'),
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
    '555-555-5555',
    'http://www.example.com',
    'Ut enim ad minim veniam',
]

# Preprocess the test data using the same vectorizer used to preprocess the training data
X_test = vectorizer.transform(test_data)

# Make predictions on the test data using the trained classifier
y_pred = clf.predict(X_test)

# Print the predictions
for i in range(len(test_data)):
    print(f'{test_data[i]} matches the {y_pred[i]} regular expression')
