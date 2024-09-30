# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example dataset: text data with corresponding categories
documents = [
    'I love programming in Python',
    'Python is great for machine learning',
    'I hate bugs in my code',
    'Debugging is a useful skill',
    'I love reading books about data science',
    'Data science is an interesting field',
    'I hate slow performance',
    'Machine learning models are amazing'
]

# Labels for each document (0 = negative, 1 = positive)
labels = [1, 1, 0, 1, 1, 1, 0, 1]

# Step 1: Convert text data into numerical feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)  # Convert text to a bag-of-words representation

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Step 3: Initialize the Naive Bayes classifier (MultinomialNB)
naive_bayes = MultinomialNB()

# Step 4: Train the classifier with the training data
naive_bayes.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = naive_bayes.predict(X_test)

# Step 6: Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Step 7: Make predictions on new data (optional)
new_documents = ['I enjoy learning new programming languages', 'I hate debugging code']
new_X = vectorizer.transform(new_documents)
predictions = naive_bayes.predict(new_X)

for doc, category in zip(new_documents, predictions):
    print(f'"{doc}" is classified as: {"positive" if category == 1 else "negative"}')
