import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


data = pd.read_csv('movie.csv')
X = data[['No: of Popular Celebrities', 'Estimated Budget']]
y = data['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=X.columns, class_names=y.unique(), filled=True)
plt.show()
celebrities = int(input("Enter the Number of Celebrities: "))
budget = int(input("Enter the Estimated Budget: "))
input_data = pd.DataFrame({'No: of Popular Celebrities': [celebrities], 'Estimated Budget': [budget]})
prediction = clf.predict(input_data)
print("Predicted Result:", prediction[0])