
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay



df = pd.read_csv(r'V:\test\ml lab\User_Datas.csv')  # Provide the full path if necessary

print("Columns in dataset:", df.columns)
target_column = 'Purchased'
feature_columns = ['Gender', 'Age', 'EstimatedSalary']
X = df[feature_columns]
y = df[target_column]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Plot feature distributions
plt.figure(figsize=(12, 6))

# Plot histogram for Age
plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')

# Plot histogram for Estimated Salary
plt.subplot(1, 3, 2)
sns.histplot(df['EstimatedSalary'], kde=True, bins=30)
plt.title('Estimated Salary Distribution')

# Plot bar plot for Gender distribution
plt.subplot(1, 3, 3)
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')

plt.tight_layout()
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

x1 = pd.DataFrame({
    'Age': [36],
    'EstimatedSalary': [9000],
    'Gender': ['Female']
})

# Ensure x1 has the same columns as X_train
x1 = x1.reindex(columns=X.columns, fill_value=0)

# Make prediction
y1 = model.predict(x1)

print(y1)