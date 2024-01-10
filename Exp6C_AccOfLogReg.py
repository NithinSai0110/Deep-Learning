import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
dataset = pd.read_csv("/content/IRIS.csv")

# Assuming the features are sepal_length, sepal_width, petal_length, and petal_width
features = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target = dataset['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the Logistic Regression classifier
logistic_regression_classifier = LogisticRegression(max_iter=1000)  # You can adjust the max_iter parameter as needed

# Train the classifier on the training set
logistic_regression_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logistic_regression_classifier.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
