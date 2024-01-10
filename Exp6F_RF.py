import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Importing the dataset
dataset = pd.read_csv("/content/IRIS.csv")

# Assuming the features are sepal_length, sepal_width, petal_length, and petal_width
features = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target = dataset['species']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Feature Scaling (not necessary for Random Forest, included for completeness)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Training the Random Forest Classification model on the Training set
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators as needed
random_forest_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = random_forest_classifier.predict(X_test)

# Making the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy: {:.2%}".format(accuracy))