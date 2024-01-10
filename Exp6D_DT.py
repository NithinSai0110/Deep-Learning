import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Importing the dataset
dataset = pd.read_csv("/content/IRIS.csv")

# Assuming the features are sepal_length, sepal_width, petal_length, and petal_width
features = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target = dataset['species']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Feature Scaling (not necessary for Decision Trees, included for completeness)
# No need for feature scaling for Decision Trees, but you can uncomment the following lines if needed.
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Display the Decision Tree
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)

# Display the decision tree rules
tree_rules = export_text(decision_tree_classifier, feature_names=list(features.columns))
print("Decision Tree Rules:\n", tree_rules)

# Display the decision tree structure (plot)
plt.figure(figsize=(12, 8))
tree.plot_tree(decision_tree_classifier, feature_names=list(features.columns), class_names=dataset['species'].unique(), filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Predicting the Test set results
y_pred = decision_tree_classifier.predict(X_test)

# Display the results (confusion matrix and accuracy)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy: {:.2%}".format(accuracy))
