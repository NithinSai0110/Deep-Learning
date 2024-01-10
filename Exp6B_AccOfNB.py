import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
dataset = pd.read_csv("/content/IRIS.csv")
features = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target = dataset['species']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
y_pred = naive_bayes_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')