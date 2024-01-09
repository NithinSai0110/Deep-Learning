import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
actual_labels_dog = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
predicted_labels_dog = np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 0])
conf_matrix_dog = confusion_matrix(actual_labels_dog, predicted_labels_dog)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_dog, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Dog', 'Dog'], yticklabels=['Not Dog', 'Dog'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Dog Classification Confusion Matrix')
data_bc = load_breast_cancer()
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(data_bc.data, data_bc.target, test_size=0.2, random_state=42)
model_bc = LogisticRegression()
model_bc.fit(X_train_bc, y_train_bc)
predicted_labels_bc = model_bc.predict(X_test_bc)
conf_matrix_bc = confusion_matrix(y_test_bc, predicted_labels_bc)
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_bc, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Breast Cancer Classification Confusion Matrix')
plt.tight_layout()
plt.show()