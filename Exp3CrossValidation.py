import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
svm_classifier = SVC(kernel='linear')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(svm_classifier, X, y, cv=cv, scoring='accuracy')

# Display cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))