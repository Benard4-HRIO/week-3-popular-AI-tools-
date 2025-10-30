# iris_decision_tree.py
# Classical ML with scikit-learn: preprocess, train Decision Tree, evaluate (accuracy, precision, recall)
# Author: (your name)
# Run: python iris_decision_tree.py
# If using a notebook, split by comments into cells.

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------
# 1) Load dataset
# -----------------------
iris = load_iris()
X = iris.data          # numeric features
y = iris.target        # numeric labels (0,1,2)
feature_names = iris.feature_names
target_names = iris.target_names

# Put data into DataFrame for clarity and easy inspection
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

print("First 5 rows of the dataset:")
print(df.head())

# -----------------------
# 2) Check for missing values
# -----------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# Demonstrate how to handle missing values if present:
# (Iris doesn't normally have missing values; here we show a general approach)
# If you had missing values in features, you could do:
imputer = SimpleImputer(strategy='mean')  # numeric features: replace missing by mean

# Example: (commented out) if you had missing values:
# X_imputed = imputer.fit_transform(X)

# For completeness, we'll run imputer on X (it will do nothing for complete data)
X = imputer.fit_transform(X)

# -----------------------
# 3) Encode labels
# -----------------------
# In this dataset, y is already numeric (0,1,2). If labels were strings:
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(string_labels)
# But let's keep both numeric and human-readable for reporting:
y_numeric = y
y_names = np.array(target_names)[y_numeric]

# -----------------------
# 4) Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_numeric, test_size=0.25, random_state=42, stratify=y_numeric
)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# -----------------------
# 5) Feature scaling (optional)
# -----------------------
# Decision Trees don't need scaling, but if you wanted to compare with other models:
scaler = StandardScaler()
# Uncomment if you want scaling:
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# For Decision Tree we'll use unscaled features:
X_train_used = X_train
X_test_used = X_test

# -----------------------
# 6) Train Decision Tree
# -----------------------
clf = DecisionTreeClassifier(random_state=42, max_depth=4)  # max_depth optional
clf.fit(X_train_used, y_train)

# -----------------------
# 7) Predict & Evaluate
# -----------------------
y_pred = clf.predict(X_test_used)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Precision and recall (multiclass) - show per-class and macro average
precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

print("\nPrecision per class:", precision_per_class)
print("Recall per class   :", recall_per_class)
print(f"Precision (macro avg): {precision_macro:.4f}")
print(f"Recall    (macro avg): {recall_macro:.4f}")

# Full classification report (precision, recall, f1-score per class)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# -----------------------
# 8) Plot confusion matrix and tree
# -----------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix heatmap (simple)
ax[0].imshow(cm, interpolation='nearest')
ax[0].set_title('Confusion matrix')
ax[0].set_xticks(np.arange(len(target_names)))
ax[0].set_yticks(np.arange(len(target_names)))
ax[0].set_xticklabels(target_names, rotation=45, ha='right')
ax[0].set_yticklabels(target_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax[0].text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2. else 'black')

# Decision Tree plot
plot_tree(clf, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, ax=ax[1], fontsize=8)
ax[1].set_title('Decision Tree')

plt.tight_layout()
plt.show()

# -----------------------
# 10) Feature importance
# -----------------------
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

# Optional: plot them
plt.figure(figsize=(6,4))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# -----------------------
# 9) Save model (optional)
# -----------------------
# If you want to save the trained model:
# import joblib
# joblib.dump(clf, 'decision_tree_iris.joblib')
# To load later:
# clf2 = joblib.load('decision_tree_iris.joblib')

# End of script
