# Step 1: Import Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Step 2: Load the Dataset
data = pd.read_csv('Heart Attack Analysis & Prediction Dataset\heart.csv')

# Step 3: Data Preprocessing

# Encoding categorical variables if necessary (e.g., Sex, cp, fbs, restecg, exang)
label_encoders = {}
for column in ['sex', 'cp', 'fbs', 'restecg', 'exng', 'caa', 'thall']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Feature Scaling
scaler = StandardScaler()
features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
data[features] = scaler.fit_transform(data[features])

# Step 4: Splitting the Dataset
X = data.drop('output', axis=1)
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection and Training
# Start with a baseline model - Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = logreg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))

# Cross-validation for robustness
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean()}")

# Step 7: Experiment with More Complex Models (e.g., Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Random Forest Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Random Forest Recall: {recall_score(y_test, y_pred_rf)}")
print(f"Random Forest F1-Score: {f1_score(y_test, y_pred_rf)}")
print(f"Random Forest AUC-ROC: {roc_auc_score(y_test, y_pred_rf)}")

# Step 8: Hyperparameter Tuning (for Random Forest example)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

# Step 9: Final Evaluation with Best Model
print(f"Best Random Forest Accuracy: {accuracy_score(y_test, y_pred_best_rf)}")
print(f"Best Random Forest Precision: {precision_score(y_test, y_pred_best_rf)}")
print(f"Best Random Forest Recall: {recall_score(y_test, y_pred_best_rf)}")
print(f"Best Random Forest F1-Score: {f1_score(y_test, y_pred_best_rf)}")
print(f"Best Random Forest AUC-ROC: {roc_auc_score(y_test, y_pred_best_rf)}")

# Step 10: Calculate and Print Heart Attack Risk Statistics
at_risk_count = data['output'].sum()
total_count = data.shape[0]

print(f"Number of people at risk of a heart attack: {at_risk_count}")
print(f"Total number of people in the dataset: {total_count}")
print(f"Percentage of people at risk: {at_risk_count / total_count * 100:.2f}%")

# Step 11: Visualization

# Heart attack risk distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='output', data=data)
plt.title('Heart Attack Risk Distribution')
plt.xlabel('Heart Attack Risk (0 = Low, 1 = High)')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Pairplot of selected features
sns.pairplot(data[['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'output']], hue='output')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

    
