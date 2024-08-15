import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv(r'SWELL-Dataset\hrvdataset\data\final\train.csv')
test_df = pd.read_csv(r'SWELL-Dataset\hrvdataset\data\final\test.csv')

print("Datasets loaded successfully.")

# Handle missing values
print("Handling missing values...")
train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

# Encode the condition variable
print("Encoding condition variable...")
label_encoder = LabelEncoder()
train_df['condition_encoded'] = label_encoder.fit_transform(train_df['condition'])
test_df['condition_encoded'] = label_encoder.transform(test_df['condition'])

# Define high stress as 'time pressure'
high_stress_label = label_encoder.transform(['time pressure'])[0]

# Prepare data
print("Preparing data...")
features = [
    'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR', 'pNN25', 'pNN50', 
    'SD1', 'SD2', 'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 
    'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR', 'VLF', 'VLF_PCT', 'LF', 
    'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 'HF_LF', 'sampen', 'higuci'
]

X = train_df[features]
y = (train_df['condition_encoded'] == high_stress_label).astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Define models with default hyperparameters
models = {
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
}

# Train models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

# Generate predictions
    y_pred = model.predict(X_valid)  # This line generates predictions for the validation set

    # Print results
    print(f"\n{model_name} Classification Report:\n", classification_report(y_valid, y_pred))
    print(f"{model_name} Accuracy Score:", accuracy_score(y_valid, y_pred))
    
    # Print the number of instances classified as high stress
    high_stress_count = (y_pred == 1).sum()
    print(f"Number of instances classified as high stress with {model_name}: {high_stress_count}")
    
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_valid, models['GradientBoosting'].predict(X_valid), 'GradientBoosting')

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(y_valid, models['GradientBoosting'].predict_proba(X_valid)[:, 1], 'GradientBoosting')

def plot_precision_recall_curve(y_true, y_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.show()

plot_precision_recall_curve(y_valid, models['GradientBoosting'].predict_proba(X_valid)[:, 1], 'GradientBoosting')

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

plot_feature_importance(models['GradientBoosting'], features)







