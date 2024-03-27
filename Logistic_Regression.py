import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
pima = pd.read_csv("O_train_normalized.csv")

# Display unique values in each column to identify non-numeric values
for col in pima.columns:
    unique_values = pima[col].unique()
    print(f"Unique values in {col}: {unique_values}")

# Data cleaning: Convert non-numeric values to NaN or handle appropriately
# For example, you might need to remove non-numeric rows or impute missing values
# Drop rows with non-numeric values
pima = pima.apply(pd.to_numeric, errors='coerce').dropna()


# Define feature columns and target variable
feature_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']

# feature_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)',
#                 'systolic', 'relaxation', 'fasting blood sugar',
#                 'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin',
#                 'serum creatinine', 'AST', 'ALT', 'Gtp']

X = pima[feature_cols]
y = pima['smoking']  # Assuming 'smoking' is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Instantiate and fit the model
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)

# Predict
y_pred = logreg.predict(X_test)

# Predict probabilities
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
class_names = ['without smoking', 'with smoking']  # Assuming these are the class names
plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Classification report
target_names = ['without smoking', 'with smoking']
print(classification_report(y_test, y_pred, target_names=target_names))

# Calculate ROC AUC score
lr_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print("lr_auc:", lr_auc)

# TEST WALA DATASET
test_data = pd.read_csv("O_test_normalized.csv")

# Predict probabilities of smoking for the test dataset using logistic regression
y_proba_test = logreg.predict_proba(test_data[feature_cols])[:, 1]

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'id': test_data.index,
    'smoking': y_proba_test
})

print(submission_df.head())
# submission_df.to_csv("LR_RESULT_Cleaned_Normalized.csv", index=False)

