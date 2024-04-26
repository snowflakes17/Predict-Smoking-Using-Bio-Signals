import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

# Load dataset
pima = pd.read_csv("O_train_cleaned_normalized.csv")

# Data cleaning: Convert non-numeric values to NaN or handle appropriately
# For example, you might need to remove non-numeric rows or impute missing values
# Drop rows with non-numeric values
pima = pima.apply(pd.to_numeric, errors='coerce').dropna()

# Define feature columns and target variable
# feature_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
#                 'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
#                 'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
#                 'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']

feature_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                 'systolic', 'relaxation', 'fasting blood sugar',
                'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin',
                'serum creatinine', 'AST', 'ALT', 'Gtp']

X = pima[feature_cols]
y = pima['smoking']  # 'smoking' is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Define LightGBM parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Train LightGBM model
num_round = 500
# bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10, verbose_eval=False)
# evals_result = {}  # To store evaluation results
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], callbacks=[lgb.early_stopping(stopping_rounds=10)])

# bst = lgb.train(params, train_data, num_round, valid_sets=[train_data, test_data], evals_result=evals_result, early_stopping_rounds=10)
# bst = lgb.train(params, train_data, num_round, valid_sets=[train_data, test_data], early_stopping_rounds=10)


# Predict
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

# Confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred_binary)

# Visualize confusion matrix
class_names = ['without smoking', 'with smoking']  # Assuming these are the class names
plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
# lgb.plot_tree(bst, tree_index=0, figsize=(30, 40))


# Classification report
print(classification_report(y_test, y_pred_binary))

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC:", roc_auc)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# TEST WALA DATASET
test_data = pd.read_csv("O_test_cleaned_normalized.csv")

# Predict probabilities of smoking for the test dataset using LightGBM
y_proba_test = bst.predict(test_data[feature_cols])

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'id': test_data.index,
    'smoking': y_proba_test
})

print(submission_df.head())
# submission_df.to_csv("LGBM_RESULT_standardized_Org.csv", index=False)


