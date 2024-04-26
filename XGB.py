import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

train_data = pd.read_csv("O_train_normalized.csv")
test_data = pd.read_csv("O_test_normalized.csv")

# Drop 'id' column from test data
test_data = test_data.drop(columns=['id'])

feature_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']

X_train = train_data[feature_cols]
y_train = train_data['smoking']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

y_val_pred_proba = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print("ROC-AUC Score on Validation Set:", roc_auc)

y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy on Validation Set:", accuracy)

test_pred_proba = model.predict_proba(test_data)[:, 1]
submission_df = pd.DataFrame({'id': test_data.index, 'smoking': test_pred_proba})
# submission_df.to_csv("submission.csv", index=False)

