
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
# Read the training data
df_train = pd.read_csv("O_train_cleaned_normalized.csv")

# Define feature columns and target variable
feature_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']
X = df_train[feature_cols]
y = df_train['smoking']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Define the number of features
num_features = 22

# Define the model
# 7 hidden layers
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(num_features,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=250, verbose=1, batch_size=150)


# Predict probabilities on the test set
prediction_DNN = model.predict(X_test)

# Calculate and print the AUC score
auc = roc_auc_score(y_test, prediction_DNN)
print("AUC Score:", auc)

# Evaluate the model on the test set and print the final accuracy
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Final Accuracy:", accuracy)

y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)
y_pred

from sklearn.metrics import accuracy_score
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# Read the test data
df_test = pd.read_csv("O_test_cleaned_normalized.csv")  # Replace "test.csv" with the path to your test CSV file

# Predict probabilities on the test set
prediction_DNN = model.predict(df_test[feature_cols])

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Predict probabilities on the test set
y_pred_proba = model.predict(X_test)
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score:", roc_auc)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Create a DataFrame with the predictions
output_df = pd.DataFrame({
    'id': df_test['id'],  # Assuming 'id' is the column name for IDs in the test data
    'smoking': prediction_DNN.flatten()  # Assuming prediction_DNN is a 1D array
})

# Save the predictions to a CSV file
# output_df.to_csv("predictions.csv", index=False)
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)


# Print the top 5 lines of the output file
print(output_df.head())





