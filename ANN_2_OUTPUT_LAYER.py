import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Read the training data
df_train = pd.read_csv("O_train_normalized.csv")

# Define feature columns and target variable
feature_cols = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']
X = df_train[feature_cols]
y = df_train['smoking']

# Convert target variable to one-hot encoded format
y = pd.get_dummies(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Define the number of features
num_features = 22

# Define the model with two output layers
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
model.add(Dense(2, activation='sigmoid'))  # Two output layers

# Compile the model
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1, batch_size=150)

# Predict probabilities on the test set
predictions = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test.values.argmax(axis=1), predictions.argmax(axis=1))
print("Accuracy:", accuracy)

# Read the test data
df_test = pd.read_csv("O_test_normalized.csv")

# Predict probabilities on the test set
predictions_test = model.predict(df_test[feature_cols])

# Create a DataFrame with the predictions
output_df = pd.DataFrame({
    'id': df_test['id'],  # Assuming 'id' is the column name for IDs in the test data
    'smoking_value1': predictions_test[:, 0],  # Prediction for Smoking value1
    'smoking_value2': predictions_test[:, 1]   # Prediction for Smoking value2
})

# Save the predictions to a CSV file
output_df.to_csv("predictions.csv", index=False)

# Print the top 5 lines of the output file
print(output_df.head())
