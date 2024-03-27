import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the training dataset
train_data = pd.read_csv("cleaned_train.csv")

# Read the testing dataset
test_data = pd.read_csv("test.csv")

print("Smoking Data Analysis")

# Specify features to be normalized (X) for training data
X_train = train_data[['id', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
          'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
          'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
          'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', 'smoking']]

# Specify target variable for training data
y_train = train_data['smoking']

# Specify features to be normalized (X) for testing data
X_test = test_data[['id', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
          'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
          'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
          'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']]

# Specify target variable for testing data
y_test = test_data['id']

# Remove 'id' columns from feature set for normalization
X_train = X_train.drop(columns=['id', 'smoking'])
X_test = X_test.drop(columns=['id'])

# Fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# Transform training data
X_train_norm = norm.transform(X_train)

# Transform testing data
X_test_norm = norm.transform(X_test)

# Convert normalized arrays back to DataFrames
X_train_norm_df = pd.DataFrame(X_train_norm, columns=X_train.columns)
X_test_norm_df = pd.DataFrame(X_test_norm, columns=X_test.columns)

# Concatenate 'smoking' column back with normalized features for training data
X_train_norm_df['smoking'] = train_data['smoking']
X_train_norm_df['id'] = train_data['id']

# Concatenate 'id' column back with normalized features for testing data
X_test_norm_df['id'] = y_test.reset_index(drop=True)

# Reorder columns to place 'id' and 'smoking' at the beginning
train_norm_data = X_train_norm_df[['id','age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                                    'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                                    'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                                    'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', 'smoking']]
test_norm_data = X_test_norm_df[['id', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                                  'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                                  'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                                  'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']]

# Save normalized data to CSV files
train_norm_data.to_csv('O_train_cleaned_normalized.csv', index=False)
test_norm_data.to_csv('O_test_cleaned_normalized.csv', index=False)

print("Normalized data saved as train_normalized.csv and test_normalized.csv")
