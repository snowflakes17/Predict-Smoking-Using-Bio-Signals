import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the training dataset
train_data = pd.read_csv("train.csv")

# Read the testing dataset
test_data = pd.read_csv("test.csv")

print("Smoking Data Analysis")

# Specify features to be standardized (X) for training data
X_train = train_data[['id', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                      'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                      'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                      'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', 'smoking']]

# Specify target variable for training data
y_train = train_data['smoking']

# Specify features to be standardized (X) for testing data
X_test = test_data[['id', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                    'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                    'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                    'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']]

# Specify target variable for testing data
y_test = test_data['id']

# Remove 'id' columns from feature set for standardization
X_train = X_train.drop(columns=['id', 'smoking'])
X_test = X_test.drop(columns=['id'])

# Fit scaler on training data
scaler = StandardScaler().fit(X_train)

# Transform training data
X_train_scaled = scaler.transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Concatenate 'smoking' column back with standardized features for training data
X_train_scaled_df['smoking'] = train_data['smoking']
X_train_scaled_df['id'] = train_data['id']

# Concatenate 'id' column back with standardized features for testing data
X_test_scaled_df['id'] = y_test.reset_index(drop=True)

# Reorder columns to place 'id' and 'smoking' at the beginning
train_scaled_data = X_train_scaled_df[['id','age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                                        'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                                        'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                                        'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', 'smoking']]

test_scaled_data = X_test_scaled_df[['id', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)',
                                      'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                                      'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
                                      'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']]

# Save standardized data to CSV files
train_scaled_data.to_csv('O_train_standardized.csv', index=False)
test_scaled_data.to_csv('O_test_standardized.csv', index=False)

print("Standardized data saved as train_standardized.csv and test_standardized.csv")



