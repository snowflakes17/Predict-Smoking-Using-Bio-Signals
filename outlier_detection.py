# Step-1: Import necessary dependencies
# IQR Based Filtering (Interquartile Range)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step-2: Read and load the dataset
data = pd.read_csv('train.csv')
data.head()

heat_map_data = data.drop(columns=['id',  'hearing(left)', 'hearing(right)', 'waist(cm)'])
iqr_data = data.drop(columns=['id','smoking'])

# feature importance


# EDA Univariate Analysis
# Analyzing/visualizing the dataset by taking one variable at a time:
# Data visualization is essential; we must decide what charts to plot to better understand the data.

num_cols = data.select_dtypes(include=np.number).columns.tolist()
print("Numerical Variables:")
print(num_cols)

# for col in num_cols:
#     print(col)
#     print('Skew :', round(data[col].skew(), 2))
#     plt.figure(figsize = (15, 4))
#     plt.subplot(1, 2, 1)
#     data[col].hist(grid=False)
#     plt.ylabel('count')
#     plt.subplot(1, 2, 2)
#     sns.boxplot(x=data[col])
#     plt.show()

# Step-3: Plot the distribution plot for the features
# plt.figure(figsize=(20, 20))
# for i, feature in enumerate(data.columns):
#     plt.subplot(5, 5, i+1)
#     sns.distplot(data[feature])
# plt.show()

# Step-4: Form a box-plot for the skewed features
# plt.figure(figsize=(20, 20))
# for i, feature in enumerate(data.columns):
#     plt.subplot(5, 5, i+1)
#     sns.boxplot(data[feature])
# plt.show()

# plt.figure(figsize=(10,6))
# sns.heatmap(heat_map_data.corr(),cmap=plt.cm.Reds,annot=True)
# # plt.title('Heatmap displaying the relationship betweennthe features of the data',
# #
# plt.show()

# Step-5: Find the IQR and upper/lower limits for each feature
outlier_indices = []

# Initialize dictionary to store limits info
limits_info = {}

for feature in iqr_data.columns:
    q1 = iqr_data[feature].quantile(0.25)
    q3 = iqr_data[feature].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outliers = iqr_data[(iqr_data[feature] < lower_limit) | (iqr_data[feature] > upper_limit)].index
    outlier_indices.extend(outliers)

    # Store limits info in the dictionary
    limits_info[feature] = {'Lower Limit': lower_limit, 'Upper Limit': upper_limit}

# Step-6: Remove outliers
cleaned_data = iqr_data.drop(outlier_indices)

# Displaying the upper and lower limits for each feature
for feature, limits in limits_info.items():
    print(f"Feature: {feature}")
    print(f"Lower Limit: {limits['Lower Limit']}, Upper Limit: {limits['Upper Limit']}")

# Step-7: Compare the plots after removing outliers
# plt.figure(figsize=(10, 10))
# for i, feature in enumerate(cleaned_data.columns):
#     plt.subplot(5, 5, i+1)
#     sns.distplot(cleaned_data[feature])
# plt.show()

# plt.figure(figsize=(20, 20))
plt.title('Comparing the plots after removing outliers',fontsize=13)
# for i, feature in enumerate(cleaned_data.columns):
#     plt.subplot(5, 5, i+1)
#
#     sns.boxplot(cleaned_data[feature])
# plt.show()

# Merge 'id' and 'smoking' columns back into cleaned_data based on the index
cleaned_data = cleaned_data.merge(data[['id', 'smoking']], left_index=True, right_index=True)

# Rearrange columns with 'id' as the first column
cleaned_data = cleaned_data[['id'] + [col for col in cleaned_data.columns if col != 'id']]

# Save the cleaned dataset to a new CSV file
cleaned_data.to_csv('cleaned_train.csv', index=False)