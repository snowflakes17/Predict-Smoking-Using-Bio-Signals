from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
train_data= train_data.drop(columns=['id'])

train_data.head()
print(train_data.head())

# Check for null values in the entire dataset
null_values = train_data.isnull().sum()

# Display the number of null values for each column
print("Null values in each column:")
print(null_values)

# Check for duplicates in the entire dataset
duplicates = train_data[train_data.duplicated()]

if len(duplicates) == 0:
    print("No duplicates found in the dataset.")
else:
    print("Duplicates in the dataset:")
    print(duplicates)

X = train_data.drop('smoking',axis=1)
y = train_data['smoking']

model = ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances_sorted = feat_importances.sort_values(ascending=False)
feat_importances_sorted.plot(kind='barh')
plt.show()