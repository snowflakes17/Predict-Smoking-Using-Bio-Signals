import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('LGBM_RESULT_Normalized_Original.csv')

# Filter out rows with smoking values less than 0.5
filtered_data = data[data['smoking'] >= 0.6]

# Plot the frequency distribution
plt.hist(filtered_data['smoking'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Smoking')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Smoking (smoking >= 0.6)')
plt.show()
