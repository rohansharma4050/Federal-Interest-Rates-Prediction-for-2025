import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data (replace with the actual file path)
file_path = "finaldata.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Drop the 'Date' column
data = data.drop(columns=["Date"])

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()
