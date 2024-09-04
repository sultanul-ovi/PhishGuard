import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the Excel file
file_path = 'heatmap/sorted_model_results6.xlsx'  # Replace with your Excel file path
data = pd.read_excel(file_path)

# Clean column names by stripping leading/trailing whitespaces
data.columns = data.columns.str.strip()

# Ensure there are no leading or trailing spaces in the 'ML Model' column
data['ML Model'] = data['ML Model'].str.strip()

# Remove percentage signs, if any, and multiply the metrics by 100, round to 3 decimal points
for col in ['Accuracy', 'F1 Score', 'Recall', 'Precision']:
    data[col] = (data[col] * 100).round(3)

# Set the Model as the index for easier plotting
data.set_index('ML Model', inplace=True)

# Select the columns you need for the heatmap (Accuracy, F1 Score, Recall, Precision)
metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']

# Generate the heatmap
plt.figure(figsize=(14, 6))  # Set the size of the plot
sns.heatmap(data[metrics], annot=True, cmap='summer_r', linewidths=0.5, fmt=".3f", cbar=False,
            annot_kws={"fontweight": "bold"})  # Make the annotation text bold

# Customize the heatmap
plt.title('Model Performance Metrics Comparison on Dataset 06', fontweight='bold', fontsize=16)
plt.xlabel('Metrics', fontweight='bold', fontsize=12)
plt.ylabel('Models', fontweight='bold', fontsize=12)

# Set bold font for tick labels
plt.xticks(fontweight='bold', fontsize=10)
plt.yticks(fontweight='bold', fontsize=10)

# Save the heatmap with the Excel file's name
image_filename = os.path.splitext(os.path.basename(file_path))[0] + '_heatmap.png'
plt.savefig(image_filename)

# Display the heatmap
plt.show()

print(f"Heatmap saved as {image_filename}")