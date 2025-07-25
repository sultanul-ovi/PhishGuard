import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Step 1: Define the ML models, datasets, and their accuracies
models = ['SVM', 'Random Forest', 'XGBoost', 'CatBoost', 'AdaBoost', 'Gradient Boosting', 'PhishGuard']
datasets = ['Dataset 01', 'Dataset 02', 'Dataset 03', 'Dataset 04']
accuracy_data = [
    [97.30, 96.61, 96.19, 81.65],
    [98.60, 97.02, 97.11, 95.13],
    [99.00, 96.70, 97.11, 94.97],
    [98.65, 97.24, 97.20, 94.06],
    [97.50, 93.80, 95.93, 87.78],
    [98.85, 97.11, 97.29, 94.79],
    [99.05, 97.29, 97.33, 95.17]
]

# Step 2: Create DataFrame for heatmap
df = pd.DataFrame(accuracy_data, index=models, columns=datasets)

# Step 3: Create a figure with 4 subplots (1 row, 4 columns) to represent each dataset
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# Step 4: Loop through each dataset to generate a heatmap for each
for i, dataset in enumerate(datasets):
    # Select the data for the current dataset
    data = df[[dataset]]

    # Normalize the data for the heatmap
    data_normalized = (data - data.min()) / (data.max() - data.min())

    # Generate the heatmap with a distinct range for each value
    sns.heatmap(data_normalized, annot=data, cmap="summer_r", linewidths=0.5, fmt=".2f", 
                ax=axes[i], cbar=False, vmin=0, vmax=1)

    # Customize each subplot
    axes[i].set_title(dataset, fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('ML Models', fontsize=12, fontweight='bold')

# Step 5: Adjust the layout to make sure the subplots fit well
plt.tight_layout(pad=2.0)

# Save the combined figure
plt.savefig('combined_heatmap_distinct_colors.png', dpi=300)

# Display the combined heatmap
plt.show()

print("Combined heatmap saved as 'combined_heatmap_distinct_colors.png'")