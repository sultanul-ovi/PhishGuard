# Written by Ovi
# Date: 2024-09-13
# This script generates a line chart comparing PhishGuard with other models from various studies using orange shades for the line and markers.

import matplotlib.pyplot as plt

# Step 1: Define the models and their corresponding accuracies
models = ['Al-Sarem et al., \n2021', 'Sarasjati et al., \n2022', 'Abdul Samad et al., \n2023', 'PhishGuard']
accuracies = [98.58, 97.50, 98.27, 99.05]

# Step 2: Create the plot with orange shades for the line and markers
plt.figure(figsize=(10, 6))
# Use an orange shade for the line and a darker orange for the markers
plt.plot(models, accuracies, marker='o', linestyle='-', color='#FFA726', markerfacecolor='#FB8C00', markersize=12, linewidth=4)

# Step 3: Annotate each point with its value (using bold font for annotations)
for i, (model, acc) in enumerate(zip(models, accuracies)):
    plt.text(i, acc, f'{acc:.2f}', fontsize=13, ha='center', va='bottom', fontweight='bold', color='black')

# Step 4: Customize the plot

plt.xlabel('', fontsize=14, fontweight='bold', color='black')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='black')
plt.ylim(95, 100)  # Set y-axis limit from 95 to 100
plt.yticks(range(95, 101, 1), fontsize=12, fontweight='bold', color='black')
plt.xticks(fontsize=12, fontweight='bold', color='black')

# Remove the top and right border (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add a light grid for better visibility
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Step 5: Save the figure
plt.savefig('comparative_analysis_dataset1_orange_black.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

print("Figure saved as 'comparative_analysis_dataset1_orange_black.png'")