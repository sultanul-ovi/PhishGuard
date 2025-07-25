# Written by Ovi on today's date
# This script converts a CSV file to an Excel file.

import pandas as pd

# Load the CSV file
csv_file_path = 'results/sorted_model_results5.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file_path)

# Convert the CSV file to Excel
excel_file_path = 'sorted_model_results5.xlsx'  # Replace with the desired Excel file path
data.to_excel(excel_file_path, index=False)

print(f"CSV file successfully converted to Excel and saved at {excel_file_path}")