# Written by Ovi
# Date: 2023-10-24
# Program Summary: Convert an ARFF file to CSV format.

import csv
import arff  # You may need to install the 'liac-arff' library using pip

# Input ARFF file name
input_arff_file = 'PhishingData.arff'

# Output CSV file name
output_csv_file = 'PhishingData.csv'

try:
    with open(input_arff_file, 'r') as arff_file:
        # Load the ARFF data
        arff_data = arff.load(arff_file)
        
        # Extract the data and attribute names
        data = arff_data['data']
        attributes = arff_data['attributes']
        
        # Write data to CSV
        with open(output_csv_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write header row with attribute names
            csv_writer.writerow([attr[0] for attr in attributes])
            
            # Write data rows
            csv_writer.writerows(data)
        
    print(f'Conversion completed. ARFF file "{input_arff_file}" converted to CSV file "{output_csv_file}".')

except FileNotFoundError:
    print(f'Error: The input ARFF file "{input_arff_file}" was not found.')

except Exception as e:
    print(f'An error occurred: {str(e)}')
