import pandas as pd
import matplotlib.pyplot as plt

# Let's start by loading the uploaded Excel file and examining its structure
file_path = './car_generation.xlsx'
df_uploaded = pd.read_excel(file_path, engine='openpyxl')

# Display the first few rows to understand its structure
df_uploaded.head()

# Filter the DataFrame based on route_id prefixes and plot histograms
prefixes = ['S_to_', 'N_to_', 'E_to_', 'W_to_']
plt.figure(figsize=(10, 8))

for i, prefix in enumerate(prefixes, start=1):
    # Filter rows where 'route_id' starts with the current prefix
    filtered_df = df_uploaded[df_uploaded['route_id'].str.startswith(prefix)]
    
    # Extract 'depart' values for plotting
    depart_values = filtered_df['depart'].values
    
    # Plot histogram for the current prefix
    plt.subplot(2, 2, i)
    plt.hist(depart_values, bins='auto', alpha=0.6, label=f'{prefix} Depart Values')
    plt.title(f'Histogram of Depart Values for {prefix}')
    plt.xlabel('Depart Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()
