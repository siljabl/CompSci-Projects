import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Excel file
sheet_name = 'LearningRate'
df = pd.read_excel('TrainingResults.xlsx', sheet_name=sheet_name)

# Get the variable names (column names) from the DataFrame
variable_names = df.columns.tolist()
variable_names.remove('epochnumber')  # Remove 'epochnumber' from variable names

# Plotting
plt.figure(figsize=(10, 6))  # Set the figure size

# Define line styles and markers for each variable
line_styles = ['-', '--', ':', '-.']
markers = ['o', 's', '^', 'D']

# Plot the variables with different line styles and markers
for i, variable in enumerate(variable_names):
    sns.lineplot(x=df['epochnumber'], y=df[variable], label=variable, linestyle=line_styles[i], marker=markers[i], markersize=4)

# Set plot title and labels
plt.title(sheet_name)
plt.xlabel('Epoch Number')
plt.ylabel('Validation Accuracy')

# Display legend
plt.legend()

plt.savefig(sheet_name + '.png', dpi=300)  # Adjust the DPI value as needed

# Show the plot
plt.show()
