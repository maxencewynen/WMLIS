import matplotlib.pyplot as plt
import pandas as pd

# Given datasets
training = ['sub-055_', 'sub-063_', 'sub-160_', 'sub-184_', 'sub-230_', 'sub-005_', 'sub-169_', 'sub-164_', 'sub-032_', 'sub-057_', 'sub-193_', 'sub-008_', 'sub-217_', 'sub-110_', 'sub-017_', 'sub-198_', 'sub-125_', 'sub-206_', 'sub-056_', 'sub-114_', 'sub-089_', 'sub-243_', 'sub-107_', 'sub-051_', 'sub-220_', 'sub-061_', 'sub-156_', 'sub-208_']
validation = ['sub-136_', 'sub-132_', 'sub-062_', 'sub-129_', 'sub-038_', 'sub-106_', 'sub-054_', 'sub-189_', 'sub-059_']
test = ['sub-152_', 'sub-205_', 'sub-022_', 'sub-209_', 'sub-031_', 'sub-065_', 'sub-224_', 'sub-210_', 'sub-060_', 'sub-229_']

training = [t[:-1] for t in training]
validation = [v[:-1] for v in validation]
test = [t[:-1] for t in test]

# Read the data from the Excel file
df = pd.read_excel('D:/R4/labels/lesion_database.xlsx')

# Combine all datasets into a single DataFrame
combined_df = pd.concat([df[df['subject'].isin(training)],
                        df[df['subject'].isin(validation)],
                        df[df['subject'].isin(test)]])

# Calculate the average volume for CTRL and PRL lesions
avg_volume_ctrl = combined_df[combined_df['mwid'] >= 2000]['Lesion_Volume_ses01'].mean()
avg_volume_prl = combined_df[combined_df['mwid'].between(999, 2000, inclusive='neither')]['Lesion_Volume_ses01'].mean()

# Plot the average volume for CTRL and PRL lesions
fig, ax = plt.subplots()
ax.bar(['Control (CTRL)', 'Posterior Reversible Leukoencephalopathy (PRL)'],
       [avg_volume_ctrl, avg_volume_prl])
ax.set_ylabel('Average Lesion Volume')
ax.set_title('Average Lesion Volume for CTRL and PRL Lesions')
plt.xticks(rotation=45)

# Save the plot
plt.savefig('D:/R4/plots/average_volume_ctrl_prl.png')

# Show the plot
plt.show()

