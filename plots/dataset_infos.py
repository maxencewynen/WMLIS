import matplotlib.pyplot as plt
import pandas as pd

training = ['sub-055_', 'sub-063_', 'sub-160_', 'sub-184_', 'sub-230_', 'sub-005_', 'sub-169_', 'sub-164_', 'sub-032_', 'sub-057_', 'sub-193_', 'sub-008_', 'sub-217_', 'sub-110_', 'sub-017_', 'sub-198_', 'sub-125_', 'sub-206_', 'sub-056_', 'sub-114_', 'sub-089_', 'sub-243_', 'sub-107_', 'sub-051_', 'sub-220_', 'sub-061_', 'sub-156_', 'sub-208_']
validation = ['sub-136_', 'sub-132_', 'sub-062_', 'sub-129_', 'sub-038_', 'sub-106_', 'sub-054_', 'sub-189_', 'sub-059_']
test = ['sub-152_', 'sub-205_', 'sub-022_', 'sub-209_', 'sub-031_', 'sub-065_', 'sub-224_', 'sub-210_', 'sub-060_', 'sub-229_']

training = [t[:-1] for t in training]
validation = [v[:-1] for v in validation]
test = [t[:-1] for t in test]

df = pd.read_excel('D:/R4/labels/lesion_database.xlsx')

# Calculate the counts for each category
training_total = len(df[df['subject'].isin(training)])
training_prl = len(df[(df['subject'].isin(training)) & (df['mwid'].between(999, 2000, inclusive='neither'))])
training_ctrl = len(df[(df['subject'].isin(training)) & (df['mwid'] >= 2000)])

validation_total = len(df[df['subject'].isin(validation)])
validation_prl = len(df[(df['subject'].isin(validation)) & (df['mwid'].between(999, 2000, inclusive='neither'))])
validation_ctrl = len(df[(df['subject'].isin(validation)) & (df['mwid'] >= 2000)])

test_total = len(df[df['subject'].isin(test)])
test_prl = len(df[(df['subject'].isin(test)) & (df['mwid'].between(999, 2000, inclusive='neither'))])
test_ctrl = len(df[(df['subject'].isin(test)) & (df['mwid'] >= 2000)])

# Plotting the pie chart
labels = ['PRL', 'CTRL']
training_counts = [training_prl, training_ctrl]
validation_counts = [validation_prl, validation_ctrl]
test_counts = [test_prl, test_ctrl]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Training Set Pie Chart
training_pie = axes[0].pie(training_counts, labels=[f'{label}: {count}' for label, count in zip(labels, training_counts)], autopct='%1.1f%%', startangle=90)
axes[0].set_title(f'Training Set\nTotal: {training_total}', fontdict={'fontsize': 16})

# Validation Set Pie Chart
validation_pie = axes[1].pie(validation_counts, labels=[f'{label}: {count}' for label, count in zip(labels, validation_counts)], autopct='%1.1f%%', startangle=90)
axes[1].set_title(f'Validation Set\nTotal: {validation_total}', fontdict={'fontsize': 16})

# Test Set Pie Chart
test_pie = axes[2].pie(test_counts, labels=[f'{label}: {count}' for label, count in zip(labels, test_counts)], autopct='%1.1f%%', startangle=90)
axes[2].set_title(f'Test Set\nTotal: {test_total}', fontdict={'fontsize': 16})

# Add a common title and increase font size
fig.suptitle('Lesion category distribution across datasets', fontsize=20, fontweight='bold')

# Adjust layout and spacing
fig.tight_layout()

# Save the plot
plt.savefig('D:/R4/plots/pie_chart.png')

# Show the plot
plt.show()

