import matplotlib.pyplot as plt
import pandas as pd

# Given datasets
training = ['sub-234_', 'sub-057_', 'sub-242_', 'sub-132_', 'sub-054_', 'sub-131_', 'sub-022_', 'sub-055_', 'sub-102_', 'sub-229_', 'sub-031_', 'sub-046_', 'sub-115_', 'sub-169_', 'sub-056_', 'sub-248_', 'sub-218_', 'sub-230_', 'sub-181_', 'sub-003_', 'sub-038_', 'sub-206_', 'sub-110_', 'sub-024_', 'sub-107_', 'sub-168_', 'sub-210_', 'sub-104_', 'sub-060_', 'sub-032_', 'sub-184_', 'sub-240_', 'sub-008_', 'sub-001_', 'sub-220_', 'sub-205_', 'sub-197_']
validation = ['sub-188_', 'sub-186_', 'sub-036_', 'sub-130_', 'sub-144_', 'sub-005_', 'sub-199_', 'sub-224_', 'sub-217_', 'sub-192_', 'sub-106_', 'sub-150_', 'sub-017_']
test = ['sub-252_', 'sub-051_', 'sub-125_', 'sub-198_', 'sub-152_', 'sub-243_', 'sub-189_', 'sub-035_', 'sub-129_', 'sub-029_', 'sub-156_', 'sub-209_', 'sub-123_']

training = [t[:-1] for t in training]
validation = [v[:-1] for v in validation]
test = [t[:-1] for t in test]

# Read the data from the Excel file
df = pd.read_csv(r"/home/mwynen/data/cusl_wml/lesions.csv")

# Filter data based on the datasets
df_training = df[df['subject'].isin(training)]
df_validation = df[df['subject'].isin(validation)]
df_test = df[df['subject'].isin(test)]

# Calculate the required metrics for each dataset
def calculate_metrics(data):
    num_lesions = data.groupby('subject').size()
    num_control_lesions = data[~ data['PRL']].groupby('subject').size()
    num_prl_lesions = data[data['PRL']].groupby('subject').size()
    lesion_volume = data.groupby('subject')['size'].mean()
    return num_lesions, num_control_lesions, num_prl_lesions, lesion_volume

training_num_lesions, training_num_control_lesions, training_num_prl_lesions, training_lesion_volume = calculate_metrics(df_training)
validation_num_lesions, validation_num_control_lesions, validation_num_prl_lesions, validation_lesion_volume = calculate_metrics(df_validation)
test_num_lesions, test_num_control_lesions, test_num_prl_lesions, test_lesion_volume = calculate_metrics(df_test)

# Plotting the comparisons
fig, axes = plt.subplots(1, 4, figsize=(12, 8))

# Number of control lesions per patient
axes[0].bar(['Training', 'Validation', 'Test'],
              [training_num_control_lesions.mean(), validation_num_control_lesions.mean(), test_num_control_lesions.mean()])
axes[0].set_ylabel('Avg. Number of Control Lesions')
axes[0].set_title('Average Number of Control Lesions per Patient')

# Number of control lesions per patient
axes[1].bar(['Training', 'Validation', 'Test'],
              [training_num_control_lesions.mean(), validation_num_control_lesions.mean(), test_num_control_lesions.mean()])
axes[1].set_ylabel('Avg. Number of Control Lesions')
axes[1].set_title('Average Number of Control Lesions per Patient')

# Number of PRL per patient
axes[2].bar(['Training', 'Validation', 'Test'],
              [training_num_prl_lesions.mean(), validation_num_prl_lesions.mean(), test_num_prl_lesions.mean()])
axes[2].set_ylabel('Avg. Number of PRL Lesions')
axes[2].set_title('Average Number of PRL Lesions per Patient')

# Lesion volume per patient
lesion_volume_data = [training_lesion_volume, validation_lesion_volume, test_lesion_volume]

# Create a boxplot
axes[3].boxplot(lesion_volume_data, labels=['Training', 'Validation', 'Test'])
axes[3].set_ylabel('Lesion Volume')
axes[3].set_title('Lesion Volume Distribution per Patient')

# Adjust layout and spacing
fig.tight_layout()

# Save the plot
plt.savefig('/home/mwynen/data/cusl_wml/comparison_plot.png')

# Show the plot
plt.show()

