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

# Filter data based on the datasets
df_training = df[df['subject'].isin(training)]
df_validation = df[df['subject'].isin(validation)]
df_test = df[df['subject'].isin(test)]

# Calculate the required metrics for each dataset
def calculate_metrics(data):
    num_control_lesions = data[data['mwid'] >= 2000].groupby('subject').size()
    num_prl_lesions = data[data['mwid'].between(999, 2000, inclusive='neither')].groupby('subject').size()
    num_genders = data.groupby('subject')['sex'].value_counts().unstack(fill_value=0)
    lesion_volume = data.groupby('subject')['Lesion_Volume_ses01'].sum()
    return num_control_lesions, num_prl_lesions, num_genders, lesion_volume

training_num_control_lesions, training_num_prl_lesions, training_num_genders, training_lesion_volume = calculate_metrics(df_training)
validation_num_control_lesions, validation_num_prl_lesions, validation_num_genders, validation_lesion_volume = calculate_metrics(df_validation)
test_num_control_lesions, test_num_prl_lesions, test_num_genders, test_lesion_volume = calculate_metrics(df_test)

# Plotting the comparisons
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Number of control lesions per patient
axes[0, 0].bar(['Training', 'Validation', 'Test'],
              [training_num_control_lesions.mean(), validation_num_control_lesions.mean(), test_num_control_lesions.mean()])
axes[0, 0].set_ylabel('Avg. Number of Control Lesions')
axes[0, 0].set_title('Average Number of Control Lesions per Patient')

# Number of PRL per patient
axes[0, 1].bar(['Training', 'Validation', 'Test'],
              [training_num_prl_lesions.mean(), validation_num_prl_lesions.mean(), test_num_prl_lesions.mean()])
axes[0, 1].set_ylabel('Avg. Number of PRL Lesions')
axes[0, 1].set_title('Average Number of PRL Lesions per Patient')

# Number of each gender per patient
axes[1, 0].bar(['Training', 'Validation', 'Test'], [training_num_genders[0].sum(), validation_num_genders[0].sum(), test_num_genders[0].sum()])
axes[1, 0].bar(['Training', 'Validation', 'Test'], [training_num_genders[1].sum(), validation_num_genders[1].sum(), test_num_genders[1].sum()], bottom=[training_num_genders[0].sum(), validation_num_genders[0].sum(), test_num_genders[0].sum()])

axes[1, 0].set_ylabel('Number of Patients')
axes[1, 0].set_title('Number of Patients by Gender')
axes[1, 0].legend(['Male', 'Female'])

# Lesion volume per patient
axes[1, 1].bar(['Training', 'Validation', 'Test'],
              [training_lesion_volume.mean(), validation_lesion_volume.mean(), test_lesion_volume.mean()])
axes[1, 1].set_ylabel('Avg. Lesion Volume')
axes[1, 1].set_title('Average Lesion Volume per Patient')

# Adjust layout and spacing
fig.tight_layout()

# Save the plot
plt.savefig('D:/R4/plots/comparison_plot.png')

# Show the plot
plt.show()

