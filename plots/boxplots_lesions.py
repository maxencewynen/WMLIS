import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from Excel file
# file_path = "D:/R4/lesion_database.xlsx"
file_path = "D:/R4/labels/lesion_database.xlsx"
df = pd.read_excel(file_path)
df = df[df["session"] == 1].dropna(subset=['mwid'])

# Define a function to categorize mwid as PRL or control
def lesion_category(mwid):
    if 1000 <= mwid <= 1999:
        return "PRL"
    elif mwid >= 2000:
        return "CTRL"
    else:
        print(mwid)
        return "Unknown"

# Apply the function to create a new column "category"
df['category'] = df['mwid'].apply(lesion_category)

# Count the number of PRL and control lesions for each subject
count_df = df.groupby(['subject', 'category']).size().reset_index(name='count')

# Pivot the data to create the box plot for number of lesions
pivot_df = count_df.pivot_table(index='subject', columns='category', values='count', fill_value=0)

# Calculate the sum of lesion volumes for each subject and category
volume_df = df.groupby(['subject', 'category'])['Lesion_Volume_ses01'].sum().reset_index()

# Pivot the data to create the box plot for lesion volumes
pivot_volume_df = volume_df.pivot_table(index='subject', columns='category', values='Lesion_Volume_ses01', fill_value=0)

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot the box plot for number of lesions
sns.boxplot(data=pivot_df, ax=axes[0], width=0.5)
axes[0].set_title("Number of PRL and CTRL Lesions per Subject")
axes[0].set_xlabel("Lesion Category")
axes[0].set_ylabel("Number of Lesions")
axes[0].set_xticklabels(['CTRL', 'PRL'])

# Plot the box plot for lesion volumes
sns.boxplot(data=pivot_volume_df, ax=axes[1], width=0.5)
axes[1].set_title("Lesion Volume of PRL and CTRL Lesions per Subject")
axes[1].set_xlabel("Lesion Category")
axes[1].set_ylabel("Lesion Volume")
axes[1].set_xticklabels(['CTRL', 'PRL'])

# Adjust layout to avoid overlapping titles
plt.tight_layout()

# Count all PRL and control lesions in the dataset
total_prl = df[df['category'] == 'PRL'].shape[0]
total_ctrl = df[df['category'] == 'CTRL'].shape[0]

print("Total PRL lesions:", total_prl)
print("Total CTRL lesions:", total_ctrl)

plt.show()
# =============================================================================
# 
# =============================================================================

# Count the number of PRL and control lesions for each subject
count_df = df.groupby(['subject', 'category']).size().reset_index(name='count')

# Pivot the data to create the DataFrame with number of PRL and CTRL lesions for each subject
subject_counts_df = count_df.pivot_table(index='subject', columns='category', values='count', fill_value=0)

# Rename the columns for better readability
subject_counts_df.columns = ['PRL_count', 'CTRL_count']

# Print the DataFrame
print(subject_counts_df)

# Calculate the mean and standard deviation of the volume for PRL and control lesions
volume_stats = df.groupby('category')['Lesion_Volume_ses01'].agg(['mean', 'std'])

# Print the results
print(volume_stats)


train_subjects = ['sub-055', 'sub-063', 'sub-160', 'sub-184', 'sub-230', 'sub-005', 'sub-169', 'sub-164', 'sub-032', 'sub-057', 'sub-193', 'sub-008', 'sub-217', 'sub-110', 'sub-017', 'sub-198', 'sub-125', 'sub-206', 'sub-056', 'sub-114', 'sub-089', 'sub-243', 'sub-107', 'sub-051', 'sub-220', 'sub-061', 'sub-156', 'sub-208']
val_subjects = ['sub-136', 'sub-132', 'sub-062', 'sub-129', 'sub-038', 'sub-106', 'sub-054', 'sub-189', 'sub-059']
test_subjects = ['sub-152', 'sub-205', 'sub-022', 'sub-209', 'sub-031', 'sub-065', 'sub-224', 'sub-210', 'sub-060', 'sub-229']

# Create dataframes for training, validation, and test sets
train_data = df[df['subject'].isin(train_subjects)]
val_data = df[df['subject'].isin(val_subjects)]
test_data = df[df['subject'].isin(test_subjects)]

# Function to create combined plots
def create_combined_plots(train_data, val_data, test_data, title):
    # Add a new column to distinguish datasets
    train_data['dataset'] = 'Training'
    val_data['dataset'] = 'Validation'
    test_data['dataset'] = 'Test'

    # Combine the datasets for plotting
    combined_data = pd.concat([train_data, val_data, test_data])

    # Calculate the count of lesions and volume for each subject and dataset
    count_df = combined_data.groupby(['subject', 'category', 'dataset']).size().reset_index(name='count')
    volume_df = combined_data.groupby(['subject', 'category', 'dataset'])['Lesion_Volume_ses01'].sum().reset_index()

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot the box plot for number of lesions
    sns.boxplot(x='dataset', y='count', hue='category', order=["Training", "Validation", "Test"], data=count_df, ax=axes[0], width=0.5, hue_order=['CTRL', 'PRL'])
    axes[0].set_title(f"Number of PRL and CTRL Lesions per Subject ({title})")
    axes[0].set_xlabel("Dataset", fontsize=14)
    axes[0].set_ylabel("Number of Lesions", fontsize=14)
    axes[0].tick_params(labelsize=12)
    axes[0].legend(title='Lesion Category', fontsize=12, title_fontsize=12)

    # Plot the box plot for lesion volumes
    sns.boxplot(x='dataset', y='Lesion_Volume_ses01', hue='category', order=["Training", "Validation", "Test"], data=volume_df, ax=axes[1], width=0.5, hue_order=['CTRL', 'PRL'])
    axes[1].set_title(f"Lesion Volume of PRL and CTRL Lesions per Subject ({title})")
    axes[1].set_xlabel("Dataset", fontsize=14)
    axes[1].set_ylabel("Lesion Volume (mm³)", fontsize=14)
    axes[1].tick_params(labelsize=12)
    axes[1].legend(title='Lesion Category', fontsize=12, title_fontsize=12)

    # Adjust layout to avoid overlapping titles
    plt.tight_layout()

    # Count all PRL and control lesions in the dataset
    total_prl = df[df['category'] == 'PRL'].shape[0]
    total_ctrl = df[df['category'] == 'CTRL'].shape[0]

    print(f"Total PRL lesions ({title}):", total_prl)
    print(f"Total CTRL lesions ({title}):", total_ctrl)

    # Add a figure caption
    # fig.text(0.5, 0.01, "Figure caption: Add your description here.", ha='center', fontsize=12)

    # Save the plot as a high-resolution image
    plt.savefig('lesion_plot.png', dpi=300, bbox_inches='tight')

    plt.show()

# Call the function to create combined plots
create_combined_plots(train_data, val_data, test_data, "All Datasets")
