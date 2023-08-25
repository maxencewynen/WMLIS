import os
import matplotlib.pyplot as plt

folder_path = r"D:/R4/images"

# Get the file list from the folder (excluding zip files)
file_list = [file for file in os.listdir(folder_path) if file.endswith('.nii.gz')]

# Get the unique sequence names from the file list
sequence_names = set()
subjects = set()
for filename in file_list:
    sequence_names.add(filename[15:-7].replace("reg-T2starw_", "").replace("_reg-T2starw", ""))
    subjects.add(filename[:7])

sequence_names.remove('acq-MPRAGE_wrong_T1w')

sequence_names = sorted(list(sequence_names))
print(sequence_names)
subjects = sorted(list(subjects))
print(subjects)

subject_count = len(subjects)  # Total 
possession_counts = [0] * len(sequence_names)

for subject in subjects:
    for filename in [s for s in file_list if s[:7] == subject]:
        for i, sequence_name in enumerate(sequence_names):
            if sequence_name in filename:
                possession_counts[i] += 1
                break

non_possession_counts = [subject_count - count for count in possession_counts]

# Sort the bars in descending order of available sequences
sorted_indices = sorted(range(len(possession_counts)), key=lambda k: possession_counts[k], reverse=False)
possession_counts = [possession_counts[i] for i in sorted_indices]
non_possession_counts = [non_possession_counts[i] for i in sorted_indices]
sequence_names = [sequence_names[i] for i in sorted_indices]

# Create the stacked bar plot
fig, ax = plt.subplots(figsize=(8, 10))  # Adjust the figure size as desired
bar_height = 0.8

possession_bar = ax.barh(range(len(sequence_names)), possession_counts, bar_height, label='Available')
non_possession_bar = ax.barh(range(len(sequence_names)), non_possession_counts, bar_height,
                            left=possession_counts, label='Not available')

ax.set_ylabel('MRI Sequence', fontsize=12)  # Adjust the font size as desired
ax.set_xlabel('Subject Count', fontsize=12)  # Adjust the font size as desired
ax.set_title('Proportion of Subjects with MRI Sequences', fontsize=14)  # Adjust the font size as desired
ax.set_yticks(range(len(sequence_names)))
ax.set_yticklabels(sequence_names, fontsize=10)  # Adjust the font size as desired
ax.legend()

ax.set_xlim(0, subject_count+5)  # Set the x-axis range

plt.tight_layout()  # Ensures all elements fit within the figure area
plt.show()

