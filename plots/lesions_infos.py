import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
data = pd.read_excel('D:/R4/labels/lesion_database.xlsx')

# Number of entries with non-empty Confidence_Level_PRL
non_empty_confidence_count = data['Confidence_Level_PRL'].count()
print("Number of entries with non-empty Confidence_Level_PRL:", non_empty_confidence_count)

# Number of PRL vs CTRL lesions
prl_count = data[data['mwid'].between(999, 2000, inclusive='neither')].shape[0]
ctrl_count = data[data['mwid'] >= 2000].shape[0]
print("Number of PRL lesions:", prl_count)
print("Number of CTRL lesions:", ctrl_count)

# Number of discarded lesions
discarded_count = data[(data['mwid'].isnull()) & (data['session'] == 1)].shape[0]
print("Number of discarded lesions:", discarded_count)

# Sum of Tumerous_Lesion and CorticalLesion columns
tumerous_sum = data['Tumerous_Lesion'].sum()
cortical_sum = data['CorticalLesion'].sum()
print("Sum of Tumerous_Lesion:", tumerous_sum)
print("Sum of CorticalLesion:", cortical_sum)

# Generate plots
filtered_data = data.dropna(subset=['mwid'])
plt.figure(figsize=(8, 6))
plt.hist(filtered_data['mwid'], bins=[1000, 2000, max(filtered_data['mwid'])], edgecolor='black')
plt.xlabel('mwid')
plt.ylabel('Count')
plt.title('PRL vs CTRL Lesions')
plt.xticks([1000, 2000, max(filtered_data['mwid'])])
plt.legend(['PRL', 'CTRL'])
plt.savefig('D:/R4/plots/prl_vs_ctrl.png')

plt.figure(figsize=(8, 6))
plt.bar(['Tumerous Lesion', 'Cortical Lesion'], [tumerous_sum, cortical_sum])
plt.xlabel('Lesion Type')
plt.ylabel('Sum')
plt.title('Sum of Tumerous Lesion and Cortical Lesion')
plt.savefig('D:/R4/plots/tumerous_cortical_sum.png')

# Show the plots
plt.show()

