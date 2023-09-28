import pandas as pd
from os.path import join as pjoin
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load your CSV data into a DataFrame
directory = "/home/mwynen/data/cusl_wml"

def make_metrics_dfs_from_subjects_df(df):
    df['dworkin'] = df['Method'].str.contains('\+dworkin')
    # Escape the '+' character in the replace function
    df['Method'] = df['Method'].str.replace('\+dworkin', '', regex=True)

    # Set the index to a combination of 'Method' and 'Subject_ID' columns
    df.set_index(['Method', 'Subject_ID'], inplace=True)

    # Create separate DataFrames for each metric
    metrics = ['PQ', 'Fbeta', 'LTPR', 'PPV', 'Dice_Per_TP', 'DiC', 'Dice']
    metric_dfs = {}

    for metric in metrics:
        # Pivot the DataFrame to have 'dworkin' as columns, metric as values, and 'Method' as index
        pivot_df = df.pivot(columns='dworkin', values=metric).reset_index()
        
        # Rename the columns
        pivot_df.columns = ['Method', 'Subject_ID', f'{metric}', f'{metric}(Dworkin)']
        pivot_df = pivot_df.dropna() 
        # Store the resulting DataFrame in the dictionary
        metric_dfs[metric] = pivot_df
        
        pivot_df.to_csv(pjoin(directory, f"dworkin_comparison_{metric}.csv"), index=False)
    return metric_dfs


df = pd.read_csv(pjoin(directory, 'metrics_comparison_all_subjects.csv'))
metrics = make_metrics_dfs_from_subjects_df(df)

#for metric, df in metrics.items():
#    # Compute the differences between metric_for_X and metric_for_Y
#    df['difference'] = df[f'{metric}(Dworkin)'] - df[f'{metric}'] 
#
#    # Set the style for the plot (optional but can make the plot more visually appealing)
#    sns.set(style="whitegrid")
#
#    # Create a histogram
#    plt.figure(figsize=(8, 6))  # Set the figure size
#    sns.histplot(df['difference'], bins=15, kde=True)  # Create the histogram with a KDE plot
#    plt.xlabel(f'Differences in {metric} (with Dworkin - without Dworkin)')  # Label the x-axis
#    plt.ylabel('Frequency')  # Label the y-axis
#    plt.title(f'Histogram of Differences in {metric} (with Dworkin - without Dworkin)')  # Add a title
#    plt.show()  # Display the plot
#
#    plt.savefig(pjoin(directory, f'histogram_{metric}.png'), dpi=300, bbox_inches='tight')
#

def ttests(df, metric):
    if metric != "DiC": 
        differences = df[f"{metric}(Dworkin)"] - df[metric]

        t_stat, p_value = stats.ttest_1samp(differences, popmean=0, alternative='less')

        alpha = 0.05  # Set your desired significance level
        if p_value < alpha:
            print(f"Reject the null hypothesis. There is a significant negative effect of Dworkin on {metric} (p =", '{0:.3}'.format(p_value), ").")
        else:
            print(f"Fail to reject the null hypothesis. There is no significant negative effect of Dworkin on {metric} (p =", '{0:.3}'.format(p_value), ").")

    #if metric == "DiC" or metric == "LTPR":
    differences = df[metric] - df[f"{metric}(Dworkin)"]

    t_stat, p_value = stats.ttest_1samp(differences, popmean=0, alternative='less')

    alpha = 0.05  # Set your desired significance level
    if p_value < alpha:
        print(f"Reject the null hypothesis. There is a significant positive effect of Dworkin on {metric} (p =", '{0:.3}'.format(p_value), ").")
    else:
        print(f"Fail to reject the null hypothesis. There is no significant positive effect of Dworkin on {metric} (p =", '{0:.3}'.format(p_value), ").")


for metric, df in metrics.items():
    print("*"*20)
    print(metric)
    print("*"*20)
    # Compute the differences between metric_for_X and metric_for_Y
    df['difference'] = df[f'{metric}(Dworkin)'] - df[f'{metric}'] 

    # Set the style for the plot (optional but can make the plot more visually appealing)
    sns.set(style="whitegrid")

    # Create a single figure with four subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    # Create a histogram for each method
    methods = ['nnunet', 'spm_cropped', 'samseg']
    for i, method in enumerate(methods):
        method_df = df[df['Method'] == method]
        sns.histplot(method_df['difference'], bins=15, kde=True, ax=axes[i])
        axes[i].set_xlabel(f'Differences in {metric} ({method})')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Histogram of Differences in {metric} ({method})')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(pjoin(directory, f'histogram_{metric}.png'), dpi=300, bbox_inches='tight')

    # Display the plot (optional)
    #plt.show()

    print("## Overall ##")
    ttests(df, metric)

    for method in methods:
        print(f"## {method} ##")
        pdf = df[df["Method"] == method]
        ttests(pdf, metric)

    print()
