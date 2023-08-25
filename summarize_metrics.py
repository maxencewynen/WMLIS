import os
import pandas as pd

# Base directory
base_dir = r'/dir/scratchL/mwynen/data/cusl_wml/predictions'

# List of models (subfolders in "predictions")
models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Dictionary to store mean metrics for each model
model_mean_metrics = {}

# Loop through each model to compute mean metrics
for model in models:
    file_path = os.path.join(base_dir, model, f"metrics_{model}_val.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Calculate mean for each metric
        mean_values = df[['PQ', 'Fbeta', 'LTPR', 'PPV', 'Dice_Per_TP', 'DiC']].mean().to_dict()
        sum_values = df[['Pred_Lesion_Count','Ref_Lesion_Count']].sum().to_dict()
        
        mean_values.update(sum_values)

        # Store the result in the dictionary
        model_mean_metrics[model] = mean_values
    else:
        print(f"Warning: {file_path} does not exist!")

# Convert the dictionary to a DataFrame and write it to CSV
mean_df = pd.DataFrame.from_dict(model_mean_metrics, orient='index').round(3)
mean_df.to_csv(os.path.join(base_dir, 'model_mean_metrics.csv'))
mean_df.to_csv(os.path.join(base_dir, 'model_mean_metrics.csv'))

print("Model mean metrics saved to 'model_mean_metrics.csv'")

