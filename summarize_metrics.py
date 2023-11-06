import os
import pandas as pd
import numpy as np

# Base directory
base_dir = r'/dir/scratchL/mwynen/data/cusl_wml/predictions'

def summarize_metrics(base_dir, test=False):
    # List of models (subfolders in "predictions")
    models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Dictionary to store mean metrics for each model
    model_mean_metrics = {}
    suffix = "test" if test else "val"

    # Loop through each model to compute mean metrics
    for model in models:
        file_path = os.path.join(base_dir, model, f"metrics_{model}_{suffix}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_metrics = ['DSC', 'PQ', 'Fbeta', 'LTPR', 'PPV', 'Dice_Per_TP', 'DiC', 'CLR', 'Dice_Per_TP_CL', 
                    'Pred_Lesion_Count','Ref_Lesion_Count', 'CL_Count']
            for metric in all_metrics:
                if metric not in list(df.columns):
                    df[metric] = np.nan
            
            # Calculate mean for each metric
            mean_values = df[['DSC', 'PQ', 'Fbeta', 'LTPR', 'PPV', 'Dice_Per_TP', 'DiC', 'CLR', 'Dice_Per_TP_CL']].mean().to_dict()
            sum_values = df[['Pred_Lesion_Count','Ref_Lesion_Count', 'CL_Count']].sum().to_dict()
            
            mean_values.update(sum_values)

            # Store the result in the dictionary
            model_mean_metrics[model] = mean_values
        else:
            print(f"Warning: {file_path} does not exist!")

    # Convert the dictionary to a DataFrame and write it to CSV
    mean_df = pd.DataFrame.from_dict(model_mean_metrics, orient='index').round(3)
    mean_df.to_csv(os.path.join(base_dir, 'model_mean_metrics.csv'))

    print("Model mean metrics saved to 'model_mean_metrics.csv'")

import argparse

parser = argparse.ArgumentParser(description="Summarize segmentation metrics.")
parser.add_argument("--preds_dir", required=True, help ="directory where to find the different models' predictions")
parser.add_argument("--test", action="store_true", help="whether to use the _test suffix or not")

args = parser.parse_args()

summarize_metrics(args.preds_dir, args.test)

