import os
import numpy as np
import pandas as pd
import nibabel as nib
from metrics import *
from multiprocessing import Pool

# Define paths
data_dir = "/home/mwynen/data/cusl_wml"
gt_dir = os.path.join(data_dir, "labels")
method_dirs = ["gt_binarized", "spm_cropped", "spm_cropped+dworkin", "nnunet", "nnunet+dworkin", "samseg", "samseg+dworkin"]

def evaluate_method(method_name):
    print("*"*25)
    print("Evaluating", method_name)
    print("*"*25)
    
    metrics_dict = {"Subject_ID": [], "File": []}
    metrics_dict["PQ"] = []
    metrics_dict["Fbeta"] = []
    metrics_dict["LTPR"] = []
    metrics_dict["PPV"] = []
    metrics_dict["Dice_Per_TP"] = []
    metrics_dict["Pred_Lesion_Count"] = []
    metrics_dict["Ref_Lesion_Count"] = []
    metrics_dict["DiC"] = []
    metrics_dict["Dice"] = []
    metrics_dict["Method"] = method_name

    for gt_file in sorted(os.listdir(gt_dir)):
        if not "mask-instances" in gt_file:
            continue
        print(gt_file, f'({method_name})')

        subj_id = gt_file.split("_ses")[0].split("sub-")[-1]  # Extracting subject ID
        pred_file = "sub-" + subj_id + "_ses-01_pred-instances.nii.gz"
        pred_file_path = os.path.join(data_dir, method_name, pred_file)

        if not os.path.exists(pred_file_path):
            print(f"No prediction found for {pred_file}")
            continue

        gt_img = nib.load(os.path.join(gt_dir, gt_file)).get_fdata()
        pred_img = nib.load(pred_file_path).get_fdata()

        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred_img, gt_img)

        metrics_dict["Subject_ID"].append(subj_id)
        metrics_dict["File"].append(pred_file)
        
        pq_val = panoptic_quality(pred = pred_img, ref=gt_img, \
                matched_pairs=matched_pairs, unmatched_pred=unmatched_pred, unmatched_ref=unmatched_ref)
        metrics_dict["PQ"].append(pq_val)

        fbeta_val = f_beta_score(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred, unmatched_ref=unmatched_ref)
        metrics_dict["Fbeta"].append(fbeta_val)

        ltpr_val = ltpr(matched_pairs=matched_pairs, unmatched_ref=unmatched_ref)
        metrics_dict["LTPR"].append(ltpr_val)

        ppv_val = ppv(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred)
        metrics_dict["PPV"].append(ppv_val)
        
        dice_scores = dice_per_tp(pred_img, gt_img, matched_pairs)
        # Assuming you want the average Dice score per subject
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
        metrics_dict["Dice_Per_TP"].append(avg_dice)
        
        metrics_dict["Pred_Lesion_Count"].append(pred_lesion_count(pred_img))
        metrics_dict["Ref_Lesion_Count"].append(ref_lesion_count(gt_img))
        metrics_dict["DiC"].append(DiC(pred_img, gt_img))

        pred_bin_img = np.copy(pred_img)
        pred_bin_img[pred_bin_img > 0] = 1
        gt_bin_img = np.copy(gt_img)
        gt_bin_img[gt_bin_img > 0] = 1

        metrics_dict["Dice"].append(dice_metric(gt_bin_img, pred_bin_img))
    
    # Convert dictionary to dataframe and return it
    df = pd.DataFrame(metrics_dict)
    return df

if __name__ == "__main__":
    all_metrics = []
    
    # Create a pool of processes to parallelize method evaluation
    with Pool(processes=len(method_dirs)) as pool:
        all_metrics = pool.map(evaluate_method, method_dirs)
    
    # Concatenate the results
    all_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # Reorder columns to have 'Method' as the first column
    all_metrics = all_metrics[['Method'] + [col for col in all_metrics.columns if col != 'Method']]
    
    # Save to CSV files
    all_metrics.to_csv("metrics_comparison_all_subjects.csv", index=False)
    method_means = all_metrics.groupby('Method').mean()
    method_means.to_csv("metrics_comparison.csv")

