import nibabel as nib
import numpy as np
import glob

def dice_score(prediction, ground_truth):
    intersection = np.sum(prediction[ground_truth==1])
    dice = (2. * intersection) / (np.sum(prediction) + np.sum(ground_truth))
    return dice

def dice_norm_metric(ground_truth, predictions):
    """
    Compute Optimized Normalised Dice Coefficient (nDSC)
    for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Optimized normalised dice coefficient (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")

    # Pre-calculate these terms
    seg_and_gt = seg * gt
    seg_and_not_gt = seg * (1 - gt)
    not_seg_and_gt = (1 - seg) * gt
    
    # Calculations
    tp = np.sum(seg_and_gt)
    fp = np.sum(seg_and_not_gt)
    fn = np.sum(not_seg_and_gt)
    
    # Normalize constant
    gt_sum = np.sum(gt)
    k = (1 - r) * gt_sum / (r * (gt.size - gt_sum)) if gt_sum != 0 else 1.0
    
    # Normalized Dice Score
    dsc_norm = 2. * tp / (k * fp + 2. * tp + fn) if (k * fp + 2. * tp + fn) != 0 else 1.0
    return dsc_norm


# Glob helps to find all the pathnames matching a specified pattern
nii_files = glob.glob(r'/home/mwynen/data/bxl/test/labels/*mask-classes.nii.gz')

dice_scores = []
dice_norm_scores = []



for file in nii_files:
    # Load nifti file
    img = nib.load(file)

    # Get data from nifti file
    data = img.get_fdata()

    # Create a prediction that all voxels that are PRL are classified as PRL
    prediction = np.copy(data)
    prediction[data==2] = 1
    
    # Replace all values of 2 (CTRL) with 0 (background)
    data[data==2] = 0

    # Compute dice score and append to list
    dice_scores.append(dice_score(prediction, data))
    dice_norm_scores.append(dice_norm_metric(data, prediction))

# Compute average Dice score
average_dice_score = np.mean(dice_scores)
average_dice_norm_score = np.mean(dice_norm_scores)

print(f"The average normalized dice score is {average_dice_norm_score}")


print(f"The average dice score is {average_dice_score}")
