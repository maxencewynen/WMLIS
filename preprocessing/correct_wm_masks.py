import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from os.path import join as pjoin
import argparse

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# Example usage:
#lesion_dir = r"D:/R4/samseg"
#aseg_dir = r"D:\R4\aseg_t2starw"
#output_dir = r"D:\R4\samseg_dilated"
parser.add_argument('--labels_dir', type=str, required=False, default=None, help='Specify the path to the gt lesion masks directory (for stats on how many voxels are discarded)')
parser.add_argument('--lesions_dir', type=str, required=True, default=None, help='Specify the path to the lesion masks directory')
parser.add_argument('--aseg_dir', type=str, required=False, default=None, help='Specify the path to the aseg segmentation masks directory (output of fressurfer)')
parser.add_argument('--output_lesion_dir', type=str, required=False, default=None, help='Specify the path to the output lesion directory')
parser.add_argument('--output_wm_dir', type=str, required=False, default=None, help='Specify the path to the output lesion directory')
parser.add_argument('--ndilations_wm', type=int, required=False, default=2, help="Number of iterations for the dilation of the binary lesion mask during the computation of the wm mask")
parser.add_argument('--ndilations_lesions', type=int, required=False, default=3, help="Number of iterations for the dilation of the binary lesion mask during the computation of the lesion mask")

def process_subject_wm(lesion_path, aseg_path, output_dir, iterations=2, labels_dir=None):
    # Load lesion mask and aseg mask
    lesion_img = nib.load(lesion_path)
    aseg_img = nib.load(aseg_path)

    # Get the data arrays from the images
    lesion_data = lesion_img.get_fdata()
    aseg_data = aseg_img.get_fdata()

    # Set all voxels whose label is in 'labels_to_keep' to 1 and the rest to 0 in the aseg mask
    labels_to_keep = [2, 9, 10, 11, 12, 13, 17, 18, 19, 20, 26, 27, 28, 32, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 64]
    aseg_data = np.where(np.isin(aseg_data, labels_to_keep), 1, 0)

    # Set all voxels whose labels are set to 1 in the lesion mask to 1 as well in the output mask
    aseg_data[lesion_data == 1] = 1
    
    # Dilate the aseg_data once
    if iterations > 0:
        aseg_data = binary_dilation(aseg_data, iterations=iterations)

    # Create a new NIfTI image with the processed data
    output_img = nib.Nifti1Image(aseg_data, aseg_img.affine, aseg_img.header)

    # Save the output image to the specified output directory
    output_filename = os.path.basename(aseg_path).replace("aseg", "wm")
    output_file_path = os.path.join(output_dir, output_filename)
    nib.save(output_img, output_file_path)
    
    if labels_dir:
        sub = output_filename[:7]
        gt_mask = nib.load(pjoin(labels_dir, f"{sub}_ses-01_mask-classes.nii.gz")).get_fdata()
        ids, sums = np.unique(gt_mask, return_counts=True)
        t_prl = np.logical_and(gt_mask == 1, aseg_data == 0)
        t_ctrl = np.logical_and(gt_mask == 2, aseg_data == 0)
        unique_voxels_count_prl = np.sum(t_prl)
        unique_voxels_count_ctrl = np.sum(t_ctrl)
        
        if unique_voxels_count_prl + unique_voxels_count_ctrl > 0:
            print(sub, " has ", unique_voxels_count_prl, "/", sums[1], " PRL voxels (", round((unique_voxels_count_prl/sums[1])*100, 2),"%) and ",
                  unique_voxels_count_ctrl, "/", sums[2], " CTRL voxels (", round((unique_voxels_count_prl/sums[2])*100, 2),"%) out of ", sums[1] + sums[2], " total voxels that were excluded by the wm mask")
            # output_filename = os.path.basename(aseg_path).replace("aseg", "excluded_lesions")
            # output_file_path = os.path.join(output_dir, output_filename)
            # output_img = nib.Nifti1Image(t, aseg_img.affine, aseg_img.header)
            # nib.save(output_img, output_file_path)
    
        return sums[1], unique_voxels_count_prl, sums[2], unique_voxels_count_ctrl
    else: 
        return 0,0,0,0

def process_subject_lesion(lesion_path, output_dir, iterations=0, labels_dir=None):
    # Load lesion mask
    lesion_img = nib.load(lesion_path)
    
    # Get the data arrays from the image
    lesion_data = lesion_img.get_fdata()
    
    # Dilate the aseg_data once
    if iterations > 0:
        lesion_data = binary_dilation(lesion_data, iterations=iterations)
    
    output_filename = os.path.basename(lesion_path)
    output_file_path = os.path.join(output_dir, output_filename)
    
    # Create a new NIfTI image with the processed data
    output_img = nib.Nifti1Image(lesion_data, lesion_img.affine, lesion_img.header)
    
    if iterations > 0:    
        nib.save(output_img, output_file_path)
    
    if labels_dir:
        sub = output_filename[:7]
        gt_mask = nib.load(pjoin(labels_dir, f"{sub}_ses-01_mask-classes.nii.gz")).get_fdata()
        ids, sums = np.unique(gt_mask, return_counts=True)
        t_prl = np.logical_and(gt_mask == 1, lesion_data == 0)
        t_ctrl = np.logical_and(gt_mask == 2, lesion_data == 0)
        unique_voxels_count_prl = np.sum(t_prl)
        unique_voxels_count_ctrl = np.sum(t_ctrl)
    
        # if unique_voxels_count_prl + unique_voxels_count_ctrl > 0:
        print(sub, " has ", unique_voxels_count_prl, "/", sums[1], " PRL voxels (", round((unique_voxels_count_prl/sums[1])*100, 2),"%) and ",
              unique_voxels_count_ctrl, "/", sums[2], " CTRL voxels (", round((unique_voxels_count_prl/sums[2])*100, 2),"%) out of ", sums[1] + sums[2], 
              " total voxels that were excluded by the lesion mask")
        # output_filename = os.path.basename(aseg_path).replace("aseg", "excluded_lesions")
        # output_file_path = os.path.join(output_dir, output_filename)
        # output_img = nib.Nifti1Image(t, aseg_img.affine, aseg_img.header)
        # nib.save(output_img, output_file_path)
    
        return sums[1], unique_voxels_count_prl, sums[2], unique_voxels_count_ctrl
    else:
        return 0,0,0,0

def process_all_subjects(list_of_lesion_binary_masks, list_of_aseg_labeled_masks=None, 
                         output_dir='.', wm=False, lesion=False, iterations=0, labels_dir=None):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    total_prl = 0
    total_prl_excluded = 0
    total_ctrl = 0
    total_ctrl_excluded = 0
    if wm and lesion:
        print("Warning: not sure that this function works when setting wm and lesion to True")
    
    if wm: 
        # Process each subject
        for lesion_file, aseg_file in zip(list_of_lesion_binary_masks, list_of_aseg_labeled_masks):
            print(aseg_file)
            a = process_subject_wm(lesion_file, aseg_file, output_dir, iterations=iterations, labels_dir=labels_dir)
        
            total_prl += a[0] 
            total_prl_excluded += a[1] 
            total_ctrl += a[2] 
            total_ctrl_excluded += a[3]
    print("\n\n", "*"*25, "\n\n")
    if lesion:
        for lesion_file in list_of_lesion_binary_masks:        
            print(lesion_file)
            a = process_subject_lesion(lesion_file, output_dir, iterations=iterations, labels_dir=labels_dir)
    
            total_prl += a[0] 
            total_prl_excluded += a[1] 
            total_ctrl += a[2] 
            total_ctrl_excluded += a[3]
    
    if labels_dir:
        print(total_prl_excluded, total_prl)
        print(total_ctrl_excluded, total_ctrl)
        print()
        print("PRL: ", total_prl_excluded, "/", total_prl, "(", round(total_prl_excluded/total_prl,2) * 100, "%)")
        print("CTRL: ", total_ctrl_excluded, "/", total_ctrl, "(", round(total_ctrl_excluded/total_ctrl,2) * 100, "%)")


if __name__ == "__main__":
    args = parser.parse_args()
    
    lesion_files = sorted([pjoin(args.lesions_dir, f) for f in os.listdir(args.lesions_dir) if f.endswith('.nii.gz')])
    if args.aseg_dir:
        aseg_files = sorted([pjoin(args.aseg_dir, f) for f in os.listdir(args.aseg_dir) if f.endswith('.nii.gz')])
        print(lesion_files, "\n\n", aseg_files)
        process_all_subjects(lesion_files, aseg_files, output_dir=args.output_wm_dir, wm=True, lesion=False, iterations=args.ndilations_wm, labels_dir=args.labels_dir)
    if args.lesions_dir:
        process_all_subjects(lesion_files, output_dir=args.output_lesion_dir, wm=False, lesion=True, iterations=args.ndilations_lesions, labels_dir=args.labels_dir)

