import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from postprocess import *

def process_files(directory, threshold, l_min):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('prob.nii.gz')]
    
    # Loop through each file and process it
    for f in sorted(files):
        print(f)
        full_path = os.path.join(directory, f)
        
        # Load the NIFTI file
        img = nib.load(full_path)
        voxel_size = img.header.get_zooms()
        data = img.get_fdata()
        
        # Threshold the image
        binary_data = np.where(data >= threshold, 1, 0).astype(np.uint8)
        
        # Remove objects smaller than 9 voxels
        binary_data = remove_small_lesions_from_binary_segmentation(binary_data, voxel_size=voxel_size, l_min=l_min)

        # Save the binary segmentation
        new_img = nib.Nifti1Image(binary_data, img.affine)
        new_name = f.replace(f[-16:], 'seg-binary.nii.gz')
        new_full_path = os.path.join(directory, new_name)
        nib.save(new_img, new_full_path)
        
        # Find connected components larger than 9 voxels
        labeled_array, num_features = label(binary_data)
        
        # Save the instances
        new_img = nib.Nifti1Image(labeled_array.astype(np.int32), img.affine)
        new_name = f.replace(f[-16:], 'pred-instances.nii.gz')
        new_full_path = os.path.join(directory, new_name)
        nib.save(new_img, new_full_path)

if __name__ == "__main__":
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Process lesion probability maps.', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add arguments
    parser.add_argument('--pred_path', type=str, help='Path to directory including lesion probability map nifti files')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for binarization')
    parser.add_argument('--minimum_lesion_size', type=int, default=14, help='Minimum lesion size in mm^3')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function
    process_files(args.pred_path, args.threshold, args.minimum_lesion_size)

