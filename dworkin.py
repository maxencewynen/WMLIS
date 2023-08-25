import os
import argparse
import numpy as np
import nibabel as nib
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import label

def compute_hessian_eigenvalues(image):
    hessian_matrices = hessian_matrix(image, sigma=1, use_gaussian_derivatives=False)
    eigs = hessian_matrix_eigvals(hessian_matrices)
    return eigs

def process_files(directory, threshold=0.35):
    files = [f for f in os.listdir(directory) if f.endswith('_pred_prob.nii.gz')]
    
    for f in files:
        print(f)
        full_path = os.path.join(directory, f)
        
        img = nib.load(full_path)
        image_data = img.get_fdata()
        
        mask = image_data > threshold
        masked_image_data = np.where(mask, image_data, 0)
        
        eigenvalues = compute_hessian_eigenvalues(masked_image_data)
        lesion_centers_mask = np.all(np.array(eigenvalues) < 0, axis=0)
        
        lesion_clusters = label(lesion_centers_mask)
        
        clustered_img = nib.Nifti1Image(lesion_clusters, img.affine)
        new_name = f.replace('_pred_prob.nii.gz', '_pred_instances.nii.gz')
        new_full_path = os.path.join(directory, new_name)
        
        nib.save(clustered_img, new_full_path)
        
        num_lesion_centers = len(np.unique(lesion_clusters)) - 1
        print(f"Processed {f}: Number of identified lesion centers: {num_lesion_centers}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process lesion probability maps to identify lesion centers.')
    parser.add_argument('--pred_path', type=str, help='Path to directory containing lesion probability map NIFTI files.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for lesion probability (default is 0.5)')
    
    args = parser.parse_args()
    process_files(args.pred_path, args.threshold)

