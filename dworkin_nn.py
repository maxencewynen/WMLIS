import os
import argparse
import numpy as np
import nibabel as nib
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import label
from scipy.spatial import distance_matrix
from multiprocessing import Pool
from scipy.spatial.distance import cdist

def compute_hessian_eigenvalues(image):
    hessian_matrices = hessian_matrix(image, sigma=1, use_gaussian_derivatives=False)
    eigs = hessian_matrix_eigvals(hessian_matrices)
    return eigs

def find_nearest_lesion_labels(unlabelled_voxels_indices, lesion_clusters):
    labelled_voxels_indices = np.transpose(np.array(np.where(lesion_clusters > 0)))

    # Initialize an array to store the nearest labels
    nearest_labels = np.zeros(len(unlabelled_voxels_indices), dtype=lesion_clusters.dtype)

    # Loop through unlabelled voxels and compute distances incrementally
    for i, voxel_index in enumerate(unlabelled_voxels_indices):
        distances = cdist([voxel_index], labelled_voxels_indices)
        nearest_index = np.argmin(distances)
        nearest_labels[i] = lesion_clusters[tuple(labelled_voxels_indices[nearest_index])]

    # Assign the nearest lesion labels to unlabelled voxels
    lesion_clusters[tuple(unlabelled_voxels_indices.T)] = nearest_labels

def process_file(file_path, directory, threshold=0.35):
    print(file_path)
    img = nib.load(file_path)
    image_data = img.get_fdata()

    binary_image_data = np.where(image_data > threshold, 1, 0)
    binary_seg_filename = file_path.replace('_pred-prob.nii.gz', '_seg-binary.nii.gz')
    nib.save(nib.Nifti1Image(binary_image_data.astype(np.uint8), img.affine), binary_seg_filename)

    mask = image_data > threshold
    masked_image_data = np.where(mask, image_data, 0)

    eigenvalues = compute_hessian_eigenvalues(masked_image_data)
    lesion_centers_mask = np.all(eigenvalues < 0, axis=0)

    lesion_clusters = label(lesion_centers_mask)

    # Identify unlabelled voxels in binary image and assign nearest lesion labels
    unlabelled_voxels = np.logical_and(binary_image_data == 1, lesion_clusters == 0)
    unlabelled_voxels_indices = np.transpose(np.where(unlabelled_voxels))
    
    if len(unlabelled_voxels_indices) > 0:
        find_nearest_lesion_labels(unlabelled_voxels_indices, lesion_clusters)

    clustered_img = nib.Nifti1Image(lesion_clusters, img.affine)
    new_name = file_path.replace('_pred-prob.nii.gz', '_pred-instances.nii.gz')
    new_full_path = os.path.join(directory, new_name)

    nib.save(clustered_img, new_full_path)

    num_lesion_centers = len(np.unique(lesion_clusters)) - 1
    print(f"Processed {file_path}: Number of identified lesion centers: {num_lesion_centers}")

def process_files(directory, threshold=0.35):
    files = [f for f in os.listdir(directory) if f.endswith('_pred-prob.nii.gz')]
    file_paths = [os.path.join(directory, f) for f in sorted(files)]

    with Pool(processes=8) as pool:
        pool.starmap(process_file, [(file_path, directory, threshold) for file_path in file_paths])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process lesion probability maps to identify lesion centers.')
    parser.add_argument('--pred_path', type=str, help='Path to directory containing lesion probability map NIFTI files.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for lesion probability (default is 0.5)')

    args = parser.parse_args()
    process_files(args.pred_path, args.threshold)

