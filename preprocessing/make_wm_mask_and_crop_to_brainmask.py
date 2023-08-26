# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 22:51:05 2023

@author: Admin
"""
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from os.path import join as pjoin
import argparse

# parser = argparse.ArgumentParser(description='Get all command line arguments.')
# parser.add_argument('--aseg_dir', type=str, required=False, default=None, help='Specify the path to the aseg segmentation masks directory (output of fressurfer)')
# parser.add_argument("--brainmsks", type=str, help="Directory containing the brainmask files.")
# parser.add_argument('--lesions_dir', type=str, required=True, default=None, help='Specify the path to the lesion masks directory')
# parser.add_argument('--output_wm_dir', type=str, required=False, default=None, help='Specify the path to the output lesion directory')


def find_bounding_box(mask_data):
    """
    Finds the bounding box of the brain mask.
    """
    indices = np.where(mask_data > 0)
    x_min, x_max = np.min(indices[0]), np.max(indices[0])
    y_min, y_max = np.min(indices[1]), np.max(indices[1])
    z_min, z_max = np.min(indices[2]), np.max(indices[2])
    return x_min, x_max, y_min, y_max, z_min, z_max

def extract_wm_mask_from_aseg(aseg_mask, lesion_mask):
    labels_to_keep = [2, 9, 10, 11, 12, 13, 17, 18, 19, 20, 26, 27, 28, 32, 41, 
                      48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 64]
    wm_mask = np.where(np.isin(aseg_mask, labels_to_keep), 1, 0)
    wm_mask[lesion_mask >= 1] = 1
    return wm_mask

def crop_to_brainmask(wm_mask, brainmask):
    x_min, x_max, y_min, y_max, z_min, z_max = find_bounding_box(brainmask)
    cropped_wm_mask_data = wm_mask[x_min:x_max, y_min:y_max, z_min:z_max]
    return cropped_wm_mask_data

def make_wm_mask_and_crop_to_brainmask(aseg_path, lesion_path, brainmask_path, output_path):
    # Load lesion mask and aseg mask
    lesion_img = nib.load(lesion_path)
    aseg_img = nib.load(aseg_path)
    brainmask_img = nib.load(brainmask_path)

    # Get the data arrays from the images
    lesion_data = lesion_img.get_fdata()
    aseg_data = aseg_img.get_fdata()
    brainmask_data = brainmask_img.get_fdata()
    
    wm_mask = extract_wm_mask_from_aseg(aseg_data, lesion_data)
    cropped_wm_mask_data = crop_to_brainmask(wm_mask, brainmask_data)
    
    output_nifti = nib.Nifti1Image(cropped_wm_mask_data, affine=aseg_img.affine)
    nib.save(output_nifti, output_path)
    
def compute_all(aseg_dir, lesions_dir, brainmasks_dir, output_wm_dir):
    os.makedirs(output_wm_dir, exist_ok=True)
    for aseg_path in os.listdir(aseg_dir):
        subject_id = aseg_path.split('_')[0]
        aseg_path = pjoin(aseg_dir, aseg_path)
        print(subject_id)
        
        lesion_filename = [x for x in os.listdir(lesions_dir) if x.startswith(subject_id)][0]
        lesion_path = pjoin(lesions_dir, lesion_filename)
        
        brainmask_filename = [x for x in os.listdir(brainmasks_dir) if x.startswith(subject_id)][0]
        brainmask_path = pjoin(brainmasks_dir, brainmask_filename)
        
        output_path = pjoin(output_wm_dir, f"{subject_id}_wm_cropped.nii.gz")
        
        make_wm_mask_and_crop_to_brainmask(aseg_path, lesion_path, brainmask_path, output_path)

if __name__=="__main__":
    lesions_dir = r"D:/R4/samseg"
    aseg_dir = r"D:\R4\aseg_t2starw"
    brainmasks_dir = r"D:\R4\brainmasks"
    output_wm_dir = r"D:\R4\wm_cropped"
    compute_all(aseg_dir, lesions_dir, brainmasks_dir, output_wm_dir)
        