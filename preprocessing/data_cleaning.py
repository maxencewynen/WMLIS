# -*- coding: utf-8 -*-
"""
maxencewynen@gmail.com
"""
import os
from shutil import copy
import pandas as pd
from scipy import ndimage
import nibabel as nib
import numpy as np
import tqdm

# Load the masks for CTRL and FPRL
ctrl = nib.load("D:/R4/FullSegmentationsMask/sub-005/ses-01/sub-005_ses-01_mask-CTRL_T2starw.nii.gz").get_fdata()
fprl = nib.load("D:/R4/FullSegmentationsMask/sub-005/ses-01/sub-005_ses-01_mask-FPRL_T2starw.nii.gz").get_fdata()

# List of subject IDs
subjs = ['005', '008', '017', '022', '031', '032', '038', '051', '054', '055', '056',
         '057', '059', '060', '061', '062', '063', '065', '089', '106', '107', '110',
         '114', '125', '129', '132', '136', '152', '156', '160', '164', '169', '184',
         '189', '193', '198', '205', '206', '208', '209', '210', '217', '220', '224',
         '229', '230', '243']

# Dictionaries to store discarded lesions
CTRL_DISCARDED_LESIONS = {}
FPRL_DISCARDED_LESIONS = {}

# Set to store all subject IDs
all_subjs = set()
for x in os.listdir("D:\R4"):
    if "sub-" in x:
        all_subjs.add(x[4:7])
all_subjs = sorted(list(subjs))

# Constants for lesion classes
CLASS_PRL = 1
CLASS_CTRL = 2

# Directories
root_dir = r"D:\R4"
tgt_dir = r"D:\R4\labels"
data_cleaning_dir = r"D:\R4\data_cleaning"

# Excel files
excel_file = r"D:/R4/FullSegmentationsMask/Instant_Segmentation_LesionDatabase.xlsx"
updated_excel_file = os.path.join(tgt_dir, "lesion_database.xlsx")

# Read the original DataFrame from the Excel file
orig_df = pd.read_excel(excel_file)

# Filter the DataFrame for session 1
df = orig_df[["subject", "session", "Lesion_ID", "Confidence_Level_PRL",
              "Lesion_Mask", "CorticalLesion", "InfratentorialLesion"]]
df = df[df["session"] == 1]

def is_prl(subject, lesion_id, filename):
    """
    Check if a lesion is classified as PRL based on the DataFrame information.

    Args:
        subject (str): Subject ID
        lesion_id (int): Lesion ID
        filename (str): Filename of the mask

    Returns:
        bool: True if the lesion is classified as PRL, False otherwise
    """
    tdf = df[(df["subject"] == subject) & (df["Lesion_ID"] == lesion_id)]
    if len(tdf) > 1:
        if "FPRL" in filename:
            tdf = tdf[(tdf["Lesion_Mask"] == 2) | (tdf["Lesion_Mask"] == 3)]
        elif 'CTRL' in filename:
            tdf = tdf[(tdf["Lesion_Mask"] == 0)]
        else:
            raise Exception
    
    try:
        return int(tdf["Confidence_Level_PRL"]) >= 4
    except (TypeError, ValueError) as e:
        return False
    except Exception as e:
        print('-- New error message: ', e)
        raise e

def is_discarded_lesion(df, subject, lesion_id, filename):
    """
    Check if a lesion is discarded based on the DataFrame information.

    Args:
        df (DataFrame): DataFrame containing the lesion information
        subject (str): Subject ID
        lesion_id (int): Lesion ID
        filename (str): Filename of the mask

    Returns:
        bool: True if the lesion is discarded, False otherwise
    """
    tdf = df[(df["subject"] == subject) & (df["Lesion_ID"] == lesion_id)]
    if len(tdf) > 1:
        if "FPRL" in filename:
            tdf = tdf[(tdf["Lesion_Mask"] == 2) | (tdf["Lesion_Mask"] == 3)]
        elif 'CTRL' in filename:
            tdf = tdf[(tdf["Lesion_Mask"] == 0)]
        else:
            raise Exception
        
    discarded = int(tdf["CorticalLesion"].any()) or bool(tdf["InfratentorialLesion"].any())
    if discarded:
        print(f"Lesion {lesion_id}, {filename} has been discarded")
    return discarded

def check_voxel_connectivity(mask, fprl_id, subj, filename, affine):
    """
    Check the voxel connectivity for a specific lesion ID.

    Args:
        mask (ndarray): Mask data
        fprl_id (int): Lesion ID
        subj (str): Subject ID
        filename (str): Filename of the mask
        affine (ndarray): Affine matrix

    Returns:
        None
    """
    labeled_mask, num_labels = ndimage.label(mask == fprl_id)
    if num_labels > 1:
        indices = np.where(mask == fprl_id)
        print(f"Voxels with id {fprl_id} are not all connected. (#components = {num_labels}) ({filename})")
        nib.save(nib.Nifti1Image(labeled_mask, affine), 
                  os.path.join(data_cleaning_dir, f"{subj}_{filename}_lesion-{fprl_id}.nii.gz"))

def merge_and_copy(subject, df):
    """
    Merge the masks and copy the resulting masks and DataFrame.

    Args:
        subject (str): Subject ID
        df (DataFrame): DataFrame containing the lesion information

    Returns:
        DataFrame: Updated DataFrame
    """
    subj = f"sub-{subject}"
    print(subj, end="   ")
    sourcedir = os.path.join(root_dir, "FullSegmentationsMask", subj, "ses-01")
    target_dir = tgt_dir
    
    ctrl_file = os.path.join(sourcedir, f"{subj}_ses-01_mask-CTRL_T2starw.nii.gz")
    fprl_file = os.path.join(sourcedir, f"{subj}_ses-01_mask-FPRL_T2starw.nii.gz")
    
    fprl_nib = nib.load(fprl_file)
    fprl = fprl_nib.get_fdata()
    affinef = fprl_nib.affine
    final_affine = affinef
    
    t2starmag = os.path.join(r"D:/R4/images/" + f"{subj}_ses-01_part-mag_T2starw.nii.gz")
    affine_image = nib.load(t2starmag).affine
    
    if np.sum(np.abs(affine_image - affinef)) < 1e-5:
        print("[WARNING] Affine from image and fprl are too far from each other")
        final_affine = affine_image
    
    # Check if mask-CTRL exists
    if os.path.exists(ctrl_file):
        print("(has ctrl & fprl)   //   ", end="   ")
        print("SKIPPING FOR NOW")
        return df
        # Assert no conflict between the files
        # Just add 2000 to this one and 1000 to the other one
        # Update the lesion IDs in the target DataFrame
        # Create new merged mask in target_dir   
        
        ctrl_nib = nib.load(ctrl_file)
        ctrl = ctrl_nib.get_fdata()
        affinec = ctrl_nib.affine
        
        assert np.sum(np.abs(affinef - affinec)) < 1e-5, "affine from ctrl and fprl are too far from each other"
        
        if np.sum(np.abs(affine_image - affinec)) < 1e-5:
            print("[WARNING] Affine from image and ctrl are too far from each other")
            final_affine = affine_image
        
        c1 = ctrl[np.where(fprl != 0)]
        conflicting_labels_ctrl = (c1[c1!=0])
        f1 = fprl[np.where(ctrl != 0)]
        conflicting_labels_fprl = (f1[f1!=0])
        n_conflicting_voxels = len(conflicting_labels_ctrl)

        # If some voxels have labels both in the mask-CTRL and in the mask-FPRL
        if n_conflicting_voxels != 0:
            print(f"has {n_conflicting_voxels} conflicting voxels")
            clabels = np.unique(conflicting_labels_ctrl)
            flabels = np.unique(conflicting_labels_fprl)
            merged_instances = np.zeros_like(fprl)            
            merged_semantic = np.zeros_like(fprl)
            
            last_prl_id = 1000
            last_ctrl_id = 2000
            
            for fprl_id, cnt in zip(*np.unique(fprl, return_counts=True)):
                if fprl_id == 0 or is_discarded_lesion(df, subj, fprl_id, "FPRL"):
                    continue
                if cnt < 10: 
                    print(f"Lesion {fprl_id} in FPRL is {cnt} voxels")
                check_voxel_connectivity(fprl, fprl_id, subj, "FPRL", affinef)
    
                if is_prl(subj, fprl_id, fprl_file):
                    new_id = last_prl_id
                    last_prl_id += 1           
                    merged_semantic[fprl == fprl_id] = CLASS_PRL
                    
                else:
                    new_id = last_ctrl_id
                    last_ctrl_id += 1
                    
                    merged_semantic[fprl == fprl_id] = CLASS_CTRL
                condition = (df['subject'] == subj) & (df['Lesion_ID'] == fprl_id) & (df['Lesion_Mask'] >= 2)

                df.loc[condition, 'mwid'] = new_id
                merged_instances[fprl == fprl_id] = new_id
                
            for ctrl_id, cnt in zip(*np.unique(ctrl, return_counts=True)):
                if ctrl_id == 0 or is_discarded_lesion(df, subj, ctrl_id, "CTRL"):
                    continue
                if cnt < 10: 
                    print(f"Lesion {ctrl_id} in CTRL is {cnt} voxels")
                check_voxel_connectivity(ctrl, ctrl_id, subj, "CTRL", affinec)
    
                if is_prl(subj, ctrl_id, ctrl_file):
                    new_id = last_prl_id
                    last_prl_id += 1           
                    merged_semantic[ctrl == ctrl_id] = CLASS_PRL
                    
                else:
                    new_id = last_ctrl_id
                    last_ctrl_id += 1
                    
                    merged_semantic[ctrl == ctrl_id] = CLASS_CTRL
                condition = (df['subject'] == subj) & (df['Lesion_ID'] == ctrl_id) & (df['Lesion_Mask'] == 0)

                df.loc[condition, 'mwid'] = new_id
                merged_instances[ctrl == ctrl_id] = new_id
            
            # TODO: Choose a strategy for conflicting instances
            for ctrl_label in clabels:
                ctrl_is_prl = is_prl(subj, ctrl_label, ctrl_file)
                for fprl_label in flabels:
                    fprl_is_prl = is_prl(subj, fprl_label, fprl_file)
                    conflicts = np.where((ctrl == ctrl_label) & (fprl_label == fprl_label))
                    if not fprl_is_prl and not ctrl_is_prl:
                        print("none are prl")
                    elif fprl_is_prl and not ctrl_is_prl:
                        print("fprl is prl")
                        merged_semantic[conflicts] = CLASS_PRL
                    elif not ctrl_is_prl and ctrl_is_prl:
                        print("ctrl is prl")
                        merged_semantic[conflicts] = CLASS_PRL
                    else: # both are prl
                        print("both are prl")
                            
                    pass
            print()
            
        else: # No conflicts between the mask-CTRL and mask-FPRL files
            print("has 0 conflicting voxels")
            merged_instances = np.zeros_like(fprl)            
            merged_semantic = np.zeros_like(fprl)
            
            last_prl_id = 1000
            last_ctrl_id = 2000
            
            for fprl_id, cnt in zip(*np.unique(fprl, return_counts=True)):
                if fprl_id == 0 or is_discarded_lesion(df, subj, fprl_id, "FPRL"):
                    continue
                if cnt < 10: 
                    print(f"Lesion {fprl_id} in FPRL is {cnt} voxels")
                check_voxel_connectivity(fprl, fprl_id, subj, "FPRL", affinef)
    
                if is_prl(subj, fprl_id, fprl_file):
                    new_id = last_prl_id
                    last_prl_id += 1           
                    merged_semantic[fprl == fprl_id] = CLASS_PRL
                    
                else:
                    new_id = last_ctrl_id
                    last_ctrl_id += 1
                    
                    merged_semantic[fprl == fprl_id] = CLASS_CTRL
                condition = (df['subject'] == subj) & (df['Lesion_ID'] == fprl_id) & (df['Lesion_Mask'] >= 2)

                df.loc[condition, 'mwid'] = new_id
                merged_instances[fprl == fprl_id] = new_id
                
            for ctrl_id, cnt in zip(*np.unique(ctrl, return_counts=True)):
                if ctrl_id == 0 or is_discarded_lesion(df, subj, ctrl_id, "CTRL"):
                    continue
                if cnt < 10: 
                    print(f"Lesion {ctrl_id} in CTRL is {cnt} voxels")
                check_voxel_connectivity(ctrl, ctrl_id, subj, "CTRL", affinec)
    
                if is_prl(subj, ctrl_id, ctrl_file):
                    new_id = last_prl_id
                    last_prl_id += 1           
                    merged_semantic[ctrl == ctrl_id] = CLASS_PRL
                    
                else:
                    new_id = last_ctrl_id
                    last_ctrl_id += 1
                    
                    merged_semantic[ctrl == ctrl_id] = CLASS_CTRL
                condition = (df['subject'] == subj) & (df['Lesion_ID'] == ctrl_id) & (df['Lesion_Mask'] == 0)

                df.loc[condition, 'mwid'] = new_id
                merged_instances[ctrl == ctrl_id] = new_id
            print()
        
    else:
        print("only has a FPRL file")
        merged_instances = np.zeros_like(fprl)        
        fprl_nib = nib.load(fprl_file)
        fprl = fprl_nib.get_fdata()
        affinef = fprl_nib.affine
        
        merged_instances = np.zeros_like(fprl)            
        merged_semantic = np.zeros_like(fprl)
        
        last_prl_id = 1000
        last_ctrl_id = 2000
        
        for fprl_id, cnt in zip(*np.unique(fprl, return_counts=True)):
            if fprl_id == 0 or is_discarded_lesion(df, subj, fprl_id, "FPRL"):
                continue
            if cnt < 10: 
                print(f"Lesion {fprl_id} in FPRL is {cnt} voxels")
            check_voxel_connectivity(fprl, fprl_id, subj, "FPRL", affinef)

            if is_prl(subj, fprl_id, fprl_file):
                new_id = last_prl_id
                last_prl_id += 1           
                merged_semantic[fprl == fprl_id] = CLASS_PRL
                
            else:
                new_id = last_ctrl_id
                last_ctrl_id += 1
                
                merged_semantic[fprl == fprl_id] = CLASS_CTRL
            condition = (df['subject'] == subj) & (df['Lesion_ID'] == fprl_id) & (df['Lesion_Mask'] >= 2)

            df.loc[condition, 'mwid'] = new_id
            merged_instances[fprl == fprl_id] = new_id
    
    df.to_excel(updated_excel_file, index=False)
    nib.save(nib.Nifti1Image(merged_instances.astype('int16'), final_affine), 
              os.path.join(target_dir, f"{subj}_ses-01_mask-instances.nii.gz"))
    nib.save(nib.Nifti1Image(merged_semantic.astype('int16'), final_affine), 
              os.path.join(target_dir, f"{subj}_ses-01_mask-classes.nii.gz"))
    
    return df

new_df = orig_df.copy()
new_df["mwid"] = None
#new_df = pd.read_excel("D:/R4/labels/lesion_database.xlsx")

# Merge and copy for each subject
for sub in subjs:
    new_df = merge_and_copy(sub, new_df)
