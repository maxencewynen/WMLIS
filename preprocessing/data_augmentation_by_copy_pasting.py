import os
import nibabel as nib
import numpy as np
import random
from skimage.filters import gaussian
from os.path import join as pjoin
import argparse
from scipy.ndimage import binary_dilation, gaussian_filter, binary_erosion
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def data_augmentation_copy_paste(path_to_lesion_instance_segmentation, 
                                 path_to_wm_mask,
                                 paths_to_modalities, 
                                 factor, 
                                 output_dir_labels,
                                 output_dir_modalities, 
                                 max_tries=50, 
                                 sigma=1,
                                 dilation_factor=0,
                                 lesion_db=None,
                                 subject=None):
    
    # Load the lesion instance segmentation, wm mask, and modalities
    lesion_segmentation_img = nib.load(path_to_lesion_instance_segmentation)
    wm_mask_img = nib.load(path_to_wm_mask)
    modalities_imgs = [nib.load(modality_path) for modality_path in paths_to_modalities]

    # Get the data arrays from the loaded images
    lesion_segmentation = lesion_segmentation_img.get_fdata()
    wm_mask = wm_mask_img.get_fdata()
    modalities_data = [modality_img.get_fdata() for modality_img in modalities_imgs]

    # Initialize output arrays
    output_segmentation = np.copy(lesion_segmentation)
    output_modalities = [np.copy(modality_data) for modality_data in modalities_data]

    # Identify PRL lesions
    prl_mask = (lesion_segmentation >= 1000) & (lesion_segmentation < 2000)
    prl_instances = np.unique(lesion_segmentation[prl_mask])
    
    # Function to check if the location is inside the white matter mask
    def is_inside_wm_mask(x, y, z):
        try:
            return wm_mask[x, y, z] == 1
        except IndexError:
            return False

    # Function to check if the location overlaps with any existing lesion
    def is_overlap(x, y, z, mask):
        return mask[x, y, z] != 0

    def get_random_rotation():
        axis = random.choice([0, 1, 2])  # Choose a random axis: 0 for x, 1 for y, and 2 for z
        angle = random.randint(0, 3) * 90  # Random rotation in multiples of 90 degrees
        return axis, angle

    def rotate_3d_array(array, axis, angle):
        for _ in range(angle // 90):
            if axis == 0:
                array = np.rot90(array, axes=(1, 2))
            elif axis == 1:
                array = np.rot90(array, axes=(0, 2))
            elif axis == 2:
                array = np.rot90(array, axes=(0, 1))
        return array
    
    def find_extended_borders(binary_mask, iterations=1):
        # create an erosion of the binary mask
        eroded_mask = binary_erosion(binary_mask, iterations=iterations)
    
        # the border is the difference between the mask and its erosion
        border = binary_mask.astype(bool) ^ eroded_mask
        
        return border
    
    def smooth_region_borders(image, dilated_mask, dest_to_src_correspondance, rotated_image, sigma=1.0):
        # find the borders
        border = find_extended_borders(dilated_mask)
    
        # create a copy of the image
        image_copy = image.copy()
        
        # apply the gaussian filter to the whole image
        smooth_image = gaussian_filter(image_copy, sigma=sigma)
        
        # create a mask where all pixels are True
        mask_all = np.ones(image.shape, dtype=bool)
    
        # set only the border pixels in the mask to False
        mask_all[border == 1] = False
    
        # remove the original border in the image
        image_without_border = image_copy * mask_all
    
        # add the smoothed border to the image, effectively replacing the original border
        smoothed_image = image_without_border + smooth_image * border
        
        # Make sure the region inside the borders stays exactly the same as in the original image
        for (x_dst, y_dst, z_dst), (x_src, y_src, z_src) in dest_to_src_correspondance.items():
            smoothed_image[x_dst,y_dst,z_dst] = rotated_image[x_src,y_src,z_src]
    
        return smoothed_image

    # processed_lesions = {}    
    new_label = max(prl_instances) + 1
    
    if lesion_db is not None:
        high_confidence_level_prls = lesion_db[(lesion_db["subject"] == "sub-005") & 
                                               (lesion_db["Confidence_Level_PRL"] == 5)]
        high_confidence_level_prls = list(high_confidence_level_prls.dropna(subset=["mwid"])["mwid"])
    else:
        high_confidence_level_prls = None
        

    for copy_num in range(factor):
        for prl_instance in prl_instances:
            if high_confidence_level_prls is not None:
                if not prl_instance in high_confidence_level_prls:
                    print(f"XXX Skipping PRL {prl_instance} (Confidence level too low) XXX")
                    continue
            
            print(f"PRL {prl_instance} copy number {copy_num+1}")
            # Randomly choose a location inside the white matter mask
            x_random, y_random, z_random = random.choice(list(zip(*np.where(wm_mask == 1))))
            # Check if the random location does not overlap with any existing lesion
            while (not is_inside_wm_mask(x_random, y_random, z_random)) or is_overlap(x_random, y_random, z_random, output_segmentation):
                x_random, y_random, z_random = random.choice(list(zip(*np.where(wm_mask == 1))))

            is_new_location_ok=False
            tries = 0
            
            while (not is_new_location_ok) and tries < max_tries:
                tries += 1
                print(f"Trying new location for PRL {prl_instance} (Try {tries})", end=" ... ")
                x_random, y_random, z_random = random.choice(list(zip(*np.where(wm_mask == 1))))
                
                is_new_location_ok = True
                
                # Find the voxels corresponding to the current PRL instance
                prl_mask_current = (lesion_segmentation == prl_instance)
                prl_mask_current_dilated = prl_mask_current
                
                if dilation_factor > 0:
                    prl_mask_current_dilated = binary_dilation(prl_mask_current, iterations=dilation_factor)
                
                # Randomly rotate the prl_mask_current by multiples of 90 degrees along a random axis
                axis, rotation_angle = get_random_rotation()
                
                print(f"rotating by {rotation_angle} degrees along the axis {axis}")
                prl_mask_current = rotate_3d_array(prl_mask_current, axis, rotation_angle)
                prl_mask_current_dilated = rotate_3d_array(prl_mask_current_dilated, axis, rotation_angle)
                
                # Apply the same rotation to the modalities
                rotated_modalities_data = [rotate_3d_array(modality_data, axis, rotation_angle) for modality_data in modalities_data]

                # Find the indices of the (dilated) lesion in the mask
                x_prl_dil, y_prl_dil, z_prl_dil = np.where(prl_mask_current_dilated)
                
                # Create a mask for the newly pasted lesion copy
                prl_mask_copy = np.zeros_like(lesion_segmentation)
                for x_src, y_src, z_src in zip(x_prl_dil, y_prl_dil, z_prl_dil):
                    x_dst, y_dst, z_dst = x_random + (x_src - x_prl_dil[0]), y_random + (y_src - y_prl_dil[0]), z_random + (z_src - z_prl_dil[0])
                                        
                    if (not is_inside_wm_mask(x_dst, y_dst, z_dst)) or \
                        is_overlap(x_dst, y_dst, z_dst, output_segmentation) or \
                            is_overlap(x_dst, y_dst, z_dst, prl_mask_copy):
                        is_new_location_ok = False
                        break
                
                if not is_new_location_ok: continue
                new_label += 1
                print(f"New lesion: label {new_label}")
                
                dest_to_src_correspondance = {}
                
                for x_src, y_src, z_src in zip(x_prl_dil, y_prl_dil, z_prl_dil):
                    x_dst, y_dst, z_dst = x_random + (x_src - x_prl_dil[0]), y_random + (y_src - y_prl_dil[0]), z_random + (z_src - z_prl_dil[0])
                    
                    if 0 <= x_dst < output_segmentation.shape[0] and 0 <= y_dst < output_segmentation.shape[1] and 0 <= z_dst < output_segmentation.shape[2]:
                        if prl_mask_current[x_src, y_src, z_src] == 1:
                            output_segmentation[x_dst, y_dst, z_dst] = new_label
                            dest_to_src_correspondance[(x_dst, y_dst, z_dst)] = (x_src, y_src, z_src)
                            
                        prl_mask_copy[x_dst, y_dst, z_dst] = 1
                        
                        for modality_idx in range(len(rotated_modalities_data)):
                            if 0 <= x_dst < output_modalities[modality_idx].shape[0] and 0 <= y_dst < output_modalities[modality_idx].shape[1] and 0 <= z_dst < output_modalities[modality_idx].shape[2]:
                                output_modalities[modality_idx][x_dst, y_dst, z_dst] = rotated_modalities_data[modality_idx][x_src, y_src, z_src]
                
                # If sigma > 0, smoothe only the lesions borders 
                if sigma > 0:
                    for modality_idx in range(len(rotated_modalities_data)):
                        output_modalities[modality_idx] = smooth_region_borders(output_modalities[modality_idx], 
                                                                                np.copy(prl_mask_copy), dest_to_src_correspondance, 
                                                                                rotated_modalities_data[modality_idx], sigma=sigma)

    os.makedirs(output_dir_labels, exist_ok=True)
    os.makedirs(output_dir_modalities, exist_ok=True)
    
    # Save the resulting instance segmentation and classes
    output_segmentation_img = nib.Nifti1Image(output_segmentation, affine=lesion_segmentation_img.affine)
    nib.save(output_segmentation_img, pjoin(output_dir_labels, os.path.basename(path_to_lesion_instance_segmentation)))
    
    mask_classes = np.zeros_like(output_segmentation)
    mask_classes[output_segmentation >= 2000] = 2
    mask_classes[(output_segmentation >= 1000) & (output_segmentation < 2000)] = 1
    mask_classes_img = nib.Nifti1Image(mask_classes, affine=lesion_segmentation_img.affine)
    nib.save(mask_classes_img, pjoin(output_dir_labels, os.path.basename(path_to_lesion_instance_segmentation.replace("instances", "classes"))))

    # for modality_idx, output_path_modality in enumerate(output_paths_modalities):
    for modality_idx, modality_path in enumerate(paths_to_modalities):
        output_modalities_img = nib.Nifti1Image(output_modalities[modality_idx], affine=modalities_imgs[modality_idx].affine)
        nib.save(output_modalities_img, pjoin(output_dir_modalities, os.path.basename(modality_path)))


def augment_all_dataset(input_dir, output_dir, factor=1, sequences=('FLAIR', 'MPRAGE', 'phase'), 
                        sigma=1, dilation_factor=2, lesion_db=None):
    input_images_dir = pjoin(input_dir, "images")
    input_labels_dir = pjoin(input_dir, "labels")
    output_images_dir = pjoin(output_dir, "images")
    output_labels_dir = pjoin(output_dir, "labels")
    wm_dir = pjoin(input_dir, "wm_cropped")
    
    
    if lesion_db is not None:
        lesion_db = pd.read_excel(lesion_db)
    
    for maskname in sorted(os.listdir(input_labels_dir)):
        if "classes" in maskname: continue
        subject = maskname.split('_')[0]
        print("\n"*2,"*"*15, f"{subject}","*"*15, "\n"*2)
        paths_to_modalities = [pjoin(input_images_dir, x) for x in os.listdir(input_images_dir) \
                              if (any(s in x for s in sequences)) and subject in x]
        path_to_wm_mask = pjoin(wm_dir, f"{subject}_wm_cropped.nii.gz")
        path_to_lesion_instance_segmentation = pjoin(input_labels_dir, maskname)
        
        data_augmentation_copy_paste(path_to_lesion_instance_segmentation, 
                                     path_to_wm_mask,
                                     paths_to_modalities, 
                                     factor, 
                                     output_labels_dir, 
                                     output_images_dir,
                                     sigma=sigma,
                                     dilation_factor=dilation_factor,
                                     lesion_db=lesion_db,
                                     subject=subject)

def worker(maskname, input_images_dir, sequences, subject, wm_dir, input_labels_dir, 
           factor, output_labels_dir, output_images_dir, sigma, dilation_factor,
           lesion_db):
    
    print("\n"*2,"*"*15, f"{subject}","*"*15, "\n"*2)
    paths_to_modalities = [pjoin(input_images_dir, x) for x in os.listdir(input_images_dir) \
                          if (any(s in x for s in sequences)) and subject in x]
    path_to_wm_mask = pjoin(wm_dir, f"{subject}_wm_cropped.nii.gz")
    path_to_lesion_instance_segmentation = pjoin(input_labels_dir, maskname)
    
    data_augmentation_copy_paste(path_to_lesion_instance_segmentation, 
                                 path_to_wm_mask,
                                 paths_to_modalities, 
                                 factor, 
                                 output_labels_dir, 
                                 output_images_dir,
                                 sigma=sigma,
                                 dilation_factor=dilation_factor,
                                 lesion_db=lesion_db,
                                 subject=subject)

def augment_all_dataset_parallel(input_dir, output_dir, factor=1, sequences=('FLAIR', 'MPRAGE', 'phase'), 
                                 sigma=1, dilation_factor=2, lesion_db=None):
    input_images_dir = pjoin(input_dir, "images")
    input_labels_dir = pjoin(input_dir, "labels")
    output_images_dir = pjoin(output_dir, "images")
    output_labels_dir = pjoin(output_dir, "labels")
    wm_dir = pjoin(input_dir, "wm_cropped")
    if lesion_db is not None:
        lesion_db = pd.read_excel(lesion_db)
    
    with ThreadPoolExecutor() as executor:
        for maskname in sorted(os.listdir(input_labels_dir)):
            if "classes" in maskname: continue
            subject = maskname.split('_')[0]
            
            executor.submit(worker, maskname, input_images_dir, sequences, subject, 
                            wm_dir, input_labels_dir, factor, output_labels_dir, 
                            output_images_dir, sigma, dilation_factor, lesion_db)


    
if __name__=="__main__":
    pass
    parser = argparse.ArgumentParser(description="Apply binary masks to images from the same subject.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, help="Directory containing images/ and labels/.")
    parser.add_argument("--output_dir", type=str, help="Output directory where augmented images and masks will be saved.")
    parser.add_argument("--factor", type=int, default=1, help="How many copies per lesion.")
    parser.add_argument("--sigma", type=int, default=1, help="How many copies per lesion.")
    parser.add_argument("--dilation_factor", type=int, default=2, help="How many iterations for the dilation of the lesion.")
    parser.add_argument("--sequences", nargs='+', type=str, default=('FLAIR', 'MPRAGE', 'phase'), help="Sequences to augment.")
    parser.add_argument("--lesion_db", type=str, default=None, help="File to the lesion database including visual confidence levels.")
    parser.add_argument("--parallel", action="store_true", default=False)
    args = parser.parse_args()
    if not args.parallel:
        augment_all_dataset(args.input_dir, args.output_dir, args.factor, args.sequences, args.sigma, args.dilation_factor, args.lesion_db)
    else:
        augment_all_dataset_parallel(args.input_dir, args.output_dir, args.factor, args.sequences, args.sigma, args.dilation_factor, args.lesion_db)
    
    # path_to_lesion_instance_segmentation = "C:/Users/Admin/Downloads/sub-005_ses-01_mask-instances.nii.gz"
    # path_to_wm_mask = "C:/Users/Admin/Downloads/sub-005_wm_cropped.nii.gz"
    # paths_to_modalities = ["C:/Users/Admin/Downloads/sub-005_ses-01_reg-T2starw_FLAIR.nii.gz"]
    # factor = 1
    # output_path_segmentation = "C:/Users/Admin/Downloads/augmented_deleteme"
    # output_paths_modalities = "C:/Users/Admin/Downloads/augmented_deleteme"
    # sigma = 1
    # dilation_factor=2
    # lesion_db = "D:/R4/labels/lesion_database.xlsx"
    
    # if lesion_db:
    #     lesion_db = pd.read_excel(lesion_db)
    
    # data_augmentation_copy_paste(path_to_lesion_instance_segmentation, 
    #                               path_to_wm_mask,
    #                               paths_to_modalities, 
    #                               factor, 
    #                               output_path_segmentation,
    #                               output_paths_modalities, 
    #                               max_tries=20,
    #                               sigma=sigma,
    #                               dilation_factor=dilation_factor,
    #                               lesion_db = lesion_db,
    #                               subject="sub-005")
