import argparse
import nibabel as nib
import os
import numpy as np
from os.path import join as pjoin
from pathlib import Path

def find_bounding_box(mask_data):
    """
    Finds the bounding box of the brain mask.
    """
    indices = np.where(mask_data > 0)
    x_min, x_max = np.min(indices[0]), np.max(indices[0])
    y_min, y_max = np.min(indices[1]), np.max(indices[1])
    z_min, z_max = np.min(indices[2]), np.max(indices[2])
    return x_min, x_max, y_min, y_max, z_min, z_max

def apply_mask(images_dir, masks_dir, output_dir, output_mask_dir, overwrite=False, list_of_directories=None):
    """
    Applies a binary mask from masks_dir for every file in images_dir that comes
    from the same subject and saves the resulting image in output_dir. A same subject can have several image files, the name of every image file should be preserved in the output
    images_dir has files that follow the naming pattern: "sub-xxx_ses-01_[...].nii.gz"
    masks_dir has files that follow the naming pattern: "sub-xxx_brainmask.nii.gz"
    overwrite: whether to overwrite the image file or not
    list_of_directories: list of all the directories in which there are files we want to apply the brainmask to. Default to None.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_mask_dir)
    list_of_directories = [Path(d) for d in list_of_directories]

    # Get a list of all mask files in masks_dir
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('_brainmask.nii.gz')])

    # Create the output directories if they don't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if output_mask_dir:
        os.makedirs(output_mask_dir, exist_ok=True)

    def process_subject_files(subject_files, output_dir):
        for subject_id, file_pairs in subject_files.items():
            for image_path, mask_path in file_pairs:
                output_filename = pjoin(output_dir, os.path.basename(image_path))
                
                # Check if the image file exists
                if not os.path.isfile(image_path):
                    print(f"Image file not found for {subject_id}. Skipping...")
                    continue

                # Check if the output file exists and overwrite is False
                if not overwrite and os.path.exists(output_filename):
                    print(f"Masked skipped for {subject_id} {image_path}")
                    continue

                # Load the image and mask data using nibabel
                image_nifti = nib.load(image_path)
                mask_nifti = nib.load(mask_path)

                try:
                    image_data = image_nifti.get_fdata()
                except EOFError as e:
                    print(f"Error while reading image file for {subject_id}: {e}")
                    print("Skipping this file")
                    continue

                mask_data = mask_nifti.get_fdata()

                # Find the bounding box of the brain mask
                x_min, x_max, y_min, y_max, z_min, z_max = find_bounding_box(mask_data)

                # Crop both the image and the mask using the bounding box
                cropped_image_data = image_data[x_min:x_max, y_min:y_max, z_min:z_max]
                cropped_mask_data = mask_data[x_min:x_max, y_min:y_max, z_min:z_max]

                # Apply the cropped brain mask to the cropped image
                masked_cropped_image_data = cropped_image_data * cropped_mask_data

                # Save the resulting cropped image with the same affine information as the input image
                output_nifti = nib.Nifti1Image(masked_cropped_image_data, affine=image_nifti.affine)
                nib.save(output_nifti, output_filename)
                
                if output_mask_dir:
                    output_mask_filename = pjoin(output_mask_dir, f"{subject_id}_brainmask.nii.gz")
                    # Save the resulting cropped brain mask with the same affine information as the input mask
                    output_mask_nifti = nib.Nifti1Image(cropped_mask_data, affine=mask_nifti.affine)
                    nib.save(output_mask_nifti, output_mask_filename)

                print(f"Mask applied and saved for {subject_id} {image_path}")
                print(f"New path: {subject_id} {output_filename}")

    # Process the mask files from the main directory
    subject_files = {}
    for mask_file in mask_files:
        # Extract the subject ID from the mask file name
        subject_id = mask_file.split('_')[0]

        # Find the corresponding image files for the subject
        image_files = [f for f in os.listdir(images_dir) if f.startswith(subject_id)]
        subject_files[subject_id] = [(pjoin(images_dir, image_file), pjoin(masks_dir, mask_file)) for image_file in image_files]

    process_subject_files(subject_files, output_dir)
    
    # Process the files from each directory in list_of_directories
    if list_of_directories:
        for directory in list_of_directories:
            if directory.endswith('/') or directory.endswith('\\'):
                directory = directory[:-1]
            print(directory)
            parent_dir = os.path.dirname(directory)
            print(parent_dir)
            parent_dir = parent_dir[:-1] if parent_dir[-1] == '/' or parent_dir[-1] == r"\\" else parent_dir
            os.makedirs(pjoin(parent_dir, f"{os.path.basename(directory)}_cropped"), exist_ok=True)
            out_dir = pjoin(parent_dir, f"{os.path.basename(directory)}_cropped")
            

            subject_files = {}
            for mask_file in mask_files:
                # Extract the subject ID from the mask file name
                subject_id = mask_file.split('_')[0]

                # Find the corresponding image files for the subject
                image_files = [f for f in os.listdir(directory) if f.startswith(subject_id)]
                subject_files[subject_id] = [(pjoin(directory, image_file), pjoin(masks_dir, mask_file)) for image_file in image_files]

            process_subject_files(subject_files, out_dir)

# Example usage:
# apply_mask("path_to_images_dir", "path_to_masks_dir", "path_to_output_dir", "path_to_output_mask_dir", list_of_directories=["/labels", "/lesions"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply binary masks to images from the same subject.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images_dir", type=str, help="Directory containing the image files.")
    parser.add_argument("--brainmasks_dir", type=str, help="Directory containing the mask files.")
    parser.add_argument("--output_dir", type=str, help="Output directory where masked images will be saved.")
    parser.add_argument("--output_brainmask_dir", type=str, default=None, help="Output directory where cropped brain masks will be saved.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Whether to overwrite files that already exist or not.")
    parser.add_argument("--list_of_directories", nargs='*', type=str, default=None, help="List of (ABSOLUTE PATHS to) directories containing the files to apply the brain mask to.")
    args = parser.parse_args()

    apply_mask(args.images_dir, args.brainmasks_dir, args.output_dir, args.output_brainmask_dir, args.overwrite, args.list_of_directories)
