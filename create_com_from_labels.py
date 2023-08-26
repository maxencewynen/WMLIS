import numpy as np
import nibabel as nib
from scipy.ndimage.measurements import center_of_mass

def process_nifty(file_path, output_dir):
    # Load the NIfTI file using nibabel
    nii = nib.load(file_path)
    data = nii.get_fdata()
    
    # Initialize new numpy matrices
    com_x = np.zeros(data.shape, dtype=np.float32)
    com_y = np.zeros(data.shape, dtype=np.float32)
    com_z = np.zeros(data.shape, dtype=np.float32)
    
    # For each unique lesion ID (excluding 0 as it's typically background)
    lesion_ids = np.unique(data)[1:]
    
    for lesion_id in lesion_ids:
        mask = (data == lesion_id)
        com = center_of_mass(mask)
        
        com_x[mask] = com[0]
        com_y[mask] = com[1]
        com_z[mask] = com[2]

    # Save the matrices as new NIfTI files in the output directory
    output_prefix = file_path.split('/')[-1].replace('.nii.gz', '')
    
    nib.save(nib.Nifti1Image(com_x, nii.affine), f"{output_dir}/{output_prefix}_com_x.nii.gz")
    nib.save(nib.Nifti1Image(com_y, nii.affine), f"{output_dir}/{output_prefix}_com_y.nii.gz")
    nib.save(nib.Nifti1Image(com_z, nii.affine), f"{output_dir}/{output_prefix}_com_z.nii.gz")

# Example usage
files = ['path/to/file1.nii.gz', 'path/to/file2.nii.gz'] # Add all your file paths here
output_dir = 'path/to/output_directory'

for file_path in files:
    process_nifty(file_path, output_dir)

