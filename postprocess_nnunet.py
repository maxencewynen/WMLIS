import numpy as np
import nibabel as nib
import argparse
import os
import shutil
parser = argparse.ArgumentParser(description="Postprocess nnunet output to rename it & convert npz to nii.gz.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dir", type=str, required=True,  help="Directory containing the image files.")
args = parser.parse_args()


for filename in sorted(os.listdir(args.dir)):
    if not filename.endswith(".npz"):
        continue
    print(filename[:7])
    file = os.path.join(args.dir, filename)
    source_file = file.replace(".npz", ".nii.gz")

    npz = np.load(file)
    
    proba = npz['softmax'][1]

    img = nib.load(source_file)
    img_data = img.get_fdata()
    
    affine = img.affine
    proba = np.swapaxes(proba, 2, 0)
    assert proba.shape == img_data.shape
    out_file = file.replace("reg-T2starw_FLAIR.npz", "pred_prob.nii.gz")
    nib.save(nib.Nifti1Image(proba.astype(np.float32), affine), out_file)
    
    shutil.move(source_file, source_file.replace("reg-T2starw_FLAIR.nii.gz", "seg_binary.nii.gz"))


