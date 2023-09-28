import os
import pandas as pd
from os.path import join as pjoin
import nibabel as nib
import numpy as np

labels_dir = "/home/mwynen/data/cusl_wml/labels"
infos = []

for filename in os.listdir(labels_dir):
    if "mask-instances.nii.gz" not in filename:
        continue
    subject = filename.split("_")[0]
    img = nib.load(pjoin(labels_dir, filename))
    img_data = img.get_fdata()
    pixdim = img.header["pixdim"][1] * img.header["pixdim"][2] * img.header["pixdim"][3]
    labels, counts = np.unique(img_data, return_counts=True)

    for label, count in zip(labels, counts):
        if label <= 0:
            continue
        infos.append({"subject": subject,
                      "lesion_id": label,
                      "voxel_size": count,
                      "size": pixdim * count,
                      "PRL": label < 2000})

# df = pd.DataFrame(columns=["subject", "lesion_id", "voxel_size", "PRL"])
df = pd.DataFrame(data=infos)
df.to_csv(r"/home/mwynen/data/cusl_wml/lesions.csv")
print(df)

