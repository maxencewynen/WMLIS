import os
from os.path import join as pjoin
import random
import shutil
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# save options
parser.add_argument('--sourcedir', type=str, required=True,
                    help='Specify the path to the source directory including sourcedir/images and sourcedir/labels')

parser.add_argument('--targetdir', type=str, required=True, help='Specify the path to the target directory')
parser.add_argument('--other_directories', type=str, nargs="+", required=False, default=[],
                        help='Specify the names of other directories that also have to be classified into train val and test')

parser.add_argument('--ignore_images', action='store_true', default=False, help="Whether to ignore the images folder")
parser.add_argument('--ignore_labels', action='store_true', default=False, help="Whether to ignore the labels folder")
parser.add_argument('--ignore_brainmasks', action='store_true', default=False, help="Whether to ignore the brainmasks folder")
parser.add_argument('--seed', type=int, default=42, help="Random seed")

args = parser.parse_args()

print(f"Other directories : {args.other_directories}")

# Set the paths to the source and destination directories
images_source_dir = pjoin(args.sourcedir, "images")
labels_source_dir = pjoin(args.sourcedir, "labels")
brainmasks_source_dir = pjoin(args.sourcedir, "brainmasks")
train_dest_dir = pjoin(args.targetdir, "train")
val_dest_dir = pjoin(args.targetdir, "val")
test_dest_dir = pjoin(args.targetdir, "test")
sourcedirs=[]
targetdirs=[]
for d in args.other_directories:
    sourcedirs.append(pjoin(args.sourcedir, d))
    targetdirs.append(pjoin(args.targetdir, d))

# List of subjects
subj = sorted(list(set([s[:8] for s in os.listdir(images_source_dir) ])))
print(f"List of subjects: \n{subj}")

# Set the random seed for reproducibility
random.seed(args.seed)

# Split the subjects into train, val, and test sets
train_subjects, remaining_subjects = train_test_split(subj, test_size=0.4, random_state=3)
val_subjects, test_subjects = train_test_split(remaining_subjects, test_size=0.5, random_state=3)

print("*"*20)
print(f"Training subjects ({len(train_subjects)}): {train_subjects}")
print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
print("*"*20)
input('Press Enter to continue')

# Create destination directories if they don't exist
os.makedirs(os.path.join(train_dest_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dest_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(train_dest_dir, "brainmasks"), exist_ok=True)
os.makedirs(os.path.join(val_dest_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dest_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(val_dest_dir, "brainmasks"), exist_ok=True)
os.makedirs(os.path.join(test_dest_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dest_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(test_dest_dir, "brainmasks"), exist_ok=True)
for d in args.other_directories:
    os.makedirs(pjoin(train_dest_dir, d), exist_ok=True)
    os.makedirs(pjoin(val_dest_dir, d), exist_ok=True)
    os.makedirs(pjoin(test_dest_dir, d), exist_ok=True)

# Move subject images and labels to the appropriate directories
for subject in subj:
    print(subject)
    # Determine the partition based on the subject
    if subject in train_subjects:
        partition = train_dest_dir
    elif subject in val_subjects:
        partition = val_dest_dir
    else:
        partition = test_dest_dir
    
    if not args.ignore_images:
        # Move images
        for filename in os.listdir(images_source_dir):
            if subject in filename and filename.endswith(".nii.gz"):
                src_path = os.path.join(images_source_dir, filename)
                dest_path = os.path.join(partition, "images", filename)
                shutil.copy(src_path, dest_path)
    if not args.ignore_labels:
        # Move labels
        for filename in os.listdir(labels_source_dir):
            if subject in filename and filename.endswith(".nii.gz"):
                src_path = os.path.join(labels_source_dir, filename)
                dest_path = os.path.join(partition, "labels", filename)
                shutil.copy(src_path, dest_path)
    if not args.ignore_brainmasks:
        # Move brainmasks
        for filename in os.listdir(brainmasks_source_dir):
            if subject in filename and filename.endswith(".nii.gz"):
                src_path = os.path.join(brainmasks_source_dir, filename)
                dest_path = os.path.join(partition, "brainmasks", filename)
                shutil.copy(src_path, dest_path)

    for i, (sdir, tdir) in enumerate(zip(sourcedirs, targetdirs)):
        for filename in os.listdir(sdir):
            if subject in filename and filename.endswith(".nii.gz"):
                src_path = os.path.join(sdir, filename)
                dest_path = os.path.join(partition, args.other_directories[i], filename)
                shutil.copy(src_path, dest_path)


