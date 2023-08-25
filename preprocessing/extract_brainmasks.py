import nibabel as nib
# import the BrainExtractor class
from brainextractor import BrainExtractor
import os
from os.path import join as pjoin

source_dir = r"/data/all/images"
target_dir = r"/data/all/brainmasks"

# read in the image file first
filenames = sorted([f for f in os.listdir(source_dir) if "FLAIR" in f])

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for filename in filenames:
    print("\n" + "*"*20 + "\n")
    print(filename)
    input_img = nib.load(pjoin(source_dir,filename))
    # create a BrainExtractor object using the input_img as input
    # we just use the default arguments here, but look at the
    # BrainExtractor class in the code for the full argument list
    bet = BrainExtractor(img=input_img)

    # run the brain extraction
    # this will by default run for 1000 iterations
    # I recommend looking at the run method to see how it works
    bet.run()


    # save the computed mask out to file
    bet.save_mask(pjoin(target_dir, filename[:7]+"_brainmask.nii.gz"))
