import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load("./oasis-images/OAS30001_MR_d0129/anat1/NIFTI/sub-OAS30001_ses-d0129_acq-TSE_T2w.nii.gz")
data = img.get_fdata()
print(data.shape)

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice0 = data[100, :,:]
slice1 = data[:, 100,:]
slice2 = data[:, :,13]
show_slices([slice0, slice1, slice2])
plt.suptitle("Center slices for anatomical image") 

plt.show()