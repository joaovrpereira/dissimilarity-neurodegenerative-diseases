import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt



img = nib.load("./oasis-images/OAS30001_MR_d0129/anat1/NIFTI/sub-OAS30001_ses-d0129_acq-TSE_T2w.nii.gz")
data = img.get_fdata()
print(data.shape)

slice_index = data.shape[2] // 2

fig, ax = plt.subplots()
img_plot = ax.imshow(data[:, :, slice_index], cmap="gray")
ax.set_title(f"Slice {slice_index}")

def on_scroll(event):
    global slice_index
    if event.button == 'up':
        slice_index = (slice_index + 1) % data.shape[2]
    else:
        slice_index = (slice_index - 1) % data.shape[2]
    img_plot.set_data(data[:, :, slice_index])
    ax.set_title(f"Slice {slice_index}")
    fig.canvas.draw()

fig.canvas.mpl_connect('scroll_event', on_scroll)
plt.show()