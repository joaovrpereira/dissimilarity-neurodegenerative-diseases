import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def userInput():
    imgs = []
    img_objects = []
    while True: 
        print("Escreva um ID válido de um indivíduo (Ex: 30001): ")
        idIndividuo = input().strip()

        idIndividuoPattern = rf"^OAS{idIndividuo}_MR_d\d+$"
        pattern = re.compile(idIndividuoPattern)

        directory = "./oasis-images"

        matchingFolders = [f for f in os.listdir(directory) 
                           if os.path.isdir(os.path.join(directory, f)) and pattern.match(f)]
        if matchingFolders:
            print("Pastas encontradas:", matchingFolders)
            for folder in matchingFolders:
                nifti_path = os.path.join(directory, folder, "anat1", "NIFTI")
                if os.path.exists(nifti_path) and os.path.isdir(nifti_path):
                    nii_files = [f for f in os.listdir(nifti_path) if f.lower().endswith(".nii.gz")]
                    if nii_files:
                        
                        file_path = os.path.join(nifti_path, nii_files[0])
                        print("Carregando arquivo:", file_path)
                        img_obj = nib.load(file_path)
                        imgs.append(img_obj.get_fdata())
                        img_objects.append(img_obj)
                    else:
                        print(f"Nenhum arquivo .nii.gz encontrado em {nifti_path}")
                else:
                    print(f"Pasta NIFTI não encontrada em {folder}")
            if imgs:
                return imgs, img_objects
            else:
                print("Nenhum MRI carregado. Tente outro ID.")
        else: 
            print("Nenhuma pasta encontrada para o ID:", idIndividuo)

def plotMultipleMRIs(imgs, img_objects=None):
    n_mris = len(imgs)
    
    
    min_slices = min(img.shape[2] for img in imgs)
    slice_index = min_slices // 2 

    fig, axes = plt.subplots(1, n_mris, figsize=(5*n_mris, 5))

    if n_mris == 1:
        axes = [axes]

    img_plots = []
    current_slices = [slice_index] * n_mris 
    
    for i, ax in enumerate(axes):
       
        safe_slice = min(slice_index, imgs[i].shape[2] - 1)
        im = ax.imshow(imgs[i][:, :, safe_slice], cmap="gray")
        
       
        dims = imgs[i].shape
        ax.set_title(f"MRI {i+1}\nSlice {safe_slice}/{dims[2]-1}\nDims: {dims}")
        img_plots.append(im)
        ax.axis('off')

    def on_scroll(event):
        nonlocal current_slices
        
        for i, im in enumerate(img_plots):
            if event.button == 'up':
                current_slices[i] = min(current_slices[i] + 1, imgs[i].shape[2] - 1)
            elif event.button == 'down':
                current_slices[i] = max(current_slices[i] - 1, 0)
            
           
            im.set_data(imgs[i][:, :, current_slices[i]])
            axes[i].set_title(f"MRI {i+1}\nSlice {current_slices[i]}/{imgs[i].shape[2]-1}\nDims: {imgs[i].shape}")
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_key(event):
        nonlocal current_slices
        
        if event.key == 'right' or event.key == 'up':
            for i in range(n_mris):
                current_slices[i] = min(current_slices[i] + 1, imgs[i].shape[2] - 1)
        elif event.key == 'left' or event.key == 'down':
            for i in range(n_mris):
                current_slices[i] = max(current_slices[i] - 1, 0)
        
        for i, im in enumerate(img_plots):
            im.set_data(imgs[i][:, :, current_slices[i]])
            axes[i].set_title(f"MRI {i+1}\nSlice {current_slices[i]}/{imgs[i].shape[2]-1}\nDims: {imgs[i].shape}")
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

def main():
    imgs, img_objects = userInput()
    
    print("\nImagens carregadas:")
    for i, img in enumerate(imgs):
        print(f"MRI {i+1}: Dimensões {img.shape}")
    
    plotMultipleMRIs(imgs, img_objects)

if __name__ == "__main__":
    main()