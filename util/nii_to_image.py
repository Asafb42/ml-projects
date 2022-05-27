
from joblib import PrintTime
import numpy as np
import os    # Traverse folders 
import nibabel as nib #nii Format 1 This bag will be used in general 
import imageio   # Convert to an image 
from PIL import Image

def nii_to_image(niifile, mask=0):
    filenames = os.listdir(filepath) # Read nii Folder 
    slice_trans = []
    
    progress = 0
    for f in filenames:
        # Start reading nii Documents 
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)    # Read nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii','')   # Remove nii Suffix name of 
        img_f_path = os.path.join(imgfile, fname)
        # Create nii The folder of the corresponding image 
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)    # New Folder 
 
        # Start converting to an image 
        (x,y,z) = img.shape

        for i in range(z):      #z Is a sequence of images 
            slice = img_fdata[:, :, i]   # You can choose which direction of slice 
            slice = np.rot90(slice)

            if mask:
                livermask = np.where(slice > 0 , 255, 0).astype(np.uint8)
                lesionmask = np.where(slice == 2, 255, 0).astype(np.uint8)
                slice = livermask/2 + lesionmask/2
                slice = np.ceil(slice).astype(np.uint8)
                
                # Save masks 
                imageio.imwrite(os.path.join(img_f_path,'{}_livermask_{}.png'.format(fname,i)), livermask)
                imageio.imwrite(os.path.join(img_f_path,'{}_lesionmask_{}.png'.format(fname,i)), lesionmask)
            
            else:
                # Normalize image
                if (np.min(slice) < 0):
                   slice = slice - np.min(slice)
                
                if (np.max(slice) != 0):
                   slice = slice/np.max(slice)*65535

                slice = slice.astype(np.int16)

            # Save an image 
            imageio.imwrite(os.path.join(img_f_path,'{}_{}.png'.format(fname,i)), slice)
            Image.fromarray(slice).save(os.path.join(img_f_path,'{}_{}.tif'.format(fname,i)))

        progress = progress + 1
        print('Done {}/{} in path {}'.format(progress,len(filenames), filepath))

if __name__ == '__main__':
    
    folderpath = '..\..\Datasets\LITS'
    filepath = os.path.join(folderpath, 'segmentations')
    imgfile = os.path.join(folderpath, 'image_dataset/temp')
    nii_to_image(filepath, mask=1)

    filepath = os.path.join(folderpath, 'volumes')
    imgfile = os.path.join(folderpath, 'image_dataset/temp')
    nii_to_image(filepath, mask=0)