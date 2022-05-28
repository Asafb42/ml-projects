
import numpy as np
import os    # Traverse folders 
import imageio   # Convert to an image 
from PIL import Image
import shutil


if __name__ == '__main__':
    
    folderpath = '..\..\Datasets\LITS_organized_curated'
    

    for top_dir in os.listdir(folderpath):
        
        progress = 0
        segmentations_path = os.path.join(folderpath,top_dir,"segmentations")
        volumes_path = os.path.join(folderpath,top_dir,"volumes")
        livermasks_path = os.path.join(folderpath,top_dir,"livermasks")
        lesionmasks_path = os.path.join(folderpath,top_dir,"lesionmasks")

        segmentations_num = len(os.listdir(segmentations_path))

        for file in os.listdir(segmentations_path):
            mask = imageio.imread(os.path.join(segmentations_path, file))
            
            # remove data with empty segmentation mask
            if np.max(mask) == 0:
                volume_name = "volume-" + file.split('-')[1]
                livermask_name = file.split('_')[0] + "_livermask_" + file.split('_')[1]
                lesionmask_name = file.split('_')[0] + "_lesionmask_" + file.split('_')[1]

                os.remove(os.path.join(segmentations_path, file))
                os.remove(os.path.join(volumes_path, volume_name))
                os.remove(os.path.join(livermasks_path, livermask_name))
                os.remove(os.path.join(lesionmasks_path, lesionmask_name))

            progress = progress + 1
            print("{} progress: {}/{}".format(top_dir, progress, segmentations_num))