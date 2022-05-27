
import numpy as np
import os    # Traverse folders 
import imageio   # Convert to an image 
from PIL import Image
import shutil


if __name__ == '__main__':
    
    folderpath = '..\..\Datasets\LITS_organized'
    filepath = os.path.join(folderpath, 'segmentations')
    imgfile = os.path.join(folderpath, 'image_dataset/temp')
    
    for top_dir in os.listdir(folderpath):
        
        livermasks_path = os.path.join(folderpath,top_dir,"livermasks")
        lesionmasks_path = os.path.join(folderpath,top_dir,"lesionmasks")

        if not os.path.exists(livermasks_path):
            os.mkdir(livermasks_path)

        if not os.path.exists(lesionmasks_path):
            os.mkdir(lesionmasks_path)

        for mid_dir in os.listdir(os.path.join(folderpath,top_dir)):
            class_path = os.path.join(folderpath,top_dir,mid_dir)
            
            if (mid_dir=='volumes'):
                for vol_dir in os.listdir(os.path.join(folderpath,top_dir,mid_dir)):
                    vol_path = os.path.join(class_path, vol_dir)
                    print(vol_path)
                    for file in os.listdir(vol_path):
                        shutil.move(os.path.join(vol_path, file), class_path)

            if (mid_dir=='segmentations'):
                for seg_dir in os.listdir(os.path.join(folderpath,top_dir,mid_dir)):
                    seg_path = os.path.join(class_path, seg_dir)
                    print(seg_path)
                    for file in os.listdir(seg_path):

                        if file.split('_')[1] == 'livermask':
                            shutil.move(os.path.join(seg_path, file), livermasks_path)

                        elif file.split('_')[1] == 'lesionmask':
                            shutil.move(os.path.join(seg_path, file), lesionmasks_path)

                        else:   
                            shutil.move(os.path.join(seg_path, file), class_path)
