"""
Script to create data directories in accordance to the expected dataroot of the main train/test scripts.
The user can choose whether to create train, test and validation data directories from a given source or multiple source directories. 
The source directory should contain a subdirectory for each class in the dataset, even if the dataset contains a single class. 
The user can also choose the train/val/test split ratio.
The default directory format will have train, test, val directories and a subdirectory for each class.
If the user choose the use_suffix option the directory will be arrange in the trainA, trainB, testA, testB, etc. format.
"""
from options.directory_options import DirectoryOptions
from util import util
import splitfolders
import shutil
import os


def remove_empty_folders(path):
    """Function to remove empty folders"""
    if not os.path.isdir(path):
        return

    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                remove_empty_folders(fullpath)

    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0:
        os.rmdir(path)


def arrange_dir(dataroot):

    # Add the suffix A, B, etc. to the folders and move them to the top level. 
    for top_dir in os.listdir(dataroot):
        suffix = 'A'
        top_path = os.path.join(dataroot, top_dir)

        if os.path.isdir(top_path):
            for sub_dir in os.listdir(top_path):
                # Move the sub-folder to the root directory and change it's according to the suffix. 
                shutil.move(os.path.join(top_path, sub_dir), os.path.join(dataroot, top_dir + suffix))
                suffix = chr(ord(suffix) + 1); # Update suffix.

    # Remove all the remoaning excess folders.
    remove_empty_folders(dataroot)


if __name__ == '__main__':
    opt = DirectoryOptions().parse() # get training options
    
    train_ratio = 100 - opt.val - opt.test # calculate train ratio
    if train_ratio < 0:
        raise ValueError("Invalid train ratio: {}.".format(train_ratio))

    split_ratio = (train_ratio / 100, opt.val / 100, opt.test / 100)
    
    splitfolders.ratio(opt.source_dir, output=opt.dest_dir, ratio=split_ratio)
    
    if opt.use_suffix:
        arrange_dir(opt.dest_dir)
