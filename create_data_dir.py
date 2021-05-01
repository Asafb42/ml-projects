"""
Script to create data directories (trainA, trainB, testA, testB, etc.) in accordance to the expected dataroot of the main train/test scripts.
The user can choose whether to create train, test and validation data directories from a given source or multiple source directories. 
The user can also choose the train/val/test split ratio.
"""
from options.directory_options import DirectoryOptions
from util import util
import splitfolders
import shutil
import os


def arrange_dir(dataroot, suffix):
    for (root, dirs, files) in os.walk('dataroot', topdown=True):
        # remove empty folders.
        if not files:
            os.rmdir(root)
        # if the folder is not empty, add the suffix to the folder name.
        else:
            os.rename(root, os.pathroot + suffix)


if __name__ == '__main__':
    opt = DirectoryOptions().parse() # get training options
    
    train_ratio = 100 - opt.val - opt.test # calculate train ratio
    if train_ratio < 0:
        raise ValueError("Invalid train ratio: {}.".format(train_ratio))

    split_ratio = (train_ratio / 100, opt.val / 100, opt.test / 100)
    
    splitfolders.ratio(opt.source_dir_A, output=opt.dest_dir, ratio=split_ratio)
    arrange_dir(opt.dest_dir, 'A')

    splitfolders.ratio(opt.source_dir_B, output=opt.dest_dir, ratio=split_ratio)
    arrange_dir(opt.dest_dir, 'B')
