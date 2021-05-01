"""
Script to create data directories (trainA, trainB, testA, testB, etc.) in accordance to the expected dataroot of the main train/test scripts.
The user can choose whether to create train, test and validation data directories from a given source or multiple source directories. 
The source directory should contain a subdirectory for each class in the dataset, even if the dataset contains a single class. 
The user can also choose the train/val/test split ratio.
"""
from options.directory_options import DirectoryOptions
from util import util
import splitfolders
import shutil
import os


def arrange_dir(dataroot):
    for (root, dirs, files) in os.walk('dataroot', topdown=True):
        # remove empty folders.
        if len(files):
            os.rmdir(root)

if __name__ == '__main__':
    opt = DirectoryOptions().parse() # get training options
    
    train_ratio = 100 - opt.val - opt.test # calculate train ratio
    if train_ratio < 0:
        raise ValueError("Invalid train ratio: {}.".format(train_ratio))

    split_ratio = (train_ratio / 100, opt.val / 100, opt.test / 100)
    
    splitfolders.ratio(opt.source_dir, output=opt.dest_dir, ratio=split_ratio)
    arrange_dir(opt.dest_dir)
