import argparse
import os
from util import util
class DirectoryOptions():
    """This class defines options used during dataset creation

    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        # Define parameters
        parser.add_argument('--source_dir', required=True, help='path to the source image directory, containing subdirectories for each class')
        parser.add_argument('--dest_dir', type=str, default='./dataset', help='path to the destination directory')
        parser.add_argument('--val', type=int, default=0, choices=range(0,101), help='precentage [0-100] of validation split of the data. Default is no validation split')
        parser.add_argument('--test', type=int, default=0, choices=range(0,101), help='precentage [0-100] of test split of the data. Default is no test split')
        parser.add_argument('--use_suffix', action='store_true', help='Arrange the dataset by trainA, trainB, testA, testB, etc.')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with options(only once).
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [dest_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if os.path.exists(opt.dest_dir):
            raise ValueError("Dataset directory {} already exists".format(opt.dest_dir))

        util.mkdirs(opt.dest_dir)
        file_name = os.path.join(opt.dest_dir, 'dir_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create destination directory."""
        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt
        return self.opt
