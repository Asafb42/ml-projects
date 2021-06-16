import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

class MultilabelDataset(BaseDataset):

    """
    This dataset class can load multilabeled datasets.

    It requires a train directories to host training images from each of the classes in the format '/path/to/data/trainA', '/path/to/data/trainB' etc. 
    And a test directories to host test images from each class in the format '/path/to/data/testA', '/path/to/data/testB' etc. 
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        label_num is the number of classes in the multilabel classification task. It's defaults to the binary case.
        Returns:
            the modified parser.
        """
        parser.add_argument('--label_num', type=int, default=2, help='Num of labels for a multilabel classification task')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        self.label_num = opt.label_num
        self.dirs  = []
        self.paths = []
        self.sizes = []

        for i in range(self.label_num):
            # A list for all the labels directories
            label_letter_index = chr(ord('A') + i)
            self.dirs.append(os.path.join(opt.dataroot, opt.phase + label_letter_index))

            # A list for all the labels images paths
            self.paths.append(sorted(make_dataset(self.dirs[i], opt.max_dataset_size)))

            # A list of the sizes of each class dataset
            self.sizes.append(len(self.paths[i]))

        # Get the required transforms
        self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))

        # Get the full dataset size
        self.dataset_size = sum(self.sizes)

        # Set num of test images for test mode.
        if opt.phase == 'test':
            opt.num_test = self.dataset_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains image, label and path
            data (tensor)   --  an input image
            label (int)      -- image label
            path  (str)      -- image path
        """
        
        # randomize the label index.
        label = random.randint(0, self.label_num - 1)

        # Get the path and image
        path = self.paths[label][index % self.sizes[label]]  # make sure index is within the range
        img = Image.open(path).convert('RGB')

        # apply image transformation.
        img_transformed = self.transform(img)

        return {'data': img_transformed, 'label': label, 'path': path}

    def __len__(self):
        """Return the total number of images in the dataset, across all labels.

        """
        return self.dataset_size
