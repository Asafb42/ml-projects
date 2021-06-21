import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

class MultilabelDataset(BaseDataset):

    """
    This dataset class can load multilabeled datasets.

    It requires a train/val/test directories to host training images from each of the classes in the format '/path/to/data/train/labelA', '/path/to/data/train/labelB' etc. 
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        label_num is the number of classes in the multilabel classification task and it is tehe same as the number of subdirectories in the dataroot.
        It's defaults to the binary case.
        Returns:
            the modified parser.
        """
        parser.add_argument('--label_num', type=int, default=2, help='Num of labels for a multilabel classification task')
        parser.add_argument('--labelA', type=str, default=None, help='The name of the first label for a binary classification')
        parser.add_argument('--labelB', type=str, default=None, help='The name of the second label for a binary classification')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        self.label_num = opt.label_num
        self.labels = []
        self.dirs  = []
        self.paths = []
        self.sizes = []

        phase_dir = os.path.join(opt.dataroot, opt.phase)
        
        # Get the class labels
        if (self.label_num == 2) and (opt.labelA) and (opt.labelB):
            self.labels = [opt.labelA, opt.labelB]
        else:
            for class_name in os.listdir(phase_dir):
                class_dir = os.path.join(phase_dir, class_name)
                if os.path.isdir(class_dir):
                    self.labels.append(class_name)

        # Remove excess labels. Taking only the first label_num labels by alphabetical order.
        self.labels = self.labels[:self.label_num]

        for i in range(self.label_num):
            # A list for all the labels directories
            self.dirs.append(os.path.join(phase_dir, self.labels[i]))

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
