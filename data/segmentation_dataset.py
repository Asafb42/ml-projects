import os
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image

class SegmentationDataset(BaseDataset):

    """
    This dataset class can load multi-class segmentation dataset.

    It requires a train/val/test directories to host images and segmentations in the format '/path/to/data/train/images', '/path/to/data/train/segmentations' etc. 
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        label_num is the number of classes in the multilabel segmentation task.
        The Default is one segmentation class.
        Returns:
            the modified parser.
        """
        parser.set_defaults(output_nc=1)  # The default segmentation dataset outputs a greyscale segmentation mask.
        parser.add_argument('--label_num', type=int, default=1, help='Num of labels for a multilabel segmentation task')
        parser.add_argument('--images_dir', type=str, default='volumes', help='The images directory name')
        parser.add_argument('--segmentation_dir', type=str, default='segmentations', help='The segmentations directory name')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        self.label_num = opt.label_num
        self.img_paths = []
        self.seg_paths = []

        phase_dir = os.path.join(opt.dataroot, opt.phase)
        self.img_dir_path = os.path.join(phase_dir, opt.images_dir)
        self.seg_dir_path = os.path.join(phase_dir, opt.segmentation_dir)

        # A list for all the images paths
        self.img_paths = sorted(make_dataset(self.img_dir_path, opt.max_dataset_size))

        # A list for all the segmentatgions paths
        self.seg_paths = sorted(make_dataset(self.seg_dir_path, opt.max_dataset_size))

        # Get the required transforms
        self.img_transform = get_transform(opt, grayscale=(opt.input_nc == 1))
        self.seg_transform = get_transform(opt, grayscale=(opt.output_nc == 1))

        # Get the full dataset size
        self.dataset_size = len(self.img_paths)

        # Set num of test images for test mode.
        if opt.phase == 'test':
            opt.num_test = self.dataset_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            img (tensor) - - an image in the input domain
            seg (tensor) - - its corresponding segmentation mask in the target domain
            img_paths (str) - - images paths
            seg_paths (str) - - segmentations paths
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        seg_path = self.seg_paths[index]

        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path).convert('L')

        # apply the same transform to both A and B

        img = self.img_transform(img)
        seg = self.seg_transform(seg)
        
        return {'img': img, 'seg': seg, 'img_paths': img_path, 'seg_paths': seg_path}

    def __len__(self):
        """Return the total number of images in the dataset, across all labels.

        """
        return self.dataset_size
