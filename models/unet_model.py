"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from .Unet_networks.models import UNet, UNet_2Plus, UNet_3Plus
from .Unet_networks.loss import bceLoss, iouLoss, msssimLoss

class UnetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='segmentation')  # Unet model uses a segmentation dataset.
        if is_train:
            parser.add_argument('--unet_type', type=str, default='unet', help='Specify the type of Unet to use [unet | unet_2plus | unet_3plus | unet_3plus_deepsup | unet_3plus_deepsup_cgm]')

        return parser


    def get_network_by_name(self, opt):

        # Create the UNet model.
        if opt.unet_type == 'unet':
            model = UNet.UNet(in_channels=opt.input_nc, n_classes=opt.label_num, is_deconv=True, is_batchnorm=True)
        elif opt.unet_type == 'unet_2plus':
            model = UNet_2Plus.UNet_2Plus(in_channels=opt.input_nc, n_classes=opt.label_num, is_deconv=True, is_batchnorm=True)
        elif opt.unet_type == 'unet_3plus':
            model = UNet_3Plus.UNet_3Plus(in_channels=opt.input_nc, n_classes=opt.label_num, is_deconv=True, is_batchnorm=True)
        elif opt.unet_type == 'unet_3plus_deepsup':
            model = UNet_3Plus.UNet_3Plus_DeepSup(in_channels=opt.input_nc, n_classes=opt.label_num, is_deconv=True, is_batchnorm=True)
        elif opt.unet_type == 'unet_3plus_deepsup_cgm':
            model = UNet_3Plus.UNet_3Plus_DeepSup_CGM(in_channels=opt.input_nc, n_classes=opt.label_num, is_deconv=True, is_batchnorm=True)      
        else:
            raise NotImplementedError('Unet model name [%s] is not recognized' % opt.unet_type)

        # Update input size
        #opt.crop_size = 320
        
        # Initialize network for training.
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            model.to(self.gpu_ids[0])
            net = torch.nn.DataParallel(model, self.gpu_ids)  # multi-GPUs
        return net

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['IOU', 'BCE']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['img', 'seg']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['Unet']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netUnet = self.get_network_by_name(opt)
        if self.isTrain:  # only defined during training time
            #self.criterionIOU = iouLoss.IOU_loss()
            #self.criterionBCE = bceLoss.BCE_loss()

            # define and initialize optimizers. You can define one optimizer for each network.
            #self.optimizer = torch.optim.Adam(self.netUnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer = torch.optim.SGD(self.netUnet.parameters(), lr=0.001, momentum=0.9)

            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.img = input['img'].to(self.device)  # get image data
        self.seg = input['seg'].to(self.device)  # get segmentation data
        self.image_paths = input['img_paths']    # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = self.netUnet(self.img)  # generate output segmentation given the input image

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_IOU = iouLoss.IOU_loss(self.output, self.seg)
        #self.loss_IOU.backward()       # calculate gradients of the network w.r.t. IOU loss

        self.loss_BCE = bceLoss.BCE_loss(self.output, self.seg)
        self.loss_BCE.backward()       # calculate gradients of the network w.r.t. BCE loss

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
