import torch
import torchvision 
from .base_model import BaseModel
from . import networks
from .attention import SelfAttentionClassifier, LinearAttentionClassifier, calculate_heatmap
from random import randint

class ClassificationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.
    
        For classification we use a multilabel dataset. We add a parameter to specify the model architecture.
        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='multilabel')  # Classification model uses a multilabel dataset.
        parser.add_argument('--architecture', type=str, default='resnet50', help='Classification model architecture [resnet50]')  # You can define new arguments for this model.
        parser.add_argument('--pretrained', action='store_true', help='Load a classification model pretrained on imagenet')
        parser.add_argument('--attention', type=str, default=None, help='Specify the type of attention layer to use at the end of the network [None | self | linear]')
        if is_train:
            parser.set_defaults(use_val=True, lr_policy='cosine')
        else:
            parser.set_defaults(eval=True)
        return parser


    def get_network_by_name(self, opt):
        model = None

        if opt.architecture == 'resnet50':
            model = torchvision.models.resnet50(pretrained=opt.pretrained)

            if opt.attention is not None:
                # Get the Resnet50 Convolutional backbone.
                layers = []
                for name, children in model.named_children():
                    if (name is not 'fc') and (name is not 'avgpool'):
                        layers.append(children)
                
                backbone = torch.nn.Sequential(*layers)
                
                # Create the Resnet50 model with self attention.
                if opt.attention == 'self':
                    model = SelfAttentionClassifier(in_features=model.fc.in_features, label_num=opt.label_num, backbone=backbone)
                if opt.attention == 'linear':
                    model = LinearAttentionClassifier(in_features=model.fc.in_features, label_num=opt.label_num, backbone=backbone)
                else:
                    raise NotImplementedError('Attention model name [%s] is not recognized' % opt.attention)

                #print(model)
            else:
                # Update classification layer size
                fc_layer = torch.nn.Linear(model.fc.in_features, opt.label_num)
                model.fc = fc_layer

            # Update input size
            opt.crop_size = 224

        else:
            raise NotImplementedError('Classification model name [%s] is not recognized' % opt.architecture)

        # Initialize network for training.
        net = networks.init_net(model, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
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
        self.loss_names = ['train_loss', 'train_acc']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['Classification']

        # Visualize attention gates
        if self.opt.attention is not None:
            self.visual_names.append('heatmap')
        else:
        # If there is no attention remove the display option
            self.opt.no_display = True

        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netClassification = self.get_network_by_name(opt)

        if self.isTrain:  # only defined during training time
            # Define loss functions.
            self.criterionLoss = torch.nn.CrossEntropyLoss().to(self.device)
            # define and initialize optimizers.
            self.optimizer = torch.optim.SGD(self.netClassification.parameters(), lr=0.001, momentum=0.9)
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.data = input['data'].to(self.device)  # get image data
        self.label = input['label'].to(self.device)
        self.image_path = input['path']  # get image paths


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if self.opt.attention is not None:
            self.output, self.attention = self.netClassification(self.data)  # generate output image given the input data
        else:
            self.output = self.netClassification(self.data)  # generate output image given the input data

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_train_loss = self.criterionLoss(self.output, self.label)
        self.loss_train_loss.backward() # calculate gradients

        _, preds = torch.max(self.output, 1)
        corrects = torch.sum(preds == self.label)
        self.loss_train_acc = corrects.double() / len(preds)

    def get_predictions(self):
        """ Returns predictions and corresponding ground truth labels
        """
        # forward data
        with torch.no_grad():
            self.forward()

        # calculate prediction running corrects
        _, preds = torch.max(self.output, 1)

        return preds, self.label

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network existing gradients
        self.backward()              # calculate gradients for network
        self.optimizer.step()        # update gradients for network

    def compute_visuals(self):
        """Calculate additional output images for visualization"""
        
        # Choose a random sample from the batch
        if self.opt.attention is not None:
            idx = randint(0, self.opt.batch_size - 1)
            self.heatmap = calculate_heatmap(self.data, self.attention)
